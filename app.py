from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
from sqlalchemy import select, create_engine
import os

from models import db, CityData

app = Flask(__name__)

# Specify the absolute path for the database
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'dengue_data.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database with the app
db.init_app(app)

from dengue_model import (
    generate_fictional_data,
    engineer_features,
    create_graph,
    DengueGNN,
    train_test_split_graph,
    generate_explanation,
    train_model,
    evaluate_model
)

# Global variables to store the model and data
model = None
graph_data = None
feature_scaler = None
city_names = []

# Create database engine
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

def initialize_database():
    with app.app_context():
        db.create_all()
        global city_names
        city_names = [city[0] for city in db.session.query(CityData.city).distinct()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    with app.app_context():
        total_cities = db.session.query(CityData.city).distinct().count()
        total_cases = db.session.query(db.func.sum(CityData.dengue_cases)).scalar() or 0
        avg_temperature = db.session.query(db.func.avg(CityData.temperature)).scalar() or 0
        avg_rainfall = db.session.query(db.func.avg(CityData.rainfall)).scalar() or 0

    return render_template('dashboard.html', 
                           total_cities=total_cities,
                           total_cases=total_cases,
                           avg_temperature=avg_temperature,
                           avg_rainfall=avg_rainfall)

@app.route('/data')
def data_management():
    with app.app_context():
        sample_data = CityData.query.limit(10).all()
    return render_template('data_management.html', sample_data=sample_data)

@app.route('/model')
def model_management():
    global model
    model_status = "Trained" if model is not None else "Not trained"
    return render_template('model_management.html', model_status=model_status)

@app.route('/generate_data', methods=['POST'])
def generate_data():
    if CityData.query.first():
        return jsonify({"message": "Data already exists in the database"})
    
    city_data = generate_fictional_data()
    
    for _, row in city_data.iterrows():
        city_record = CityData(
            city=row['city'],
            date=row['date'],
            temperature=row['temperature'],
            rainfall=row['rainfall'],
            population=row['population'],
            dengue_cases=row['dengue_cases']
        )
        db.session.add(city_record)
    
    db.session.commit()
    
    global city_names
    city_names = [city[0] for city in db.session.query(CityData.city).distinct()]
    
    return jsonify({"message": "Fictional data generated and stored in the database successfully"})

@app.route('/train', methods=['POST'])
def train():
    global model, graph_data, feature_scaler, city_names, engine
    
    stmt = select(CityData)
    with engine.connect() as connection:
        result = connection.execute(stmt)
        city_data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if city_data.empty:
        return jsonify({"error": "No data available. Please generate data first."})
    
    city_data = engineer_features(city_data)
    graph_data, feature_scaler = create_graph(city_data)
    
    train_data, test_data = train_test_split_graph(graph_data, test_size=0.2, random_state=42)
    
    model = train_model(train_data)
    
    mse, r2 = evaluate_model(model, test_data)
    
    city_names = city_data['city'].unique().tolist()
    
    return jsonify({
        "message": "Model trained successfully",
        "performance": {
            "mse": float(mse),  # Convert to standard Python float
            "r2": float(r2)     # Convert to standard Python float
        }
    })

@app.route('/plot', methods=['GET'])
def plot():
    stmt = select(CityData)
    with engine.connect() as connection:
        result = connection.execute(stmt)
        city_data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if city_data.empty:
        return jsonify({"error": "No data available. Please generate data first."})
    
    plt.figure(figsize=(10, 6))
    avg_cases = city_data.groupby('date')['dengue_cases'].mean()
    plt.plot(avg_cases.index, avg_cases.values)
    plt.xlabel('Date')
    plt.ylabel('Average Dengue Cases')
    plt.title('Average Dengue Cases Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    
    return jsonify({"plot": plot_data})

@app.route('/predict', methods=['POST'])
def predict():
    global model, graph_data, feature_scaler, city_names
    
    if not city_names:
        return jsonify({"error": "No cities available. Please generate data and train the model first."})
    
    city_name = request.form['city_name']
    if city_name not in city_names:
        return jsonify({"error": f"City '{city_name}' not found in the database."})
    
    city_index = city_names.index(city_name)
    
    if model is None:
        return jsonify({"error": "Model not trained. Please train the model first."})
    
    with torch.no_grad():
        prediction = model(graph_data.x, graph_data.edge_index, None)[city_index].item()
    
    features = graph_data.x[city_index].cpu().numpy()
    feature_dict = {
        'rainfall': feature_scaler.inverse_transform(features.reshape(1, -1))[0, 1],
        'temperature': feature_scaler.inverse_transform(features.reshape(1, -1))[0, 0],
        'population': feature_scaler.inverse_transform(features.reshape(1, -1))[0, 2]
    }
    
    thresholds = {
        'high_risk': 50,
        'moderate_risk': 20,
        'heavy_rainfall': 100,
        'high_temperature': 30,
        'high_population_density': 1000000
    }
    
    explanation = generate_explanation(city_name, prediction, feature_dict, thresholds)
    
    return jsonify({
        "prediction": prediction,
        "explanation": explanation
    })


@app.route('/get_cities', methods=['GET'])
def get_cities():
    global city_names
    return jsonify({"cities": city_names})

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True)