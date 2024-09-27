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
from geopy.distance import geodesic
from flask_migrate import Migrate
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

from models import db, CityData

app = Flask(__name__)
migrate = Migrate(app, db)

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
model_status = "Not trained"
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
    page = request.args.get('page', 1, type=int)
    per_page = 10  # You can adjust this number
    city_filter = request.args.get('city', '')
    date_filter = request.args.get('date', '')

    query = CityData.query

    if city_filter:
        query = query.filter(CityData.city.ilike(f'%{city_filter}%'))
    if date_filter:
        query = query.filter(CityData.date == date_filter)

    pagination = query.order_by(CityData.date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    data = pagination.items

    return render_template('data_management.html', 
                           data=data, 
                           pagination=pagination,
                           city_filter=city_filter,
                           date_filter=date_filter)

@app.route('/reset_data', methods=['POST'])
def reset_data():
    try:
        # Delete all existing data
        db.session.query(CityData).delete()
        db.session.commit()

        # Generate new data
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
        
        return jsonify({"message": "Data reset successfully. New data generated and stored in the database."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

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
            dengue_cases=row['dengue_cases'],
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        db.session.add(city_record)
    
    db.session.commit()
    
    global city_names
    city_names = [city[0] for city in db.session.query(CityData.city).distinct()]
    
    return jsonify({"message": "Fictional data generated and stored in the database successfully"})

@app.route('/train', methods=['POST'])
def train():
    global model, graph_data, feature_scaler, city_names, engine, model_status
    
    stmt = select(CityData)
    with engine.connect() as connection:
        result = connection.execute(stmt)
        city_data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if city_data.empty:
        return jsonify({"error": "No data available. Please generate data first."})
    
    print("Sample of city_data before engineering features:")
    print(city_data.head())
    print(city_data.dtypes)
    
    city_data = engineer_features(city_data)
    
    print("Sample of city_data after engineering features:")
    print(city_data.head())
    print(city_data.dtypes)
    
    graph_data, feature_scaler = create_graph(city_data)
    
    train_data, test_data = train_test_split_graph(graph_data, test_size=0.2, random_state=42)
    
    model = train_model(train_data)
    
    mse, r2 = evaluate_model(model, test_data)
    
    city_names = city_data['city'].unique().tolist()
    
    model_status = "Trained"
    
    return jsonify({
        "message": "Model trained successfully",
        "performance": {
            "mse": float(mse),
            "r2": float(r2)
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

def fetch_city_data(city_name, weeks=12):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(weeks=weeks)
    
    stmt = select(CityData).where(
        (CityData.city == city_name) &
        (CityData.date >= start_date) &
        (CityData.date <= end_date)
    )
    with engine.connect() as connection:
        result = connection.execute(stmt)
        data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    data['week'] = pd.to_datetime(data['date']).dt.isocalendar().week
    return data

def calculate_thresholds(data):
    return {
        'high_risk': np.percentile(data['dengue_cases'], 75),
        'moderate_risk': np.percentile(data['dengue_cases'], 50),
        'heavy_rainfall': np.percentile(data['rainfall'], 75),
        'high_temperature': np.percentile(data['temperature'], 75),
    }

def perform_statistical_analysis(data):
    # Basic statistics
    stats_dict = {
        'mean_cases': data['dengue_cases'].mean(),
        'median_cases': data['dengue_cases'].median(),
        'std_cases': data['dengue_cases'].std(),
        'mean_temperature': data['temperature'].mean(),
        'mean_rainfall': data['rainfall'].mean(),
    }
    
    # Correlation analysis
    corr_temp = stats.pearsonr(data['temperature'], data['dengue_cases'])
    corr_rain = stats.pearsonr(data['rainfall'], data['dengue_cases'])
    stats_dict['temp_correlation'] = corr_temp[0]
    stats_dict['temp_correlation_p'] = corr_temp[1]
    stats_dict['rain_correlation'] = corr_rain[0]
    stats_dict['rain_correlation_p'] = corr_rain[1]
    
    # Trend analysis
    trend = seasonal_decompose(data['dengue_cases'], model='additive', period=4).trend
    stats_dict['trend_start'] = trend.iloc[0]
    stats_dict['trend_end'] = trend.iloc[-1]
    stats_dict['trend_direction'] = 'Increasing' if trend.iloc[-1] > trend.iloc[0] else 'Decreasing'
    
    return stats_dict

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
        prediction = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)[city_index].item()
    
    features = graph_data.x[city_index].cpu().numpy()
    feature_dict = {
        'rainfall': feature_scaler.inverse_transform(features.reshape(1, -1))[0, 1],
        'temperature': feature_scaler.inverse_transform(features.reshape(1, -1))[0, 0],
        'population': feature_scaler.inverse_transform(features.reshape(1, -1))[0, 2]
    }
    
    city_data = fetch_city_data(city_name)
    thresholds = calculate_thresholds(city_data)
    stats_analysis = perform_statistical_analysis(city_data)
    
    context = {
        'current_week': datetime.now().isocalendar().week,
        'weekly_data': city_data.groupby('week').agg({
            'temperature': 'mean',
            'rainfall': 'mean',
            'dengue_cases': 'sum'
        }).to_dict()
    }
    
    # Find neighboring cities with elevated risk
    neighboring_cities = []
    for i, edge in enumerate(graph_data.edge_index.t()):
        if edge[0] == city_index:
            neighbor_index = edge[1]
            neighbor_prediction = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)[neighbor_index].item()
            if neighbor_prediction > thresholds['moderate_risk']:
                neighboring_cities.append(city_names[neighbor_index])
    
    explanation = generate_explanation(city_name, prediction, feature_dict, thresholds, context, stats_analysis, neighboring_cities)
    
    return jsonify(explanation)

@app.route('/risk_monitor')
def risk_monitor():
    return render_template('risk_monitor.html')

@app.route('/get_risk_data')
def get_risk_data():
    global model, graph_data, feature_scaler, city_names, engine

    if model is None:
        return jsonify({"error": "Model not trained yet."})

    try:
        # Fetch the latest data for each city
        stmt = select(CityData).distinct(CityData.city).order_by(CityData.city, CityData.date.desc())
        with engine.connect() as connection:
            result = connection.execute(stmt)
            latest_data = pd.DataFrame(result.fetchall(), columns=result.keys())

        # Prepare data for prediction
        latest_data = engineer_features(latest_data)
        graph_data, _ = create_graph(latest_data)

        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr).cpu().numpy()

        # Define risk thresholds
        thresholds = {
            'high_risk': 50,
            'moderate_risk': 20,
            'heavy_rainfall': 100,
            'high_temperature': 30,
            'high_population_density': 1000000
        }

        # Prepare risk data
        risk_data = []
        for i, city in enumerate(city_names):
            city_data = latest_data[latest_data['city'] == city].iloc[0]
            
            # Determine risk level
            if predictions[i] > thresholds['high_risk']:
                risk_level = "High"
            elif predictions[i] > thresholds['moderate_risk']:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            # Find neighboring cities with elevated risk
            neighboring_cities = []
            for edge in graph_data.edge_index.t():
                if edge[0] == i:
                    neighbor_index = edge[1].item()
                    if predictions[neighbor_index] > thresholds['moderate_risk']:
                        neighboring_cities.append(city_names[neighbor_index])
            
            # Prepare feature dictionary for explanation
            feature_dict = {
                'rainfall': float(city_data['rainfall']),
                'temperature': float(city_data['temperature']),
                'population': int(city_data['population'])
            }
            
            # Generate explanation
            explanation = generate_explanation(city, float(predictions[i]), feature_dict, thresholds, neighboring_cities)
            
            risk_data.append({
                "city": city,
                "prediction": float(predictions[i]),
                "risk_level": risk_level,
                "temperature": float(city_data['temperature']),
                "rainfall": float(city_data['rainfall']),
                "population": int(city_data['population']),
                "explanation": explanation,
                "neighboring_cities": neighboring_cities
            })

        return jsonify(risk_data)

    except Exception as e:
        print(f"Error in get_risk_data: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/get_cities', methods=['GET'])
def get_cities():
    global city_names
    return jsonify({"cities": city_names})

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True)