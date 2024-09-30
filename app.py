import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from geopy.distance import geodesic
from flask_migrate import Migrate
from datetime import datetime, timedelta
from sqlalchemy import select, create_engine, and_, func
import logging
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from models import db, CityData
from dengue_model import (
    engineer_features,
    create_graph,
    DengueGNN,
    train_test_split_graph,
    generate_explanation,
    train_model,
    evaluate_model,
    perform_statistical_analysis,
    apply_symbolic_rules
)

app = Flask(__name__)
migrate = Migrate(app, db)

basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'dengue_data.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db.init_app(app)

engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

model = None
model_status = "Not trained"
graph_data = None
feature_scaler = None
city_names = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_database():
    with app.app_context():
        db.create_all()
        global city_names
        city_names = [city[0] for city in db.session.query(CityData.adm3_en).distinct()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    with app.app_context():
        total_cities = db.session.query(CityData.adm3_en).distinct().count()
        total_cases = db.session.query(db.func.sum(CityData.case_total_dengue)).scalar() or 0
        avg_temperature = db.session.query(db.func.avg(CityData.tave)).scalar() or 0
        avg_rainfall = db.session.query(db.func.avg(CityData.pr)).scalar() or 0

    return render_template('dashboard.html', 
                           total_cities=total_cities,
                           total_cases=total_cases,
                           avg_temperature=avg_temperature,
                           avg_rainfall=avg_rainfall)

@app.route('/get_dashboard_data')
def get_dashboard_data():
    # Fetch necessary data from the database
    with app.app_context():
        total_cities = db.session.query(CityData.adm3_en).distinct().count()
        total_cases = db.session.query(db.func.sum(CityData.case_total_dengue)).scalar() or 0
        avg_temperature = db.session.query(db.func.avg(CityData.tave)).scalar() or 0
        avg_rainfall = db.session.query(db.func.avg(CityData.pr)).scalar() or 0

        # Get risk distribution
        risk_distribution = {
            'low': db.session.query(CityData).filter(CityData.case_total_dengue < 10).count(),
            'moderate': db.session.query(CityData).filter(CityData.case_total_dengue.between(10, 50)).count(),
            'high': db.session.query(CityData).filter(CityData.case_total_dengue > 50).count()
        }

        # Get cases over time
        cases_over_time = db.session.query(
            CityData.date,
            db.func.sum(CityData.case_total_dengue)
        ).group_by(CityData.date).order_by(CityData.date).all()

        # Get top 5 high-risk cities
        high_risk_cities = db.session.query(CityData).order_by(CityData.case_total_dengue.desc()).limit(5).all()

    return jsonify({
        'total_cities': total_cities,
        'total_cases': int(total_cases),
        'avg_temperature': round(float(avg_temperature), 2),
        'avg_rainfall': round(float(avg_rainfall), 2),
        'risk_distribution': risk_distribution,
        'cases_over_time': {
            'dates': [str(date) for date, _ in cases_over_time],
            'cases': [int(cases or 0) for _, cases in cases_over_time]
        },
        'high_risk_cities': [
            {
                'city': city.adm3_en,
                'risk_level': 'High' if city.case_total_dengue > 50 else 'Moderate',
                'predicted_cases': int(city.case_total_dengue or 0),
                'temperature': round(float(city.tave), 2) if city.tave is not None else None,
                'rainfall': round(float(city.pr), 2) if city.pr is not None else None
            } for city in high_risk_cities
        ]
    })
    
@app.route('/data')
def data_management():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    city_filter = request.args.get('city', '')
    date_filter = request.args.get('date', '')

    query = CityData.query

    if city_filter:
        query = query.filter(CityData.adm3_en.ilike(f'%{city_filter}%'))
    if date_filter:
        query = query.filter(CityData.date == date_filter)

    pagination = query.order_by(CityData.date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    data = pagination.items

    return render_template('data_management.html', 
                           data=data, 
                           pagination=pagination,
                           city_filter=city_filter,
                           date_filter=date_filter)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Clear existing data
            db.session.query(CityData).delete()
            db.session.commit()
            
            # Read and process CSV file
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                city_data = CityData(
                    adm3_en=row['adm3_en'],
                    adm3_pcode=row['adm3_pcode'],
                    date=datetime.strptime(row['date'], '%Y-%m-%d').date(),
                    year=row['year'],
                    week=row['week'],
                    doh_pois_count=row['doh_pois_count'],
                    ndvi=row['ndvi'],
                    pct_area_cropland=row['pct_area_cropland'],
                    pct_area_flood_hazard_5yr_high=row['pct_area_flood_hazard_5yr_high'],
                    pct_area_flood_hazard_5yr_low=row['pct_area_flood_hazard_5yr_low'],
                    pct_area_flood_hazard_5yr_med=row['pct_area_flood_hazard_5yr_med'],
                    pct_area_herbaceous_wetland=row['pct_area_herbaceous_wetland'],
                    pct_area_mangroves=row['pct_area_mangroves'],
                    pct_area_permanent_water_bodies=row['pct_area_permanent_water_bodies'],
                    pnp=row['pnp'],
                    pop_count_mean=row['pop_count_mean'],
                    pop_count_stdev=row['pop_count_stdev'],
                    pop_count_total=row['pop_count_total'],
                    pop_density_mean=row['pop_density_mean'],
                    pop_density_stdev=row['pop_density_stdev'],
                    pr=row['pr'],
                    rh=row['rh'],
                    rwi_mean=row['rwi_mean'],
                    rwi_std=row['rwi_std'],
                    spi3=row['spi3'],
                    spi6=row['spi6'],
                    tave=row['tave'],
                    tmax=row['tmax'],
                    tmin=row['tmin'],
                    case_total_dengue=row['case_total_dengue'],
                    death_total_dengue=row['death_total_dengue']
                )
                db.session.add(city_data)
            
            db.session.commit()
            return jsonify({"message": "CSV file processed successfully"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            os.remove(filepath)  # Remove the uploaded file after processing
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/model')
def model_management():
    global model
    model_status = "Trained" if model is not None else "Not trained"
    return render_template('model_management.html', model_status=model_status)

@app.route('/train', methods=['POST'])
def train():
    global model, graph_data, feature_scaler, city_names, engine, model_status
    
    stmt = select(CityData)
    with engine.connect() as connection:
        result = connection.execute(stmt)
        city_data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if city_data.empty:
        return jsonify({"error": "No data available. Please upload data first."})
    
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
    
    city_names = city_data['adm3_en'].unique().tolist()
    
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
        return jsonify({"error": "No data available. Please upload data first."})
    
    plt.figure(figsize=(10, 6))
    avg_cases = city_data.groupby('date')['case_total_dengue'].mean()
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
        (CityData.adm3_en == city_name) &
        (CityData.date >= start_date) &
        (CityData.date <= end_date)
    )
    with engine.connect() as connection:
        result = connection.execute(stmt)
        data = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    data['week'] = pd.to_datetime(data['date']).dt.isocalendar().week
    return data

def calculate_thresholds(data):
    if data.empty:
        return {
            'high_risk': 50,
            'moderate_risk': 20,
            'heavy_rainfall': 100,
            'high_temperature': 30,
        }
    
    return {
        'high_risk': np.percentile(data['case_total_dengue'], 75) if len(data['case_total_dengue']) > 0 else 50,
        'moderate_risk': np.percentile(data['case_total_dengue'], 50) if len(data['case_total_dengue']) > 0 else 20,
        'heavy_rainfall': np.percentile(data['pr'], 75) if len(data['pr']) > 0 else 100,
        'high_temperature': np.percentile(data['tave'], 75) if len(data['tave']) > 0 else 30,
    }

@app.route('/predict', methods=['POST'])
def predict():
    global model, graph_data, feature_scaler, city_names
    
    if not city_names:
        return jsonify({"error": "No cities available. Please upload data and train the model first."})
    
    city_name = request.form['city_name']
    if city_name not in city_names:
        return jsonify({"error": f"City '{city_name}' not found in the database."})
    
    city_index = city_names.index(city_name)
    
    if model is None:
        return jsonify({"error": "Model not trained. Please train the model first."})
    
    try:
        with torch.no_grad():
            prediction = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.symbolic_rules)[city_index].item()
        
        if np.isnan(prediction):
            raise ValueError("NaN value in prediction")
        
        features = graph_data.x[city_index].cpu().numpy()
        feature_dict = {
            'pr': feature_scaler.inverse_transform(features.reshape(1, -1))[0, graph_data.x.shape[1] - 3],
            'tave': feature_scaler.inverse_transform(features.reshape(1, -1))[0, graph_data.x.shape[1] - 2],
            'pop_count_total': feature_scaler.inverse_transform(features.reshape(1, -1))[0, graph_data.x.shape[1] - 1]
        }
        
        city_data = fetch_city_data(city_name)
        thresholds = calculate_thresholds(city_data)
        
        if len(city_data) >= 2:
            stats_analysis = perform_statistical_analysis(city_data)
        else:
            stats_analysis = None
        
        neighboring_cities = []
        for i, edge in enumerate(graph_data.edge_index.t()):
            if edge[0] == city_index:
                neighbor_index = edge[1]
                neighbor_prediction = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.symbolic_rules)[neighbor_index].item()
                if neighbor_prediction > thresholds['moderate_risk']:
                    neighboring_cities.append(city_names[neighbor_index])
        
        symbolic_rules_impact = apply_symbolic_rules(city_data.iloc[-1:])
        explanation = generate_explanation(city_name, prediction, feature_dict, thresholds, neighboring_cities, stats_analysis, symbolic_rules_impact.iloc[0])
        
        return jsonify(explanation)
    
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/risk_monitor')
def risk_monitor():
    return render_template('risk_monitor.html')

@app.route('/get_risk_data')
def get_risk_data():
    global model, graph_data, feature_scaler, city_names, engine

    logger.info("Starting get_risk_data function")

    if model is None:
        logger.warning("Model not trained yet")
        return jsonify({"error": "Model not trained yet. Please train the model first."})

    try:
        logger.info("Fetching unique cities")
        stmt = select(CityData.adm3_en).distinct()
        with engine.connect() as connection:
            result = connection.execute(stmt)
            cities = [row[0] for row in result]

        logger.info(f"Found {len(cities)} unique cities")
        if not cities:
            logger.warning("No cities available in the database")
            return jsonify({"error": "No cities available in the database."})

        logger.info("Fetching latest data for all cities")
        stmt = select(CityData).where(CityData.adm3_en.in_(cities))
        with engine.connect() as connection:
            result = connection.execute(stmt)
            latest_data = pd.DataFrame(result.fetchall(), columns=result.keys())

        logger.info(f"Fetched {len(latest_data)} rows of data")
        if latest_data.empty:
            logger.warning("No data available in the database")
            return jsonify({"error": "No data available in the database."})

        logger.info("Filtering for latest data per city")
        latest_data = latest_data.loc[latest_data.groupby('adm3_en')['date'].idxmax()]
        logger.info(f"After filtering, {len(latest_data)} rows remain")

        logger.info("Engineering features")
        latest_data = engineer_features(latest_data)
        logger.info(f"After feature engineering, shape of data: {latest_data.shape}")
        
        if latest_data.empty:
            logger.error("Data became empty after feature engineering")
            return jsonify({"error": "Data became empty after feature engineering"})

        try:
            logger.info("Creating graph")
            graph_data, _ = create_graph(latest_data)
            logger.info(f"Graph created with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
        except ValueError as e:
            logger.error(f"Error creating graph: {str(e)}")
            return jsonify({"error": f"Error creating graph: {str(e)}"})

        logger.info("Making predictions")
        model.eval()
        with torch.no_grad():
            predictions = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.symbolic_rules).cpu().numpy()
        logger.info(f"Made {len(predictions)} predictions")

        thresholds = {
            'high_risk': 50,
            'moderate_risk': 20,
            'heavy_rainfall': 100,
            'high_temperature': 30,
            'high_population_density': 1000000
        }

        logger.info("Preparing risk data")
        risk_data = []
        for i, city in enumerate(cities):
            city_data = latest_data[latest_data['adm3_en'] == city]
            if city_data.empty:
                logger.warning(f"No data for city {city}")
                continue
            city_data = city_data.iloc[0]
            
            prediction = float(predictions[i])
            if np.isnan(prediction):
                prediction = None  # Replace NaN with None

            if prediction is not None and prediction > thresholds['high_risk']:
                risk_level = "High"
            elif prediction is not None and prediction > thresholds['moderate_risk']:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            try:
                symbolic_rules_impact = apply_symbolic_rules(city_data)
                risk_factors = [factor for factor, impact in symbolic_rules_impact.iloc[0].items() if impact > 0]
            except Exception as e:
                logger.error(f"Error applying symbolic rules for {city}: {str(e)}")
                risk_factors = []
            
            risk_data.append({
                "city": city,
                "prediction": prediction,
                "risk_level": risk_level,
                "temperature": float(city_data['tave']) if pd.notnull(city_data['tave']) else None,
                "rainfall": float(city_data['pr']) if pd.notnull(city_data['pr']) else None,
                "population": int(city_data['pop_count_total']) if pd.notnull(city_data['pop_count_total']) else None,
                "risk_factors": risk_factors,
                "ndvi": float(city_data['ndvi']) if pd.notnull(city_data['ndvi']) else None,
                "relative_humidity": float(city_data['rh']) if pd.notnull(city_data['rh']) else None,
                "pct_area_cropland": float(city_data['pct_area_cropland']) if pd.notnull(city_data['pct_area_cropland']) else None,
                "pct_area_flood_hazard": float(city_data['pct_area_flood_hazard_5yr_high'] + 
                                               city_data['pct_area_flood_hazard_5yr_med'] + 
                                               city_data['pct_area_flood_hazard_5yr_low']) if pd.notnull(city_data['pct_area_flood_hazard_5yr_high']) and 
                                                                                              pd.notnull(city_data['pct_area_flood_hazard_5yr_med']) and 
                                                                                              pd.notnull(city_data['pct_area_flood_hazard_5yr_low']) else None
            })

        logger.info(f"Prepared risk data for {len(risk_data)} cities")

        # Calculate additional statistics for dashboard charts
        avg_temperature = np.mean([city['temperature'] for city in risk_data if city['temperature'] is not None])
        avg_rainfall = np.mean([city['rainfall'] for city in risk_data if city['rainfall'] is not None])
        risk_distribution = {
            'High': len([city for city in risk_data if city['risk_level'] == 'High']),
            'Moderate': len([city for city in risk_data if city['risk_level'] == 'Moderate']),
            'Low': len([city for city in risk_data if city['risk_level'] == 'Low'])
        }

        return jsonify({
            'cities': risk_data,
            'avg_temperature': avg_temperature,
            'avg_rainfall': avg_rainfall,
            'risk_distribution': risk_distribution
        })

    except Exception as e:
        logger.error(f"Error in get_risk_data: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    with app.app_context():
        db.drop_all()
        db.create_all()
        
        for _, row in df.iterrows():
            city_data = CityData(
                adm3_en=row['adm3_en'],
                adm3_pcode=row['adm3_pcode'],
                date=datetime.strptime(row['date'], '%Y-%m-%d').date(),
                year=row['year'],
                week=row['week'],
                doh_pois_count=row['doh_pois_count'],
                ndvi=row['ndvi'],
                pct_area_cropland=row['pct_area_cropland'],
                pct_area_flood_hazard_5yr_high=row['pct_area_flood_hazard_5yr_high'],
                pct_area_flood_hazard_5yr_low=row['pct_area_flood_hazard_5yr_low'],
                pct_area_flood_hazard_5yr_med=row['pct_area_flood_hazard_5yr_med'],
                pct_area_herbaceous_wetland=row['pct_area_herbaceous_wetland'],
                pct_area_mangroves=row['pct_area_mangroves'],
                pct_area_permanent_water_bodies=row['pct_area_permanent_water_bodies'],
                pnp=row['pnp'],
                pop_count_mean=row['pop_count_mean'],
                pop_count_stdev=row['pop_count_stdev'],
                pop_count_total=row['pop_count_total'],
                pop_density_mean=row['pop_density_mean'],
                pop_density_stdev=row['pop_density_stdev'],
                pr=row['pr'],
                rh=row['rh'],
                rwi_mean=row['rwi_mean'],
                rwi_std=row['rwi_std'],
                spi3=row['spi3'],
                spi6=row['spi6'],
                tave=row['tave'],
                tmax=row['tmax'],
                tmin=row['tmin'],
                case_total_dengue=row['case_total_dengue'],
                death_total_dengue=row['death_total_dengue']
            )
            db.session.add(city_data)
        
        db.session.commit()
    
    global city_names
    city_names = df['adm3_en'].unique().tolist()
    
@app.route('/get_cities', methods=['GET'])
def get_cities():
    global city_names
    return jsonify({"cities": city_names})

if __name__ == '__main__':
    csv_path = os.path.join(basedir, 'fil_data.csv')  # Adjust this path as needed
    load_csv_data(csv_path)
    app.run(debug=True)