# dengue_model.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from geopy.distance import geodesic
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def generate_fictional_data(num_cities=50, num_weeks=104):
    np.random.seed(42)
    cities = [f'CITY_{i:03d}' for i in range(num_cities)]
    dates = pd.date_range(start='2022-01-01', periods=num_weeks, freq='W')
    
    data = []
    for city in cities:
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        base_temp = np.random.uniform(20, 30)
        base_rainfall = np.random.uniform(50, 150)
        base_population = np.random.uniform(50000, 500000)
        
        for date in dates:
            temp = max(0, base_temp + np.random.normal(0, 2))
            rainfall = max(0, base_rainfall * np.random.lognormal(0, 0.5))
            population = max(1000, base_population * (1 + np.random.uniform(-0.01, 0.01)))
            
            season_factor = 1 + 0.5 * np.sin(2 * np.pi * date.dayofyear / 365)
            dengue_cases = max(0, int(np.random.negative_binomial(n=5, p=0.5) * season_factor * (temp/25) * (rainfall/100) * (population/100000)))
            
            data.append({
                'city': city,
                'date': date.date(),
                'temperature': temp,
                'rainfall': rainfall,
                'population': population,
                'dengue_cases': dengue_cases,
                'latitude': lat,
                'longitude': lon
            })
    
    df = pd.DataFrame(data)
    
    # Ensure no NaN values
    assert df.isna().sum().sum() == 0, "NaN values found in generated data"
    
    return df

def engineer_features(df):
    logger.info(f"Starting feature engineering. Initial shape: {df.shape}")

    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(['city', 'date'])
    
    # Add lag features
    for col in ['temperature', 'rainfall', 'dengue_cases']:
        df[f'{col}_lag1'] = df.groupby('city')[col].shift(1)
    
    # Add rolling window feature
    df['cumulative_rainfall_4w'] = df.groupby('city')['rainfall'].rolling(window=4, min_periods=1).sum().reset_index(0, drop=True)
    
    # Add temporal features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # For the latest data point of each city, use the current values for lag features
    latest_mask = df.groupby('city')['date'].transform('max') == df['date']
    for col in ['temperature', 'rainfall', 'dengue_cases']:
        df.loc[latest_mask, f'{col}_lag1'] = df.loc[latest_mask, col]
    
    # Fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Ensure no infinity values
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    logger.info(f"Final shape after feature engineering: {df.shape}")
    logger.info(f"Columns after feature engineering: {df.columns.tolist()}")
    
    # Ensure no NaN values
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        logger.warning(f"NaN values found in columns: {nan_columns}")
        logger.warning(f"Number of NaN values: \n{df[nan_columns].isna().sum()}")
    
    return df

def train_test_split_graph(data, test_size=0.2, random_state=None):
    num_nodes = data.num_nodes
    num_test = int(num_nodes * test_size)
    num_train = num_nodes - num_test
    
    if random_state is not None:
        torch.manual_seed(random_state)
    
    perm = torch.randperm(num_nodes)
    train_indices = perm[:num_train]
    test_indices = perm[num_train:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    train_data = data.clone()
    test_data = data.clone()
    
    train_data.train_mask = train_mask
    test_data.test_mask = test_mask
    
    return train_data, test_data

def create_graph(df, max_distance=1000):
    logger.info(f"Starting graph creation with {len(df)} rows of data")
    
    node_mapping = {code: i for i, code in enumerate(df['city'].unique())}
    
    # Define the desired feature columns
    desired_feature_cols = ['temperature', 'rainfall', 'population', 'temperature_lag1', 'rainfall_lag1', 'cases_lag1', 'cumulative_rainfall_4w', 'day_of_year', 'month']
    
    # Select only the columns that exist in the DataFrame
    feature_cols = [col for col in desired_feature_cols if col in df.columns]
    
    logger.info(f"Using the following features for node creation: {feature_cols}")
    
    if not feature_cols:
        raise ValueError("No valid feature columns found in the DataFrame")
    
    node_features = df.groupby('city')[feature_cols].last().values
    
    # Ensure no NaN or inf values in node features
    if np.isnan(node_features).any():
        logger.warning("NaN values found in node features. Replacing with column means.")
        col_means = np.nanmean(node_features, axis=0)
        nan_mask = np.isnan(node_features)
        node_features[nan_mask] = np.take(col_means, nan_mask.nonzero()[1])
    
    if np.isinf(node_features).any():
        logger.warning("Inf values found in node features. Replacing with column means.")
        col_means = np.nanmean(np.where(np.isinf(node_features), np.nan, node_features), axis=0)
        inf_mask = np.isinf(node_features)
        node_features[inf_mask] = np.take(col_means, inf_mask.nonzero()[1])
    
    num_nodes = len(node_mapping)
    edges = []
    edge_features = []
    
    for city1, idx1 in node_mapping.items():
        city1_data = df[df['city'] == city1].iloc[-1]
        for city2, idx2 in node_mapping.items():
            if city1 != city2:
                city2_data = df[df['city'] == city2].iloc[-1]
                try:
                    distance = geodesic((city1_data['latitude'], city1_data['longitude']),
                                        (city2_data['latitude'], city2_data['longitude'])).km
                    if distance <= max_distance:
                        edges.append([idx1, idx2])
                        population_diff = abs(city1_data['population'] - city2_data['population']) / 1000000
                        edge_features.append([distance, population_diff])
                except Exception as e:
                    logger.warning(f"Error calculating distance between {city1} and {city2}: {str(e)}")
    
    if not edges:
        raise ValueError("No edges could be created. Check if cities are within the specified max_distance.")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    y = df.groupby('city')['dengue_cases'].last().values
    
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(y, dtype=torch.float))
    
    logger.info(f"Graph created with {data.num_nodes} nodes and {data.num_edges} edges")
    return data, scaler

class DengueGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_heads=4):
        super(DengueGNN, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=num_heads, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, edge_dim=num_edge_features)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(x)
        return x.squeeze(-1)

def train_model(train_data, num_epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DengueGNN(num_node_features=train_data.num_node_features, 
                      num_edge_features=train_data.num_edge_features, 
                      hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    train_data = train_data.to(device)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Check for NaN in model parameters
        if any(torch.isnan(param).any() for param in model.parameters()):
            raise ValueError("NaN values found in model parameters during training")
    
    return model


def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index, test_data.edge_attr)
        mse = mean_squared_error(test_data.y[test_data.test_mask].cpu().numpy(), 
                                 out[test_data.test_mask].cpu().numpy())
        r2 = r2_score(test_data.y[test_data.test_mask].cpu().numpy(), 
                      out[test_data.test_mask].cpu().numpy())
    return mse, r2

def generate_explanation(city_name, prediction, feature_dict, thresholds, neighboring_cities, stats_analysis):
    rainfall = feature_dict['rainfall']
    temperature = feature_dict['temperature']
    population = feature_dict['population']
    
    if prediction > thresholds['high_risk']:
        risk_level = "high"
    elif prediction > thresholds['moderate_risk']:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    factors = []
    if rainfall > thresholds['heavy_rainfall']:
        factors.append({"name": "Rainfall", "value": f"{rainfall:.2f} mm", "threshold": f"{thresholds['heavy_rainfall']} mm"})
    if temperature > thresholds['high_temperature']:
        factors.append({"name": "Temperature", "value": f"{temperature:.2f}°C", "threshold": f"{thresholds['high_temperature']}°C"})
    if 'high_population_density' in thresholds and population > thresholds['high_population_density']:
        factors.append({"name": "Population", "value": f"{population:.0f}", "threshold": f"{thresholds['high_population_density']}"})
    
    explanation = f"The system predicts a {risk_level} dengue risk in {city_name} with an estimated {prediction:.2f} cases in the next two weeks. "
    
    if factors:
        explanation += "This prediction is based on the following factors: " + ", ".join([f"{factor['name']} ({factor['value']})" for factor in factors]) + ". "
    else:
        explanation += "No specific high-risk factors were identified, but vigilance is still recommended. "
    
    if stats_analysis:
        explanation += f"\n\nStatistical Analysis:\n"
        explanation += f"- Average dengue cases: {stats_analysis['mean_cases']:.2f if stats_analysis['mean_cases'] is not None else 'N/A'} "
        explanation += f"(median: {stats_analysis['median_cases']:.2f if stats_analysis['median_cases'] is not None else 'N/A'})\n"
        explanation += f"- Case variability: {stats_analysis['std_cases']:.2f if stats_analysis['std_cases'] is not None else 'N/A'} (standard deviation)\n"
        
        if stats_analysis['temp_correlation'] is not None:
            explanation += f"- Temperature correlation: {stats_analysis['temp_correlation']:.2f} "
            explanation += f"({'significant' if stats_analysis['temp_correlation_p'] < 0.05 else 'not significant'})\n"
        
        if stats_analysis['rain_correlation'] is not None:
            explanation += f"- Rainfall correlation: {stats_analysis['rain_correlation']:.2f} "
            explanation += f"({'significant' if stats_analysis['rain_correlation_p'] < 0.05 else 'not significant'})\n"
        
        if stats_analysis['trend_direction'] is not None:
            explanation += f"- Trend: {stats_analysis['trend_direction']} "
            explanation += f"(from {stats_analysis['trend_start']:.2f} to {stats_analysis['trend_end']:.2f} cases)\n"
        
        if stats_analysis['temp_correlation_p'] is not None and stats_analysis['temp_correlation_p'] < 0.05:
            explanation += f"Temperature shows a significant {'positive' if stats_analysis['temp_correlation'] > 0 else 'negative'} correlation with dengue cases. "
        if stats_analysis['rain_correlation_p'] is not None and stats_analysis['rain_correlation_p'] < 0.05:
            explanation += f"Rainfall shows a significant {'positive' if stats_analysis['rain_correlation'] > 0 else 'negative'} correlation with dengue cases. "
        if stats_analysis['trend_direction'] is not None:
            explanation += f"The overall trend of dengue cases is {stats_analysis['trend_direction'].lower()}. "
    else:
        explanation += "\n\nInsufficient data for detailed statistical analysis."
    
    recommendations = {
        "high": [
            "Intensify vector control measures",
            "Conduct public awareness campaigns",
            "Ensure hospitals are prepared for potential outbreaks"
        ],
        "moderate": [
            "Continue regular mosquito control activities",
            "Encourage community participation in eliminating breeding sites",
            "Monitor local health facilities for increased dengue cases"
        ],
        "low": [
            "Maintain routine surveillance",
            "Educate the public on personal protection measures",
            "Keep emergency response plans updated"
        ]
    }
    
    explanation += "\nBased on this risk level and statistical analysis, we recommend: "
    explanation += ", ".join(recommendations[risk_level]) + ". "
    
    if stats_analysis and stats_analysis['trend_direction'] == 'Increasing':
        explanation += "Given the increasing trend, consider allocating additional resources for dengue prevention and control. "
    if stats_analysis and stats_analysis['temp_correlation_p'] is not None and stats_analysis['temp_correlation_p'] < 0.05 and stats_analysis['temp_correlation'] > 0:
        explanation += "With temperature significantly correlated to cases, intensify prevention efforts during warmer periods. "
    if stats_analysis and stats_analysis['rain_correlation_p'] is not None and stats_analysis['rain_correlation_p'] < 0.05 and stats_analysis['rain_correlation'] > 0:
        explanation += "As rainfall is significantly correlated with cases, increase vigilance and vector control during rainy seasons. "
    
    neighboring_info = f"Neighboring cities with elevated risk: {', '.join(neighboring_cities)}" if neighboring_cities else "No neighboring cities with elevated risk."
    
    return {
        "city": city_name,
        "prediction": float(prediction),
        "risk_level": risk_level,
        "factors": factors,
        "recommendations": recommendations[risk_level],
        "explanation": explanation,
        "neighboring_info": neighboring_info
    }

def perform_statistical_analysis(data):
    if len(data) < 2:
        return {
            'mean_cases': None,
            'median_cases': None,
            'std_cases': None,
            'mean_temperature': None,
            'mean_rainfall': None,
            'temp_correlation': None,
            'temp_correlation_p': None,
            'rain_correlation': None,
            'rain_correlation_p': None,
            'trend_start': None,
            'trend_end': None,
            'trend_direction': None
        }

    # Basic statistics
    stats_dict = {
        'mean_cases': data['dengue_cases'].mean(),
        'median_cases': data['dengue_cases'].median(),
        'std_cases': data['dengue_cases'].std(),
        'mean_temperature': data['temperature'].mean(),
        'mean_rainfall': data['rainfall'].mean(),
    }
    
    # Correlation analysis
    if len(data) > 2:  # Need more than 2 points for correlation
        corr_temp = stats.pearsonr(data['temperature'], data['dengue_cases'])
        corr_rain = stats.pearsonr(data['rainfall'], data['dengue_cases'])
        stats_dict['temp_correlation'] = corr_temp[0]
        stats_dict['temp_correlation_p'] = corr_temp[1]
        stats_dict['rain_correlation'] = corr_rain[0]
        stats_dict['rain_correlation_p'] = corr_rain[1]
    else:
        stats_dict['temp_correlation'] = None
        stats_dict['temp_correlation_p'] = None
        stats_dict['rain_correlation'] = None
        stats_dict['rain_correlation_p'] = None
    
    # Trend analysis
    if len(data) >= 4:  # Need at least 4 points for seasonal decompose
        trend = seasonal_decompose(data['dengue_cases'], model='additive', period=4).trend
        stats_dict['trend_start'] = trend.iloc[0]
        stats_dict['trend_end'] = trend.iloc[-1]
        stats_dict['trend_direction'] = 'Increasing' if trend.iloc[-1] > trend.iloc[0] else 'Decreasing'
    else:
        stats_dict['trend_start'] = None
        stats_dict['trend_end'] = None
        stats_dict['trend_direction'] = None
    
    return stats_dict