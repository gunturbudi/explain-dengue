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

def engineer_features(df):
    logger.info(f"Starting feature engineering. Initial shape: {df.shape}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['adm3_en', 'date'])
    
    # Add lag features
    for col in ['tave', 'pr', 'case_total_dengue']:
        df[f'{col}_lag1'] = df.groupby('adm3_en')[col].shift(1)
    
    # Add rolling window features
    df['cumulative_rainfall_4w'] = df.groupby('adm3_en')['pr'].rolling(window=4, min_periods=1).sum().reset_index(0, drop=True)
    
    # Add temporal features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # For the latest data point of each city, use the current values for lag features
    latest_mask = df.groupby('adm3_en')['date'].transform('max') == df['date']
    for col in ['tave', 'pr', 'case_total_dengue']:
        df.loc[latest_mask, f'{col}_lag1'] = df.loc[latest_mask, col]
    
    # Improved handling of NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in ['pct_area_cropland', 'pct_area_herbaceous_wetland', 'pct_area_mangroves', 'pct_area_permanent_water_bodies']:
            df[col] = df[col].fillna(0)
        elif col in ['pop_count_mean', 'pop_count_stdev', 'pop_count_total', 'pop_density_mean', 'pop_density_stdev']:
            df[col] = df.groupby('adm3_en')[col].transform(lambda x: x.fillna(x.mean()))
        elif col in ['spi3', 'spi6', 'pnp']:
            df[col] = df[col].fillna(df[col].mean())
        elif col == 'death_total_dengue':
            df[col] = df[col].fillna(0)  # Assume no deaths if not reported
        else:
            df[col] = df.groupby('adm3_en')[col].transform(lambda x: x.fillna(x.mean()))
    
    # Ensure no infinity values
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    logger.info(f"Final shape after feature engineering: {df.shape}")
    logger.info(f"Columns after feature engineering: {df.columns.tolist()}")
    
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        logger.warning(f"NaN values found in columns: {nan_columns}")
        logger.warning(f"Number of NaN values: \n{df[nan_columns].isna().sum()}")
    else:
        logger.info("No NaN values remain after preprocessing")
    
    return df

def create_graph(df):
    logger.info(f"Starting graph creation with {len(df)} rows of data")
    
    node_mapping = {code: i for i, code in enumerate(df['adm3_en'].unique())}
    
    feature_cols = ['tave', 'pr', 'rh', 'ndvi', 'pop_count_total', 'case_total_dengue',
                    'tave_lag1', 'pr_lag1', 'case_total_dengue_lag1', 'cumulative_rainfall_4w']
    
    logger.info(f"Using the following features for node creation: {feature_cols}")
    
    if not all(col in df.columns for col in feature_cols):
        raise ValueError("Not all required feature columns found in the DataFrame")
    
    # Handle NaN values before creating node features
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    
    node_features = df.groupby('adm3_en')[feature_cols].last().values
    
    # Check for any remaining NaN values and replace them with column means
    col_means = np.nanmean(node_features, axis=0)
    nan_mask = np.isnan(node_features)
    node_features[nan_mask] = np.take(col_means, nan_mask.nonzero()[1])
    
    num_nodes = len(node_mapping)
    edges = []
    edge_features = []
    
    # Create edges based on feature similarity
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            temp_diff = abs(node_features[i][0] - node_features[j][0])
            rain_diff = abs(node_features[i][1] - node_features[j][1])
            
            if temp_diff < 5 and rain_diff < 50:
                edges.append([i, j])
                edges.append([j, i])
                edge_features.append([temp_diff, rain_diff])
                edge_features.append([temp_diff, rain_diff])
    
    if not edges:
        raise ValueError("No edges could be created. Consider adjusting the similarity thresholds.")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    y = df.groupby('adm3_en')['case_total_dengue'].last().values
    
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    
    symbolic_rules_impact = apply_symbolic_rules(df)
    symbolic_rules_impact = symbolic_rules_impact.groupby(df['adm3_en']).last().values
    
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(y, dtype=torch.float),
                symbolic_rules=torch.tensor(symbolic_rules_impact, dtype=torch.float))
    
    logger.info(f"Graph created with {data.num_nodes} nodes and {data.num_edges} edges")
    logger.info(f"Symbolic rules shape: {symbolic_rules_impact.shape}")
    return data, scaler

def apply_symbolic_rules(data):
    # Ensure data is a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame().T

    rules_impact = pd.DataFrame(index=data.index, columns=['climate', 'environmental', 'socioeconomic', 'vector', 'temporal'])
    
    # Climate Rules
    rules_impact['climate'] = (
        ((data['tave'] >= 25) & (data['tave'] <= 35)).astype(int) +
        (data['pr'] > data['pr'].mean()).astype(int) +
        ((data['rh'] >= 60) & (data['rh'] <= 80)).astype(int)
    ).fillna(0)
    
    # Environmental Rules
    rules_impact['environmental'] = (
        (data['pct_area_cropland'] > data['pct_area_cropland'].mean()).astype(int) +
        (data['ndvi'] < data['ndvi'].mean()).astype(int)
    ).fillna(0)
    
    # Socioeconomic Rules
    rules_impact['socioeconomic'] = (
        (data['pop_density_mean'] > data['pop_density_mean'].mean()).astype(int) +
        (data['doh_pois_count'] < data['doh_pois_count'].mean()).astype(int) +
        (data['rwi_mean'] < data['rwi_mean'].mean()).astype(int)
    ).fillna(0)
    
    # Vector Dynamics Rules
    rules_impact['vector'] = (
        (data['case_total_dengue'] > data['case_total_dengue'].mean()).astype(int)
    ).fillna(0)
    
    # Temporal Rules
    try:
        data['date'] = pd.to_datetime(data['date'])
        peak_season = (data['date'].dt.month.isin([6, 7, 8, 9, 10])).astype(int)
    except (AttributeError, KeyError):
        peak_season = pd.Series(0, index=data.index)
    
    try:
        increasing_cases = (data['case_total_dengue'] > data['case_total_dengue'].shift(1)).astype(int)
    except KeyError:
        increasing_cases = pd.Series(0, index=data.index)
    
    rules_impact['temporal'] = (peak_season + increasing_cases).fillna(0)
    
    return rules_impact

class DengueGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_symbolic_rules, num_heads=4):
        super(DengueGNN, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=num_heads, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, edge_dim=num_edge_features)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
        self.symbolic_rules = torch.nn.Linear(num_symbolic_rules, hidden_channels)

    def forward(self, x, edge_index, edge_attr, symbolic_rules):
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        symbolic_impact = torch.relu(self.symbolic_rules(symbolic_rules))
        x = x + symbolic_impact  # Incorporate symbolic rules
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return x.squeeze(-1)

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

def train_model(train_data, num_epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_symbolic_rules = train_data.symbolic_rules.shape[1]
    model = DengueGNN(num_node_features=train_data.num_node_features, 
                      num_edge_features=train_data.num_edge_features, 
                      hidden_channels=64,
                      num_symbolic_rules=num_symbolic_rules).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    train_data = train_data.to(device)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, train_data.edge_attr, train_data.symbolic_rules)
        loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()
        
        if any(torch.isnan(param).any() for param in model.parameters()):
            raise ValueError("NaN values found in model parameters during training")
    
    return model

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index, test_data.edge_attr, test_data.symbolic_rules)
        mse = mean_squared_error(test_data.y[test_data.test_mask].cpu().numpy(), 
                                 out[test_data.test_mask].cpu().numpy())
        r2 = r2_score(test_data.y[test_data.test_mask].cpu().numpy(), 
                      out[test_data.test_mask].cpu().numpy())
    return mse, r2

def generate_explanation(city_name, prediction, feature_dict, thresholds, neighboring_cities, stats_analysis, symbolic_rules_impact):
    rainfall = feature_dict['pr']
    temperature = feature_dict['tave']
    population = feature_dict['pop_count_total']
    
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
    
    # Add explanation based on symbolic rules
    rule_explanations = []
    for rule, impact in symbolic_rules_impact.items():
        if impact > 0:
            if rule == 'climate':
                rule_explanations.append("The climate conditions are favorable for mosquito breeding and survival.")
            elif rule == 'environmental':
                rule_explanations.append("The environmental factors increase the potential for mosquito habitats.")
            elif rule == 'socioeconomic':
                rule_explanations.append("Socioeconomic factors may contribute to increased dengue risk.")
            elif rule == 'vector':
                rule_explanations.append("There are indications of increased mosquito population.")
            elif rule == 'human':
                rule_explanations.append("Human factors, such as previous dengue cases, contribute to the risk.")
            elif rule == 'temporal':
                rule_explanations.append("The current time period is associated with higher dengue risk.")

    if rule_explanations:
        explanation += "Based on our analysis: " + " ".join(rule_explanations)

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
    
    explanation += "\nBased on this risk level and analysis, we recommend: "
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
        'mean_cases': data['case_total_dengue'].mean(),
        'median_cases': data['case_total_dengue'].median(),
        'std_cases': data['case_total_dengue'].std(),
        'mean_temperature': data['tave'].mean(),
        'mean_rainfall': data['pr'].mean(),
    }
    
    # Correlation analysis
    if len(data) > 2:  # Need more than 2 points for correlation
        corr_temp = stats.pearsonr(data['tave'], data['case_total_dengue'])
        corr_rain = stats.pearsonr(data['pr'], data['case_total_dengue'])
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
        trend = seasonal_decompose(data['case_total_dengue'], model='additive', period=4).trend
        stats_dict['trend_start'] = trend.iloc[0]
        stats_dict['trend_end'] = trend.iloc[-1]
        stats_dict['trend_direction'] = 'Increasing' if trend.iloc[-1] > trend.iloc[0] else 'Decreasing'
    else:
        stats_dict['trend_start'] = None
        stats_dict['trend_end'] = None
        stats_dict['trend_direction'] = None
    
    return stats_dict