# dengue_model.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def generate_fictional_data(num_cities=50, num_weeks=104):
    np.random.seed(42)
    cities = [f'CITY_{i:03d}' for i in range(num_cities)]
    dates = pd.date_range(start='2022-01-01', periods=num_weeks, freq='W')
    
    data = []
    for city in cities:
        base_temp = np.random.uniform(20, 30)
        base_rainfall = np.random.uniform(50, 150)
        base_population = np.random.uniform(50000, 500000)
        
        for date in dates:
            temp = base_temp + np.random.normal(0, 2)
            rainfall = max(0, base_rainfall + np.random.normal(0, 20))
            population = base_population * (1 + np.random.uniform(-0.01, 0.01))
            
            season_factor = 1 + 0.5 * np.sin(2 * np.pi * date.dayofyear / 365)
            dengue_cases = int(np.random.poisson(10 * season_factor * (temp/25) * (rainfall/100) * (population/100000)))
            
            data.append({
                'city': city,
                'date': date,
                'temperature': temp,
                'rainfall': rainfall,
                'population': population,
                'dengue_cases': dengue_cases
            })
    
    return pd.DataFrame(data)

def engineer_features(df):
    df['temperature_lag1'] = df.groupby('city')['temperature'].shift(1)
    df['rainfall_lag1'] = df.groupby('city')['rainfall'].shift(1)
    df['cases_lag1'] = df.groupby('city')['dengue_cases'].shift(1)
    df['cumulative_rainfall_4w'] = df.groupby('city')['rainfall'].rolling(window=4).sum().reset_index(0, drop=True)
    return df.dropna()

def create_graph(df, target_col='dengue_cases'):
    node_mapping = {code: i for i, code in enumerate(df['city'].unique())}
    
    feature_cols = ['temperature', 'rainfall', 'population', 'temperature_lag1', 'rainfall_lag1', 'cases_lag1', 'cumulative_rainfall_4w']
    node_features = df.groupby('city')[feature_cols].last().values
    
    num_nodes = len(node_mapping)
    edge_index = torch.tensor(np.array(np.meshgrid(range(num_nodes), range(num_nodes))).reshape(2, -1), dtype=torch.long)
    
    y = df.groupby('city')[target_col].last().values
    
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.float))
    
    return data, scaler

class DengueGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_heads=4):
        super(DengueGNN, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.lin(x)
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
    
    train_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, train_mask=train_mask)
    test_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, test_mask=test_mask)
    
    return train_data, test_data

def train_model(train_data, num_epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DengueGNN(num_node_features=train_data.num_node_features, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, None)
        loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()
    
    return model

def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index, None)
        mse = mean_squared_error(test_data.y[test_data.test_mask].cpu().numpy(), out[test_data.test_mask].cpu().numpy())
        r2 = r2_score(test_data.y[test_data.test_mask].cpu().numpy(), out[test_data.test_mask].cpu().numpy())
    return mse, r2

def generate_explanation(city, prediction, features, thresholds):
    rainfall = features['rainfall']
    temperature = features['temperature']
    population = features['population']
    
    risk_level = "high" if prediction > thresholds['high_risk'] else "moderate" if prediction > thresholds['moderate_risk'] else "low"
    
    explanation = f"The system predicts a {risk_level} dengue risk in {city} with an estimated {prediction:.2f} cases in the next two weeks. "
    
    reasons = []
    if rainfall > thresholds['heavy_rainfall']:
        reasons.append(f"heavy rainfall ({rainfall:.2f} mm)")
    if temperature > thresholds['high_temperature']:
        reasons.append(f"high temperature ({temperature:.2f}°C)")
    if population > thresholds['high_population_density']:
        reasons.append(f"high population density ({population:.0f} people)")
    
    if reasons:
        explanation += "This prediction is based on the following factors: " + ", ".join(reasons) + ". "
    else:
        explanation += "No specific high-risk factors were identified, but vigilance is still recommended. "
    
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
    
    explanation += "Based on this risk level, we recommend: "
    explanation += ", ".join(recommendations[risk_level]) + "."
    
    return explanation