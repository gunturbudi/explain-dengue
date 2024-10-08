# ExplainDengue

ExplainDengue is an advanced disease outbreak prediction system that uses machine learning to forecast dengue fever outbreaks in various cities. The application utilizes historical data on temperature, rainfall, population, and past dengue cases to make accurate predictions and provide explanations for the forecasts.

## Features

- Generate fictional dengue outbreak data for multiple cities
- Train a Graph Attention Network (GAT) model on the generated data
- Make predictions for dengue outbreaks in specific cities
- Provide detailed explanations for predictions, including risk factors and statistical analysis
- Visualize historical trends of dengue cases
- Monitor real-time risk levels across multiple cities
- User-friendly web interface built with Flask and Tailwind CSS

## Prerequisites

- Python 3.9 or higher
- pip (for package management)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/explain-dengue.git
   cd explain-dengue
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Initialize the database:
   ```
   flask db init
   flask db migrate
   flask db upgrade
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to `http://localhost:5000` to access the ExplainDengue application.

## Usage

1. Data Generation:
   - Navigate to the "Data Management" page
   - Click the "Generate Data" button to create fictional dengue outbreak data

2. Model Training:
   - Go to the "Model Management" page
   - Click the "Train Model" button to train the GAT model on the generated data

3. Making Predictions:
   - Visit the "Dashboard" page
   - Select a city from the dropdown menu
   - Click the "Predict" button to get a dengue outbreak prediction for the selected city

4. Viewing Results:
   - The prediction result, along with a detailed explanation, will be displayed on the Dashboard
   - The explanation includes risk factors, statistical analysis, and recommended actions
   - A plot showing the historical trend of dengue cases is also available on the Dashboard

5. Risk Monitoring:
   - Visit the "Risk Monitor" page to see real-time risk levels across all cities

## Project Structure

- `app.py`: Main Flask application
- `models.py`: Database models
- `dengue_model.py`: Machine learning model and related functions
- `templates/`: HTML templates for the web interface
- `requirements.txt`: Python package dependencies

## Contributing

Contributions to ExplainDengue are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.