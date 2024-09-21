# ExplainDengue

ExplainDengue is an advanced disease outbreak prediction system that uses machine learning to forecast dengue fever outbreaks in various cities. The application utilizes historical data on temperature, rainfall, population, and past dengue cases to make accurate predictions and provide explanations for the forecasts.

## Features

- Generate fictional dengue outbreak data for multiple cities
- Train a Graph Neural Network (GNN) model on the generated data
- Make predictions for dengue outbreaks in specific cities
- Provide detailed explanations for predictions, including risk factors and confidence levels
- Visualize historical trends of dengue cases
- User-friendly web interface built with Flask and Tailwind CSS

## Prerequisites

- Python 3.9 or higher
- Conda (for environment management)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/gunturbudi/explain-dengue.git
   cd explain-dengue
   ```

2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate explain-dengue
   ```

## Running the Application


1. Run the Flask application:
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
   - Click the "Train Model" button to train the GNN on the generated data

3. Making Predictions:
   - Visit the "Dashboard" page
   - Select a city from the dropdown menu
   - Click the "Predict" button to get a dengue outbreak prediction for the selected city

4. Viewing Results:
   - The prediction result, along with a detailed explanation, will be displayed on the Dashboard
   - The explanation includes risk factors, confidence levels, and recommended actions
   - A plot showing the historical trend of dengue cases is also available on the Dashboard

## Project Structure

- `app.py`: Main Flask application
- `models.py`: Database models
- `dengue_model.py`: Machine learning model and related functions
- `templates/`: HTML templates for the web interface
- `environment.yml`: Conda environment configuration
- `requirements.txt`: Python package dependencies

## Contributing

Contributions to ExplainDengue are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.