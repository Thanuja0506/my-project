# My Project

This is a Machine Learning project that uses *FastAPI* to serve a trained model.

## Project Structure


├── main.py            # FastAPI application for model serving
├── train_model.py     # Script to train the model
├── model.pkl          # Saved trained model
├── requirements.txt   # List of dependencies
└── __pycache__/       # Compiled Python files


## Installation

1. Clone the repository or download the project files.
2. Create a virtual environment and activate it:

bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


3. Install dependencies:

bash
pip install -r requirements.txt


## Training the Model

To train the model and save it as model.pkl, run:

bash
python train_model.py


## Running the API

To start the FastAPI server:

bash
uvicorn main:app --reload


The API will be available at:  
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API Endpoints

- GET / → Root endpoint (test API)  
- POST /predict → Send input data to get predictions  

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- scikit-learn
- joblib
- numpy