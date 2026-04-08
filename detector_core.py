import os
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_model():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'model.pkl')
    return joblib.load(model_path)