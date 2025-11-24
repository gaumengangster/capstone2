import mlflow
import os
from dotenv import load_dotenv
import pandas as pd


def load_model(model_name):
    alias = "production"
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def load_model_by_run_id(run_id, model_name):
    import mlflow
    import os

    #mlflow.set_tracking_uri(mlflow_tracking_uri)
    logged_model = f"runs:/{run_id}/{model_name}"
    print(f"Loading model from URI: {logged_model}")

    try:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    return loaded_model


def predict(run_id, model_name, data):
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    gcp_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("Data input:", data)
    #model_input = pd.DataFrame([data.dict()])
    model_input = pd.DataFrame([data])  # direktno bez .dict()

    print("Load model...")
    #model = load_model(model_name)
    model = load_model_by_run_id(run_id, model_name)# "green-taxi-ride-duration"
    print("Making prediction with data: ", model_input.head())
    prediction = model.predict(model_input)
    return float(prediction[0])