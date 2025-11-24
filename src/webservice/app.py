import os
from dotenv import load_dotenv
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import io
from typing import Any, List, Optional, Dict
import json
from data_model import PredictRequest, Prediction, MlFlowConfig, ExperimentRequest, LogModelParams, ColumnMapping, LoggedModelResponse
import requests
from prometheus_fastapi_instrumentator import Instrumentator
from predict import predict, load_model_by_run_id
from reporting_util import create_report, load_report_into_sql
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from mlflow.models.signature import infer_signature
from fastapi.responses import JSONResponse
from pprint import pprint
# echo "MLFLOW_TRACKING_URI=http://<external-ip>:5000" > .env
# creating the app
app = FastAPI()

Instrumentator().instrument(app).expose(app)

ENV_FILE = ".env"
load_dotenv(ENV_FILE)

@app.post("/ml_flow")
async def save_mlflow_config(data: MlFlowConfig):
    url_value = data.url.strip()
    bucket_value = data.bucket_location.strip()

    # Kreiraj .env fajl ako ne postoji
    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w") as f:
            f.write("")

    # Učitaj postojeće promenljive
    env_vars = {}
    with open(ENV_FILE, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                env_vars[key] = value

    # Ažuriraj vrednosti
    env_vars["MLFLOW_TRACKING_URI"] = url_value
    env_vars["GCP_BUCKET_URI"] = bucket_value # "gs://<GCS bucket>/models"

    # Upisi nazad u fajl
    with open(ENV_FILE, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    os.environ["MLFLOW_TRACKING_URI"] = url_value
    os.environ["GCP_BUCKET_URI"] = bucket_value

    return {
        "message": "MLFLOW and BUCKET URLs are saved!",
        "url": url_value,
        "bucket_location": bucket_value
    }

@app.get("/check_url")
async def check_url():
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
    return MLFLOW_TRACKING_URI

@app.post("/experiments")
async def create_experiment(request: ExperimentRequest):
    experiment_name = request.experiment_name
    tags = request.tags
    # Load environment variables from .env file
    load_dotenv()
    # Get the MLflow tracking URI from environment variables
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")

    # Set up the connection to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Give the experiment a name
    #experiment_name = f"green-taxi-monitoring"
    # Get the experiment by name
    exp = mlflow.get_experiment_by_name(experiment_name)

    GCP_BUCKET_URI=os.getenv("GCP_BUCKET_URI")
    # Create the experiment if it does not exist
    if exp is None:
        experiment_id = mlflow.create_experiment(name=experiment_name, 
                            tags=tags,
                            artifact_location=GCP_BUCKET_URI)
        mlflow.set_experiment(experiment_name)
        return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "creation_timestamp": datetime.now(),
        "tags":tags,
        "created": True
    }
    else:
        print(f"Using existing experiment: {experiment_name} (ID: {exp.experiment_id})")
        return {
        "experiment_id": exp.experiment_id,
        "experiment_name": experiment_name,
        "creation_timestamp": exp.creation_time,
        "tags":exp.tags,
        "created": False}



@app.post("/log_model")
async def log_model_artifact(params: str = Form(...),
    file: UploadFile = File(...)):
    """     You should now see your run in the MLflow UI. 
    Under the created experiment, 
    you can also see the logged tags, 
    the metric and the saved model. """
    logModelParams = LogModelParams(**json.loads(params))
    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))

    if logModelParams.target_column not in df.columns:# duration
       raise HTTPException(status_code=400, detail=f"Column '{logModelParams.target_column}' not found in dataset")

    numerical = logModelParams.numerical_columns or []
    categorical = logModelParams.categorical_columns or []

    if isinstance(numerical, str):
        numerical = [numerical]
    if isinstance(categorical, str):
        categorical = [categorical]

    y = df[logModelParams.target_column]
    X = df[numerical + categorical]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=logModelParams.test_size)

    load_dotenv()
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(logModelParams.experiment_name)
    
    """ Run = kontejner za sve logove jednog treninga
    Logged model = model artifact unutar run-a
    Registered model = proizvodna verzija modela koja može imati više verzija, stage-ova i alias-a """
    with mlflow.start_run() as run:
        tags = {
            "model": "linear regression",
            "developer": "capstone-dev",
            "target": logModelParams.target_column
        }
        if logModelParams.tags:
            tags.update(logModelParams.tags)

        mlflow.set_tags(tags)
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        y_pred = lr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mlflow.log_metric("rmse", rmse)

        features_dict = {"features_numerical": numerical, 
                         "features_categorical": categorical, 
                         "target": logModelParams.target_column, 
                         "prediction":logModelParams.predict_column}
        mlflow.log_text(json.dumps(features_dict), "features.json")

        X_train.to_csv("X_train_reference.csv", index=False)
        mlflow.log_artifact("X_train_reference.csv")
        #
        X_train_float = X_train.astype(float)

        input_example = X_train_float.iloc[[0]]
        signature = infer_signature(X_train_float, y_train)
        try:
            mlflow.sklearn.log_model(
                sk_model=lr,
                name=logModelParams.model_name,
                input_example=input_example,
                signature=signature,
                registered_model_name=logModelParams.model_name
            )

            print("Model logged successfully.")
            return {"created": True}
        except Exception as e:
            print(f"Error logging model: {e}")
            return {"created":False, "error":e}

        ### ovde je kraj u prvom notebooks
        ### dodati uslov params.shouldRegister
        run_id = mlflow.active_run().info.run_id
        
        # Register model name in the model registry
        client = MlflowClient()
        model_name = "green-taxi-ride-duration-2"
        tags = {"LinearRegression.framework": "Scikit-Learn"}
        desc = "Model to predict ride duration in NYC"
        client.create_registered_model(name=model_name,tags=tags,description=desc)

        # Create a new version of the rfr model under the registered model name
        model_uri = f"runs:/{run_id}/mlflow-model"
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

    
        # Set registered model alias
        new_alias = "production"
        client.set_registered_model_alias(
            name=model_name,
            version=mv.version,
            alias=new_alias
        )

@app.post("/predict", response_model=Prediction)
def predict_duration(request: PredictRequest):#"052cdec46b664b61a2f471648fb4e395","mlflow-model"
    cleaned_data = {
        k: float(v) if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).isdigit() else v
        for k, v in request.data.items()
    }
    prediction = predict(request.run_id, request.model_name, cleaned_data)
    return Prediction(
        data=cleaned_data,   # obavezno ime polja
        prediction=prediction
    )
    try:
        print(f"Sending data to metrics application: {data}")
        response = requests.post(
            f"http://evidently_service:8085/iterate/{model_name}",
            data=Prediction(
                **data.model_dump(), prediction=prediction
            ).model_dump_json(),
            headers={"content-type": "application/json"},
        )
    except requests.exceptions.ConnectionError as error:
        print(f"Cannot reach a metrics application, error: {error}, data: {data}")

    return Prediction(**data.model_dump(), prediction=prediction)

@app.post("/apply_model")
async def apply_model(params: str = Form(...),
    file: UploadFile = File(...)):

    params = LogModelParams(**json.loads(params))
    contents = await file.read()
    df = pd.read_parquet(io.BytesIO(contents))
    if params.target_column not in df.columns:# duration
        raise HTTPException(status_code=400, detail=f"Column '{params.target_column}' not found in dataset")

    load_dotenv()
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(params.experiment_name)#"green-taxi-monitoring-1"

    #gcp_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    #os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key

    loaded_model = load_model_by_run_id("052cdec46b664b61a2f471648fb4e395", MLFLOW_TRACKING_URI, params.model_name)# "green-taxi-ride-duration"
    cat_features = ["PULocationID", "DOLocationID"]
    num_features = ["trip_distance", "passenger_count", "fare_amount", "total_amount"]

    y = df[params.target_column]
    X = df.drop(columns=[params.target_column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=params.test_size)
    
    y_pred_train = loaded_model.predict(X_train)
    train_data = X_train.copy()
    train_data[params.target_column] = y_train
    train_data["prediction"] = y_pred_train

    y_pred_test = loaded_model.predict(X_test)
    val_data = X_test.copy()
    val_data[params.target_column] = y_test
    val_data["prediction"] = y_pred_test

    print(mean_absolute_error(train_data.duration, train_data.prediction))
    print(mean_absolute_error(val_data.duration, val_data.prediction))

    result = create_report(train_data, val_data, num_features, cat_features, pred_col='prediction')
    load_report_into_sql(result)

@app.get("/experiments")
def get_experiments():
    """
    Vraća sve eksperimente u MLflow-u
    """
    load_dotenv()
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiments = client.search_experiments()
    return [
        {
            "experiment_id": exp.experiment_id,
            "experiment_name": exp.name,
            "lifecycle_stage": exp.lifecycle_stage,
            "creation_timestamp":exp.creation_time
        }
        for exp in experiments
    ]

""" @app.get("/experiments/{experiment_id}/runs")
def get_runs(experiment_id: str):
    load_dotenv()
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    runs = client.search_runs([experiment_id])
    client.search_logged_models
    return [
        {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time
        }
        for run in runs
    ] """

@app.get("/logged-models", response_model=List[LoggedModelResponse])
def get_logged_models():
    load_dotenv()
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiments = client.search_experiments()
    all_experiment_ids = [exp.experiment_id for exp in experiments]
    logged_models = client.search_logged_models(experiment_ids=all_experiment_ids)
    response = []

    for m in logged_models[:2]:
 
        try:
            local_features_path = client.download_artifacts(m.source_run_id, "features.json")
            with open(local_features_path) as f:
                features_dict = json.load(f)
            numerical_features = features_dict.get("features_numerical", [])
            categorical_features = features_dict.get("features_categorical", [])
            target = features_dict.get("target", None)

        except Exception as e:
            numerical_features = []
            categorical_features = []
            target=None

        if m.metrics:
            # if m.metrics is a list of Metric objects
            metrics_dict = {getattr(metric, "key", None): getattr(metric, "value", None) for metric in m.metrics}
            # remove any None keys
            metrics_dict = {k: v for k, v in metrics_dict.items() if k is not None}
        else:
            metrics_dict = None
        

        all_versions = client.search_model_versions(f"name='{m.name}'")

        versions_list = [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status
                }
                for v in all_versions
                if v.run_id == m.source_run_id
            ]
        response.append(LoggedModelResponse(
                    model_id=m.model_id,
                    experiment_id=m.experiment_id,    
                    name=m.name,
                    artifact_location=m.artifact_location,
                    creation_timestamp=m.creation_timestamp,
                    last_updated_timestamp=m.last_updated_timestamp,
                    model_type=m.model_type,
                    source_run_id=m.source_run_id,
                    status=m.status,
                    status_message=m.status_message,
                    tags=m.tags,
                    params=m.params,
                    metrics=metrics_dict,
                    version=versions_list,
                    columnMapping=ColumnMapping(
                        target=target,
                        categorical=categorical_features,
                        numerical=numerical_features)
        ))
    return response

""" @app.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: str):

    try:
        load_dotenv()
        MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        return [
            {
                "path": artifact.path,
                "is_dir": artifact.is_dir
            }
            for artifact in artifacts
        ]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
 """
    
@app.get("/service-urls")
def get_service_urls():
    return {
        "mlflowUrl":os.getenv("MLFLOW_TRACKING_URI"),
        "postgresqlUrl":os.getenv("DB_URI"),
        "bucketUrl":os.getenv("GCP_BUCKET_URI")
    }
    