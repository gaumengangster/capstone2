"""This module contains utility functions for Evidently monitoring service."""

import os
import re
import yaml
import logging
import hashlib
import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount
import prometheus_client

import mlflow
from mlflow import MlflowClient

import json
import pandas as pd


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load Evidently configuration from YAML.
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, "rb") as file:
        config = yaml.safe_load(file)
    
    logging.info(f"Configuration loaded from {config_path}.")
    return config


def build_data_definition(config: dict) -> DataDefinition:
    """
    Build Evidently DataDefinition
    """
    colmap = config['column_mapping']

    num_features = colmap['numerical_features']
    cat_features = colmap['categorical_features']
    prediction = colmap['prediction']

    # Define data schema
    data_definition = DataDefinition(
        numerical_columns=num_features + [prediction],
        categorical_columns=cat_features)
    
    return data_definition

def get_remote_data_defintion(ml_flow_url:str, run_id:str) -> DataDefinition:
    mlflow.set_tracking_uri(ml_flow_url)
    client = MlflowClient()

    local_features_path = client.download_artifacts(run_id, "features.json")
    print("Downloaded features.json to:", local_features_path)

    with open(local_features_path) as f:
        features_dict = json.load(f)
    print(features_dict)

    num_features = features_dict["features_numerical"]
    cat_features = features_dict["features_categorical"]
    prediction_column = features_dict["prediction"]

    data_definition = DataDefinition(
        numerical_columns=num_features + [prediction_column],
        categorical_columns=cat_features)
    
    return data_definition


def get_remote_reference_data(ml_flow_url:str, run_id:str) -> DataDefinition:
    # ID run-a koji želiš da preuzmeš
    mlflow.set_tracking_uri(ml_flow_url)
    client = MlflowClient()

    local_xtrain_path = client.download_artifacts(run_id, "X_train_reference.csv")
    print("Downloaded X_train_reference.csv to:", local_xtrain_path)

    X_train_ref = pd.read_csv(local_xtrain_path)
    print(X_train_ref.head())

    return X_train_ref


def create_report(config) -> Report:
    """
    Create Evidently Report based on configuration.
    """
    report = Report(metrics=[
        ValueDrift(column='prediction'),
        DriftedColumnsCount(),
        MissingValueCount(column='prediction')])
    logging.info("Evidently Report initialised.")
    return report

def extract_metrics_from_snapshot(snapshot):
    """
    Convert snapshot.dict()['metrics'] into a flat dict of metric_name → value.
    If a column name is present (e.g. column=prediction), include it in the key.
    """
    metrics_list = snapshot.dict().get("metrics", [])
    results = {}

    for metric in metrics_list:
        metric_id = metric.get("metric_id", "")
        value = metric.get("value")

        # Base metric name, e.g. "ValueDrift"
        base_name = metric_id.split("(")[0]

        # Extract column name if present: (column=prediction)
        col_match = re.search(r"column=([\w\d_]+)", metric_id)
        column_suffix = f"_{col_match.group(1)}" if col_match else ""

        # Build prefix for this metric
        metric_prefix = f"{base_name}{column_suffix}"

        # Handle numeric or dict value
        if isinstance(value, dict):
            for key, val in value.items():
                metric_key = f"{metric_prefix}_{key}"
                try:
                    results[metric_key] = float(val)
                except Exception:
                    results[metric_key] = val
        else:
            try:
                results[metric_prefix] = float(value)
            except Exception:
                results[metric_prefix] = value

    return results

def update_prometheus_metrics(metrics_dict, gauge_store, dataset_name="default_dataset"):
    """
    Update or register Prometheus Gauges dynamically with dataset_name label.
    
    Args:
        metrics_dict: dict of {metric_name: value}
        gauge_store: dict to cache prometheus_client.Gauge objects
        dataset_name: str name of dataset (e.g. 'green_taxi_data')
    """
    for name, value in metrics_dict.items():
        gauge_name = f"evidently_{name.lower()}"

        # Create the gauge if it doesn't exist yet
        if name not in gauge_store:
            gauge_store[name] = prometheus_client.Gauge(
                gauge_name,
                f"Evidently metric {name}",
                labelnames=["dataset_name"]
            )
            logging.info(f"Registered Prometheus gauge: {gauge_name}")

        # Always set value with dataset_name label
        gauge_store[name].labels(dataset_name=dataset_name).set(float(value))
        


def run_evidently(reference_data, current_data, data_definition, report, gauge_store, dataset_name='default_dataset'):
    """
    Run the Evidently report, save HTML, extract metrics,
    and update Prometheus gauges.
    """
    logging.info("Running Evidently report...")

    current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
    snapshot = report.run(reference_data=reference_data, current_data=current_dataset)
    snapshot.save_html("latest_report.html")

    metrics = extract_metrics_from_snapshot(snapshot)
    update_prometheus_metrics(metrics, gauge_store,dataset_name)

    logging.info(f"Report updated and {len(metrics)} metrics exported to Prometheus.")


def compute_hash(df):
    """Compute hash of DataFrame"""
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()






