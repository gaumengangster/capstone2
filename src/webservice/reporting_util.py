from typing import List
from sqlalchemy import create_engine
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount
import pandas as pd
from datetime import datetime

def load_report_into_sql(result):
    """Load the report into the Postgres database."""
    engine = create_engine("postgresql://mlflow-user:mlflowuserpassord11@34.185.233.76:5432/mlflow-db")

    report_dict = {
        "timestamp": datetime.now(),
        "prediction_drift": result['metrics'][0]['value'],
        "num_drifted_columns": result['metrics'][1]['value']['count'],
        "share_missing_values": result['metrics'][2]['value']['share']}

    df_report = pd.DataFrame([report_dict])
    df_report.to_sql(name="drift_metrics", con=engine, if_exists="append", index=False)

def prepare_evidently_dataset(df: pd.DataFrame, num_features: List[str], cat_features: List[str], pred_col: str) -> Dataset:
    """Prepare an Evidently Dataset from a pandas DataFrame."""
    data_definition = DataDefinition(numerical_columns=num_features + [pred_col], categorical_columns=cat_features)
    evidently_dataset = Dataset.from_pandas(df, data_definition)
    return evidently_dataset

def create_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, num_features: List[str] , cat_features: List[str], pred_col: str = 'prediction') -> dict:
    """Create an Evidently report comparing reference and current datasets."""

    reference_dataset = prepare_evidently_dataset(reference_data, num_features, cat_features, pred_col)
    current_dataset = prepare_evidently_dataset(current_data, num_features, cat_features, pred_col)

    report = Report(metrics=[
        ValueDrift(column='prediction'), 
        DriftedColumnsCount(),
        MissingValueCount(column='prediction')], include_tests=True)

    result = report.run(reference_data=reference_dataset, current_data=current_dataset).dict()
    return result