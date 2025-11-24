from pydantic import BaseModel
from typing import Any, List, Optional, Dict

""" class TaxiRide(BaseModel):
    PULocationID: float
    DOLocationID: float
    trip_distance: float
    passenger_count: float
    fare_amount: float
    total_amount: float """

class PredictRequest(BaseModel):
    run_id: str
    model_name: str
    data: Dict[str, Any] 
    
class Prediction(BaseModel):
    data: Dict[str, Any]
    prediction: float

class MlFlowConfig(BaseModel):
    url: str
    bucket_location: str

class ExperimentRequest(BaseModel):
    experiment_name: str
    tags: Dict[str, str]

class LogModelParams(BaseModel):
    test_size: float
    target_column: str
    predict_column:str = "prediction"
    experiment_name:str
    experiment_id:int
    model_name:str
    tags: Optional[Dict[str, str]] = None
    numerical_columns:List[str] = []
    categorical_columns:List[str] = []



class ColumnMapping(BaseModel):
    categorical: List[str] = []  # default to empty list
    numerical: List[str] = [] 
    target: Optional[str] 

class LoggedModelResponse(BaseModel):
    model_id: str
    experiment_id: str
    name: str
    artifact_location: str
    creation_timestamp: int
    last_updated_timestamp: int
    model_type: str
    source_run_id: str
    status: str
    status_message: str
    tags: Dict[str, str]
    params: Dict[str, str]
    metrics: Optional[Dict[str, Any]]
    columnMapping: ColumnMapping
    version: List[Dict[str, Any]]