import os
import logging
import flask
import pandas as pd
import prometheus_client
from flask import Flask, request, send_file
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from utils import (
    load_config, 
    build_data_definition,
    get_remote_data_defintion,
    get_remote_reference_data,
    create_report,
    run_evidently,
    Dataset,
    compute_hash)


# ------------------------------------------------------------------------------
# Flask + Prometheus setup
# ------------------------------------------------------------------------------

# Create a Flask Application
app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],#sends log messages to the console (stdout)
)

# Add Prometheus Metrics Endpoint
# Create a WSGI application for serving Prometheus metrics
prometheus_metrics_app = prometheus_client.make_wsgi_app()

# Combine the Flask app with the Prometheus metrics app using DispatcherMiddleware
# This allows the Flask app to run alongside the Prometheus metrics endpoint.
# Any request to /metrics will be routed to the prometheus_metrics_app
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app, {"/metrics": prometheus_metrics_app}
)


def init_evidently():
    """Load config, reference data, and initialize Evidently report."""
    logging.info("Initializing Evidently monitoring service...")

    config = load_config()
    mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    run_id = os.getenv("RUN_ID", "052cdec46b664b61a2f471648fb4e395")


    # data_definition = build_data_definition(config)
    data_definition = get_remote_data_defintion(mlflow_url, run_id)

    # reference_path = config['service']['reference_path']
    # reference_df = pd.read_csv(reference_path)
    reference_df = get_remote_reference_data(mlflow_url, run_id)

    reference_data = Dataset.from_pandas(data=reference_df, data_definition=data_definition)
    
    window_size = config["service"].get("window_size", 250)

    ref_hash = compute_hash(reference_df)
    logging.info(f"Reference data hash: {ref_hash}")

    # Export to prometheus gauge
    ref_hash_metric = prometheus_client.Gauge(
        'evidently_reference_dataset_hash',
        'Hash of the reference dataset used for monitoring',
        labelnames=['hash']
    )
    ref_hash_metric.labels(hash=ref_hash).set(1)

    # Create Evidently report
    report = create_report(config)

    app.config.update({
        "EVIDENTLY_CONFIG": config,
        "DATA_DEFINITION": data_definition,
        "REFERENCE_DATA": reference_data,
        "REPORT": report,
        "WINDOW_SIZE": window_size,
    })

    logging.info("Evidently initialized successfully.")
    logging.info(f"Reference dataset rows: {len(reference_df)}")
    


# Run initialization immediately at startup
init_evidently()

current_data = pd.DataFrame()
gauge_store = {}

@app.route("/iterate/<dataset>", methods=["POST"])
def iterate(dataset):
    global current_data
    global gauge_store

    # Step 1: Get incoming data from the request
    new_row = request.json
    incoming_df = pd.DataFrame([new_row])
    

    # Step 2: Append incoming data to current_data
    current_data = pd.concat([current_data, incoming_df], ignore_index=True)

    window_size = app.config["WINDOW_SIZE"]
    # Keep only the last N rows (sliding window)
    if len(current_data) > window_size:
        current_data = current_data.tail(window_size).reset_index(drop=True)
    
    # Log progress
    logging.info(f"Buffered {len(current_data)}/{window_size} rows")

    if len(current_data) < window_size:
        remaining = window_size - len(current_data)
        logging.info(f"Waiting for {remaining} more rows before running analysis.")
        return f"Buffering data... {len(current_data)}/{window_size}, 200"

    # Run Evidently drift analysis
    logging.info("Window filled — starting drift analysis.")
    # Step 3: Run Evidently report with updated current_data
    run_evidently(
        reference_data=app.config["REFERENCE_DATA"],
        current_data=current_data,
        data_definition=app.config["DATA_DEFINITION"],
        report=app.config["REPORT"],
        gauge_store=gauge_store,
        dataset_name=dataset
    )

    return 'ok'


@app.route("/report", methods=["GET"])
def view_report():
    if not os.path.exists("latest_report.html"):
        return "Report not generated yet", 404
    return send_file("latest_report.html")

if __name__ == "__main__":
    app.run(debug=True)