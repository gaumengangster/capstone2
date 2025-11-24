# MLflow server public URL
output "mlflow_server_url" {
  value       = "http://${google_compute_instance.mlflow_server.network_interface[0].access_config[0].nat_ip}:5000"
  description = "URL to access the MLflow Tracking Server"
}

# PostgreSQL connection string (Cloud SQL)
output "postgresql_connection_name" {
  value       = google_sql_database_instance.mlflow_db.connection_name
  description = "The connection name for the PostgreSQL Cloud SQL instance (for use with Cloud SQL Proxy)"
}

output "postgresql_host" {
  value       = google_sql_database_instance.mlflow_db.public_ip_address
  description = "Public IP address of the PostgreSQL database instance"
}

output "postgresql_connection_url" {
  # Substitute in your actual DB name, username, and password if desired
  value       = "postgresql://${google_sql_user.mlflow_user.name}:${var.db_password}@${google_sql_database_instance.mlflow_db.public_ip_address}:5432/${google_sql_database.mlflow_db_name.name}"
  description = "Connection URL for PostgreSQL (use with care — includes credentials if you embed them)"
  sensitive   = true
}

# GCS bucket URL for MLflow artifacts
output "mlflow_artifact_bucket_url" {
  value       = "gs://${google_storage_bucket.mlflow_artifacts.name}"
  description = "Google Cloud Storage bucket for MLflow artifacts"
}
