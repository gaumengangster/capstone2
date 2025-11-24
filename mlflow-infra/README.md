terraform init
terraform plan
terraform apply

terraform apply -auto-approve

terraform state rm google_sql_database.mlflow_db_name
terraform state rm google_sql_user.mlflow_user
terraform destroy


gcloud sql databases list --instance <INSTANCE_NAME> --project <PROJECT_ID>
gcloud sql users list --instance <INSTANCE_NAME> --project <PROJECT_ID>

terraform import google_sql_database.mlflow_db_name projects/my-project/instances/mlflow-sql/databases/mlflow_db
terraform import google_sql_user.mlflow_user projects/my-project/instances/mlflow-sql/users/mlflow_user
