terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  required_version = ">= 1.5.0"
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Create a new GCP Project (optional if not existing)
resource "google_project" "mlflow_project" {
  count       = var.create_project ? 1 : 0
  name        = var.project_name
  project_id  = var.project_id
  org_id      = var.org_id
  billing_account = var.billing_account
}



# Enable necessary APIs
resource "google_project_service" "compute" {
  service = "compute.googleapis.com"
  disable_on_destroy          = false
}

resource "google_project_service" "sqladmin" {
  service = "sqladmin.googleapis.com"
}

/* resource "google_project_service" "storage" {
  project                    = var.project_id
  service = "storage.googleapis.com"
  disable_dependent_services  = true
} */
resource "google_project_service" "storage" {
  project = var.project_id
  service = "storage.googleapis.com"
  disable_dependent_services=true
  # lifecycle {
  #   prevent_destroy = true
  # }
}






# Create firewall rule to allow your host
resource "google_compute_firewall" "mlflow_tracking" {
  name        = "mlflow-tracking-server"
  description = "Allow inbound traffic for MLflow tracking server on TCP port 5000"
  network     = "default"

  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["mlflow-tracking-server"]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

data "google_compute_network" "default" {
  name = "default"
}


resource "google_sql_database_instance" "mlflow_db" {
  name             = "mlflow-metadata-store"
  region           = "europe-west3"
  database_version = "POSTGRES_17"
  project          = var.project_id
  lifecycle {
    prevent_destroy = true
  }
  settings {
    edition  = "ENTERPRISE"
    tier = "db-f1-micro"  # Sandbox preset for Enterprise edition

    ip_configuration {
      ipv4_enabled = true

      authorized_networks {
        name  = "allow-mlflow"
        value = var.my_ip  # your host IP for public access if needed
      }

      authorized_networks {
        name  = "allow-mlflow-vm"
        value = google_compute_address.mlflow_vm_ip.address
      }
    }

    backup_configuration {
      enabled = false  # matches your previous request
    }

    disk_type           = "PD_SSD"
    disk_size           = 10
    availability_type   = "ZONAL"
  }

  deletion_protection = false
}


# Create the database
resource "google_sql_database" "mlflow_db_name" {
  name     = "mlflow-db"
  instance = google_sql_database_instance.mlflow_db.name
  lifecycle {
    prevent_destroy = true
  }
}

# Create the user
resource "google_sql_user" "mlflow_user" {
  name     = "mlflow-user"
  instance = google_sql_database_instance.mlflow_db.name
  password = "mlflowuserpassord11"
  lifecycle {
    prevent_destroy = true
  }
}


# Create GCS bucket for artifacts
# Generate a random string to make the bucket name unique
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Create the GCS bucket
resource "google_storage_bucket" "mlflow_artifacts" {
  name     = "mlflow-artifacts-${random_id.bucket_suffix.hex}"
  location = "europe-west3"
  force_destroy = true  # Allows bucket to be deleted even if not empty
  uniform_bucket_level_access = true
}

resource "google_compute_address" "mlflow_vm_ip" {
  name   = "mlflow-vm-ip"
  region = "europe-west3"
}


# Create Compute Engine instance
resource "google_compute_instance" "mlflow_server" {
  name         = "mlflow-tracking-server"
  machine_type = "e2-medium"
  zone         = "europe-west3-c"

  tags = ["mlflow-tracking-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      type  = "pd-standard"
      size  = 10
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.mlflow_vm_ip.address
    }
  }

  service_account {
    email  = var.service_account_email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e

    # Update and install dependencies
    sudo apt-get update
    sudo apt-get install -y postgresql-client git python3-pip make build-essential \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl \
      llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

    sudo mkdir -p /home/ubuntu/logs
    sudo chown ubuntu:ubuntu /home/ubuntu/logs
    chmod 755 /home/ubuntu/logs
    
    sudo apt-get install -y python3.10-venv python3-pip
    sudo -u ubuntu python3 -m venv /home/ubuntu/mlflow
    sudo -u ubuntu /bin/bash -c "source /home/ubuntu/mlflow/bin/activate && pip install --upgrade pip && pip install mlflow boto3 google-cloud-storage psycopg2-binary"


    # Start MLflow server
    sudo -u ubuntu /bin/bash -c "nohup /home/ubuntu/mlflow/bin/mlflow server \
      --backend-store-uri postgresql+psycopg2://${var.db_user}:${var.db_password}@${google_sql_database_instance.mlflow_db.ip_address[0].ip_address}:5432/${var.db_name} \
      --default-artifact-root gs://${google_storage_bucket.mlflow_artifacts.name} \
      --host 0.0.0.0 \
      --port 5000 \
      --workers 2 \
      --allowed-hosts='*' \
      --cors-allowed-origins='*' \
      > /home/ubuntu/logs/mlflow.log 2>&1 &"
  EOF
}


output "mlflow_url" {
  value = "http://${google_compute_instance.mlflow_server.network_interface[0].access_config[0].nat_ip}:5000"
}
