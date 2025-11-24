variable "project_name" {
  type        = string
  description = "Project name"
}

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "org_id" {
  type        = string
  description = "Organization ID"
  default     = null
}

variable "billing_account" {
  type        = string
  description = "Billing account ID"
  default     = null
}

variable "create_project" {
  type        = bool
  default     = false
}

variable "region" {
  type    = string
  default = "europe-west1"
}

variable "zone" {
  type    = string
  default = "europe-west1-b"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "my_ip" {
  type        = string
  description = "Public IP of your host with /32, e.g. 203.0.113.5/32"
}

variable "service_account_email" {
  type        = string
  description = "Service account email with required IAM permissions"
}

variable "db_user" {}
variable "db_name" {}

