variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "project_name" {
  type    = string
  default = "comics-manga"
}

variable "bucket" {
  type    = string
  default = "projet-terraform"
}

# doit rester egal a IMAGE_SIZE dans scripts/features.py
variable "image_size" {
  type    = number
  default = 64
}

variable "training_image" {
  type    = string
  default = "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
}

# processing job : le quota de training job est arrive apres coup, on n'a pas rebascule
variable "training_instance_type" {
  type    = string
  default = "ml.t3.medium"
}

variable "endpoint_instance_type" {
  type    = string
  default = "ml.t2.medium"
}

variable "lambda_runtime" {
  type    = string
  default = "python3.13"
}

# couche managee AWS SDK for pandas ; version non listable, a fixer a la main
variable "pandas_layer_version" {
  type    = number
  default = 14
}

variable "log_retention_days" {
  type    = number
  default = 14
}

# reprend la state machine et la lambda creees a la console ; false sur un compte vierge
variable "import_existing_resources" {
  type    = bool
  default = true
}

# « proprietaire/depot » ; vide desactive toute la partie CI/CD
variable "github_repository" {
  type    = string
  default = ""
}

# false si le fournisseur OIDC github existe deja dans le compte
variable "create_github_oidc_provider" {
  type    = bool
  default = true
}
