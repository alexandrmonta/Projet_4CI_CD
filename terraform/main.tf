terraform {
  required_version = ">= 1.5"

  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 6.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.4" }
  }

  backend "s3" {
    bucket       = "projet-terraform"
    key          = "terraform/ml-pipeline.tfstate"
    region       = "eu-central-1"
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project_name
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------- s3

# le bucket preexiste : on le reference, on ne le gere pas
data "aws_s3_bucket" "projet" {
  bucket = var.bucket
}

resource "aws_s3_object" "scripts" {
  for_each = toset(["train.py", "inference.py", "features.py"])

  bucket = data.aws_s3_bucket.projet.id
  key    = "scripts/${each.value}"
  source = "${path.module}/../scripts/${each.value}"
  etag   = filemd5("${path.module}/../scripts/${each.value}")
}

# ---------------------------------------------------------------- lambdas

# contenu produit par tools/build_lambdas.sh
data "archive_file" "process_image" {
  type        = "zip"
  source_dir  = "${path.module}/../build/lambda/process_image"
  output_path = "${path.module}/../build/process_image.zip"
}

data "archive_file" "run_inference" {
  type        = "zip"
  source_dir  = "${path.module}/../build/lambda/run_inference"
  output_path = "${path.module}/../build/run_inference.zip"
}

resource "aws_lambda_function" "process_image" {
  function_name = "process_image"
  role          = aws_iam_role.lambda.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = var.lambda_runtime
  architectures = ["x86_64"]

  filename         = data.archive_file.process_image.output_path
  source_code_hash = data.archive_file.process_image.output_base64sha256

  memory_size = 2048
  timeout     = 900

  environment {
    variables = {
      IMAGE_SIZE  = tostring(var.image_size)
      MAX_WORKERS = "16"
    }
  }
}

resource "aws_lambda_function" "run_inference" {
  function_name = "${var.project_name}-run-inference"
  role          = aws_iam_role.lambda.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = var.lambda_runtime
  architectures = ["x86_64"]

  filename         = data.archive_file.run_inference.output_path
  source_code_hash = data.archive_file.run_inference.output_base64sha256

  memory_size = 1024
  timeout     = 900

  # pandas et pyarrow, trop lourds pour l'archive
  layers = ["arn:aws:lambda:${var.region}:336392948345:layer:AWSSDKPandas-Python313:${var.pandas_layer_version}"]

  environment {
    variables = { MAX_WORKERS = "16" }
  }
}

# ---------------------------------------------------------------- step functions

# a creer explicitement : step functions ne cree pas ses groupes vendedlogs
resource "aws_cloudwatch_log_group" "state_machine" {
  name              = "/aws/vendedlogs/states/${var.project_name}-ml-pipeline"
  retention_in_days = var.log_retention_days
}

resource "aws_sfn_state_machine" "ml_pipeline" {
  name     = "ML_Pipeline"
  role_arn = aws_iam_role.states.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/../statemachine/ml_pipeline.asl.json", {
    bucket                 = var.bucket
    region                 = var.region
    process_image_arn      = aws_lambda_function.process_image.arn
    run_inference_arn      = aws_lambda_function.run_inference.arn
    sagemaker_role_arn     = aws_iam_role.sagemaker.arn
    training_image         = var.training_image
    training_instance_type = var.training_instance_type
    endpoint_instance_type = var.endpoint_instance_type
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.state_machine.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  depends_on = [aws_iam_role_policy.states]
}

# ---------------------------------------------------------------- reprise console

# adopte les ressources creees a la main au lieu de les dupliquer
import {
  for_each = var.import_existing_resources ? toset(["ML_Pipeline"]) : toset([])

  to = aws_sfn_state_machine.ml_pipeline
  id = "arn:aws:states:${var.region}:${data.aws_caller_identity.current.account_id}:stateMachine:${each.value}"
}

import {
  for_each = var.import_existing_resources ? toset(["process_image"]) : toset([])

  to = aws_lambda_function.process_image
  id = each.value
}

# ---------------------------------------------------------------- ci/cd

locals {
  github_enabled = var.github_repository != ""

  # github insere depuis peu les identifiants numeriques du proprietaire et du
  # depot dans le claim sub : on accepte l'ancien et le nouveau format
  github_parts = split("/", var.github_repository)
  github_subs = [
    "repo:${var.github_repository}:*",
    "repo:${local.github_parts[0]}@*/${try(local.github_parts[1], "")}@*:*",
  ]
  github_oidc_arn = var.create_github_oidc_provider && local.github_enabled ? one(aws_iam_openid_connect_provider.github[*].arn) : (
    "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"
  )
}

resource "aws_iam_openid_connect_provider" "github" {
  count = local.github_enabled && var.create_github_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github" {
  count = local.github_enabled ? 1 : 0

  name = "${var.project_name}-github-actions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = local.github_oidc_arn }
      Condition = {
        StringEquals = { "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com" }
        StringLike   = { "token.actions.githubusercontent.com:sub" = local.github_subs }
      }
    }]
  })
}

# le perimetre des modifications n'est pas connu a l'avance : la ci doit pouvoir
# creer n'importe quel service. poweruser couvre tout sauf iam.
resource "aws_iam_role_policy_attachment" "github" {
  count = local.github_enabled ? 1 : 0

  role       = aws_iam_role.github[0].name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

# le complement iam : les roles, jamais les utilisateurs ni l'organisation
resource "aws_iam_role_policy" "github" {
  count = local.github_enabled ? 1 : 0

  name = "gestion-des-roles"
  role = aws_iam_role.github[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "iam:*Role*",
        "iam:*RolePolicy*",
        "iam:*InstanceProfile*",
        "iam:*OpenIDConnectProvider*",
        "iam:CreatePolicy*",
        "iam:DeletePolicy*",
        "iam:Get*",
        "iam:List*",
        "iam:Tag*",
        "iam:Untag*",
      ]
      Resource = "*"
    }]
  })
}

# ---------------------------------------------------------------- demo

