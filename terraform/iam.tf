locals {
  bucket_arn = "arn:aws:s3:::${var.bucket}"

  # une politique de confiance par service, meme forme a chaque fois
  assume = { for service in ["lambda", "states", "sagemaker"] : service => jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "${service}.amazonaws.com" }
    }]
  }) }
}

# ---------------------------------------------------------------- lambdas

resource "aws_iam_role" "lambda" {
  name               = "${var.project_name}-lambda"
  assume_role_policy = local.assume["lambda"]
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda" {
  name = "acces-pipeline"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = local.bucket_arn
      },
      {
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = ["${local.bucket_arn}/train/*", "${local.bucket_arn}/test/*", "${local.bucket_arn}/processed/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:DeleteObject"]
        Resource = ["${local.bucket_arn}/processed/*", "${local.bucket_arn}/output/*"]
      },
      {
        Effect   = "Allow"
        Action   = "sagemaker:InvokeEndpoint"
        Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:endpoint/${var.project_name}-ep-*"
      },
    ]
  })
}

# ---------------------------------------------------------------- sagemaker

# AmazonSageMakerFullAccess ne donne S3 que sur les buckets nommes « *sagemaker* »
resource "aws_iam_role" "sagemaker" {
  name               = "${var.project_name}-sagemaker-execution"
  assume_role_policy = local.assume["sagemaker"]
}

resource "aws_iam_role_policy" "sagemaker" {
  name = "acces-pipeline"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = local.bucket_arn
      },
      {
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = ["${local.bucket_arn}/scripts/*", "${local.bucket_arn}/processed/*", "${local.bucket_arn}/models/*"]
      },
      {
        Effect   = "Allow"
        Action   = "s3:PutObject"
        Resource = "${local.bucket_arn}/models/*"
      },
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
        Resource = "*"
      },
    ]
  })
}

# ---------------------------------------------------------------- step functions

resource "aws_iam_role" "states" {
  name               = "${var.project_name}-stepfunctions"
  assume_role_policy = local.assume["states"]
}

resource "aws_iam_role_policy" "states" {
  name = "acces-pipeline"
  role = aws_iam_role.states.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "lambda:InvokeFunction"
        Resource = ["${aws_lambda_function.process_image.arn}*", "${aws_lambda_function.run_inference.arn}*"]
      },
      {
        Effect   = "Allow"
        Action   = ["sagemaker:Create*", "sagemaker:Describe*", "sagemaker:Delete*", "sagemaker:Stop*", "sagemaker:AddTags", "sagemaker:ListTags"]
        Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:*"
      },
      {
        # sans PassRole, step functions ne peut pas confier le role au service sagemaker
        Effect    = "Allow"
        Action    = "iam:PassRole"
        Resource  = aws_iam_role.sagemaker.arn
        Condition = { StringEquals = { "iam:PassedToService" = "sagemaker.amazonaws.com" } }
      },
      {
        # regle managee posee par le motif « .sync »
        Effect   = "Allow"
        Action   = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
        Resource = "arn:aws:events:${var.region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery", "logs:DeleteLogDelivery", "logs:ListLogDeliveries", "logs:PutResourcePolicy", "logs:DescribeResourcePolicies", "logs:DescribeLogGroups"]
        Resource = "*"
      },
    ]
  })
}
