output "state_machine_arn" {
  value = aws_sfn_state_machine.ml_pipeline.arn
}

output "start_execution_command" {
  value = "aws stepfunctions start-execution --region ${var.region} --state-machine-arn ${aws_sfn_state_machine.ml_pipeline.arn} --input '{}'"
}

output "lambda_functions" {
  value = [aws_lambda_function.process_image.function_name, aws_lambda_function.run_inference.function_name]
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "github_actions_role_arn" {
  value = local.github_enabled ? aws_iam_role.github[0].arn : "(desactive)"
}
