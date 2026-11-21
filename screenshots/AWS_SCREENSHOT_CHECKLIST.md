# Screenshot plan - capture only actual AWS Console evidence

Take each screenshot with the resource name, account/region context (where feasible), and status visible. Crop out credentials, secrets and unrelated personal details. Label screenshots in order. **Do not invent or substitute evidence.**

1. **01_s3_private_bucket.png** - Amazon S3 > Buckets > `ty-aids-sagemaker-<your-tag>`; open `breast-cancer/train/train.csv`, `breast-cancer/test/test.csv`. Record public access blocked and encryption settings in one or two screenshots.
2. **02_training_jobs.png** - Amazon SageMaker AI > Training > Training jobs. Show **both** Random Forest jobs with `Completed` status.
3. **03_training_job_details.png** - Open the selected job. Show training instance, training/billable seconds, algorithm image and output artifact S3 path. Capture logged metrics from CloudWatch training logs.
4. **04_endpoint_in_service.png** - SageMaker AI > Inference > Endpoints. Show `breast-cancer-rf-<tag>` as `InService`.
5. **05_endpoint_prediction.png** - Terminal output from `python cloud_pipeline.py deploy` or saved `aws_inference_result.json` **after actual AWS invocation**. The JSON must show prediction/class_id.
6. **06_api_gateway.png** - API Gateway > APIs > your REST API > `POST /predict`. Show `AWS_IAM` authorization and `prod` stage; capture throttle setting if possible.
7. **07_api_test.png** - Actual stdout from `python cloud_pipeline.py test-api` (HTTP 200, `prediction`, `class_id`) and matching API endpoint. Protect any AWS signature headers.
8. **08_cloudwatch_metrics.png** - CloudWatch > Metrics > AWS/SageMaker > EndpointName/VariantName; show Invocations, ModelLatency, OverheadLatency, and 4XX/5XX metrics over the test window.
9. **09_cloudwatch_alarm.png** - CloudWatch > Alarms > created `bc-rf-endpoint-5xx-<tag>`, with condition/status and Lambda/endpoint logs as needed.
10. **10_cleanup_proof.png** - After `cleanup --confirm`: SageMaker Endpoints list no longer contains the coursework endpoint; API Gateway REST APIs and Lambda function are absent. Pair with `cloud_state.json` cleanup timestamp and record that retained S3/IAM/log resources may still incur storage charges.

**Order matters:** collect `04`-`09` while the billable endpoint is running, then invoke cleanup, then collect `10`. Screenshots `01`-`03` may be captured before deploying the endpoint. The local `artifacts/` charts are preliminary model evaluation figures, not proof of AWS execution.

