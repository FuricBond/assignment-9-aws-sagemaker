# Train and Deploy a Machine Learning Model using AWS SageMaker

**Coursework:** Third Year B.Tech (AI & DS) - SageMaker end-to-end ML pipeline.

The project follows the supplied eight-page assignment. The project is **research/coursework only**, not a medical diagnostic tool. All AWS operations are separately invoked; simply opening files or running the local tests does not incur AWS charges.

## Windows installation at your requested location

The exact requested destination is `D:\SEM 5\MLAO\Assignmet 9(Sage maker)`, preserving the folder name as entered. Put `Assignment_9_AWS_SageMaker_Project.zip` and `Install_Assignment_9_Windows.ps1` together in Downloads, open PowerShell there, and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Install_Assignment_9_Windows.ps1 -CaptureScreenshot
```

This copies project files into that folder, sets up `.venv`, re-runs local training/tests, and (after asking you to arrange the displayed LOCAL evidence page) captures a real screen image in `screenshots/local_evaluation_actual_screen.png`. The screenshot is optional; omit `-CaptureScreenshot` to skip it. The script stops rather than overwriting a nonempty existing destination. It does **not** run paid AWS steps.

## What is implemented

| Requirement | Implementation | Evidence location |
|---|---|---|
| Wisconsin diagnostic data; 569 rows, 30 features | `prepare_data.py`; fixed 80:20 stratified split, seed 42 | `artifacts/data/dataset_summary.json` |
| Random Forest in SageMaker script mode | `train.py` implements training, `model_fn`, `input_fn`, `predict_fn`, `output_fn` | `artifacts/rf_*/training_log.txt` for **local** runs |
| Compare 100 trees/depth 4 vs. 200 trees/depth 8 | `evaluate_local.py` and two `cloud_pipeline.py train` jobs | `artifacts/hyperparameter_comparison.csv` |
| Accuracy, precision, recall, F1, ROC-AUC | `train.py`, plus malignant-class recall / false benign count | `artifacts/rf_*/results/metrics.json` |
| Confusion matrix and ROC | `evaluate_local.py` | `artifacts/confusion_matrices.png`, `artifacts/roc_curves.png` |
| Encrypted private S3 + SageMaker IAM execution role | `cloud_pipeline.py provision` | AWS S3 console screenshot **after real deployment** |
| SageMaker managed training & model artifact in S3 | `cloud_pipeline.py train` | AWS Training jobs screenshot **after real deployment** |
| Real-time endpoint and test prediction | `cloud_pipeline.py deploy` | AWS endpoint InService screenshot **after real deployment** |
| Lambda and API Gateway `/predict` | `lambda_function.py`, `cloud_pipeline.py publish-api`, `test-api` | AWS API test screenshot **after real deployment** |
| CloudWatch metric monitoring and 5XX alarm | `cloud_pipeline.py monitor` | AWS CloudWatch screenshot **after real deployment** |
| API IAM authorization and throttle | API Gateway `AWS_IAM` + 2 rps/5 burst; Lambda invokes only named endpoint | API Gateway method/stage screenshots |
| Explicit teardown | `cloud_pipeline.py cleanup --confirm` | Save endpoint/API deletion screenshots |

**Important evidence distinction:** the PNG plots and `local_evidence.html` show verified *local execution*. They cannot establish that an AWS job, endpoint, Lambda or alarm was created. Those screenshots must be captured in the owner's AWS Console during an actual run; see `screenshots/AWS_SCREENSHOT_CHECKLIST.md`.

## 1. Run and validate locally (no AWS account required)

From the project directory:

```bash
python -m pip install -r requirements-local.txt
python evaluate_local.py --output-dir artifacts
python -m unittest discover -s tests -v
python local_api_demo.py
python make_evidence.py
```

The labelled `LOCAL` output can be inspected using `artifacts/local_evidence.html`; the confusion matrix and ROC plots are genuine model outputs. The class mapping is **0 = malignant; 1 = benign**. Precision, recall, and F1 in the assignment use benign as the positive class. Because a **false benign** is clinically costly, the code also records recall for malignant samples and the number of malignant samples predicted benign. Neither the model nor these results should be used for patient care.

## 2. AWS prerequisites and cost consent

Use an AWS account and credentials with permissions for SageMaker AI, IAM role creation and `iam:PassRole`, S3, Lambda, API Gateway and CloudWatch. Select a region with the requested SageMaker quota and a matching ECR scikit-learn image; example `ap-south-1`. The code never contains AWS access keys. On your workstation, configure an AWS profile with IAM Identity Center or another supported AWS credential method. Test `aws sts get-caller-identity` in your own terminal before running scripts (if you have AWS CLI).

**Running managed training and a real-time endpoint incurs AWS charges.** Confirm current regional prices and service quotas before starting. `deploy` starts a continuously billed instance. Take the screenshots immediately, then run cleanup. Keep the estimated running time short, and configure a Budget alert in AWS Billing. Bucket objects, CloudWatch logs, stored training artifacts and some other account resources remain after automatic cleanup.

This project uses SageMaker Python SDK v2 as requested by the PDF (`pip install "sagemaker<3"`). The PDF's scikit-learn image `1.2-1` is now outside AWS's patch-support window. The cloud command therefore defaults to the newer `1.9-0` container advertised by AWS's image support table; if your region/SDK combination rejects that image, check the region's supported tags and pass the correct version explicitly. Do not use an unsupported image for sensitive production work.

```bash
python -m pip install -r requirements-aws.txt
```

## 3. AWS implementation commands

Replace `roll123` with your own distinct short identifier, and use a new globally unique bucket name. Execute **one command at a time**, checking its output and taking the screenshots called out below.

```bash
# Creates a new private AES256-encrypted bucket and SageMaker execution IAM role.
python cloud_pipeline.py provision --bucket ty-aids-sagemaker-roll123 --tag roll123 --region ap-south-1

# Uploads train/test data, starts TWO separate managed jobs, records model.tar.gz,
# FinalMetricDataList, actual TrainingTimeInSeconds and BillableTimeInSeconds.
python cloud_pipeline.py train

# Creates a billable real-time endpoint; tests it with real 30-value inference.
python cloud_pipeline.py deploy

# Creates an API with AWS_IAM authorization and stage throttling.
python cloud_pipeline.py publish-api

# Calls POST /predict with AWS SigV4; stores real response JSON.
python cloud_pipeline.py test-api

# Creates an Invocation5XXErrors alarm (notification is disabled by default).
python cloud_pipeline.py monitor

# Fetches real ModelLatency and OverheadLatency datapoints, among other metrics.
python cloud_pipeline.py metrics --minutes 15

# Print resource names and live endpoint state.
python cloud_pipeline.py status

# Only AFTER capturing all screenshots; deletes tracked endpoint/API resources.
python cloud_pipeline.py cleanup --confirm
```

AWS state and exact cloud resource names are recorded in `cloud_state.json` **only after commands actually run**. Do not edit or delete that file while cloud resources exist; the cleanup command depends on it. Run the `status` command and verify the AWS console is `InService` before capturing endpoint evidence.

The public API URL is **IAM-protected**, so a plain unsigned curl call receives `403`. The included `test-api` command signs its request using configured credentials. Only principals allowed `execute-api:Invoke` can call it. Lambda's execution role is restricted to `sagemaker:InvokeEndpoint` for the specific endpoint. The REST API also limits stage traffic to 2 requests per second with a burst of 5; lower these limits if needed. The 5XX alarm is created without notification delivery because no verified SNS destination was provided; optionally connect a verified SNS topic from the console.

## 4. Submission screenshots and console navigation

Follow `screenshots/AWS_SCREENSHOT_CHECKLIST.md` for the specific 8-10 screenshots, suggested captions and the order of collection, including **proof of deletion after testing**. Never present local generated plots as AWS Console screenshots.

## 5. Notebook

`SageMaker_Assignment.ipynb` lays out the architecture and safe run order. You may use it inside SageMaker Studio, but managed resource creation is intentionally not triggered by merely opening the notebook.

## 6. Monitoring metrics to document

In CloudWatch, use the `AWS/SageMaker` namespace and the relevant EndpointName/VariantName dimensions:

- `Invocations` - request volume.
- `ModelLatency` - container inference time, in microseconds.
- `OverheadLatency` - SageMaker overhead, in microseconds.
- `Invocation4XXErrors` and `Invocation5XXErrors` - client/server-side failures.
- The `Invocation5XXErrors` alarm state, and logs in `/aws/sagemaker/TrainingJobs` or `/aws/sagemaker/Endpoints/<endpoint>`.

SageMaker Console `Training jobs` shows job status, duration and billable seconds. Do **not** substitute local Python timing for cloud billable seconds. The `metrics` command saves `aws_cloudwatch_metrics.json` only after a real query; retry after several minutes if CloudWatch has not published data.

## 7. Troubleshooting

| Symptom | Suggested check |
|---|---|
| `NoCredentialsError` | Sign in through AWS credential provider or correct your profile. |
| `AccessDenied` for `iam:PassRole` | Grant the calling user permission to pass only the coursework role to SageMaker/Lambda. |
| Image tag unavailable | Check supported SageMaker scikit-learn image tags in your region and select `--framework-version`. |
| `ResourceLimitExceeded` | Request a small training/endpoint instance quota or use an approved region. |
| IAM role not assumable yet | Wait 10-30 seconds after `provision` before starting managed training. |
| API returns `403` | It requires AWS_IAM SigV4, not an anonymous request; ensure caller can `execute-api:Invoke`. |
| Lambda returns `502` | Inspect Lambda logs, SageMaker endpoint InService state, input format, invoke IAM permission. |
| Cleanup incomplete | Check `cloud_state.json`, SageMaker Endpoint/EndpointConfig/Models, API Gateway and Lambda; delete tracked resources from console. |

## Sources

- [SageMaker scikit-learn resources](https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html)
- [AWS prebuilt container support policy](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-support-policy.html)
- [SageMaker real-time endpoint cleanup](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-delete-resources.html)
- [API Gateway IAM authorization](https://docs.aws.amazon.com/apigateway/latest/developerguide/permissions.html)
- [API Gateway throttling](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html)
- [SageMaker endpoint monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)

