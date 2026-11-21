"""Run the cloud half of the SageMaker assignment with configured AWS credentials.

Nothing executes on AWS on import. Every mutating operation requires a subcommand.
For money safety, deployment and cleanup are distinct explicit commands.
"""

import argparse
import io
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import request

from prepare_data import prepare_data


ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "cloud_state.json"
ASSUME_ROLE_VERSION = "2012-10-17"


def aws_modules(region):
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise SystemExit("Install AWS packages: pip install -r requirements-aws.txt") from exc
    return boto3.Session(region_name=region), ClientError


def load_state(path=STATE_FILE):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(state, path=STATE_FILE):
    state["state_updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def account_info(session, region):
    identity = session.client("sts").get_caller_identity()
    return identity["Account"], identity["Arn"]


def make_role(iam, role_name, service, policy_name, policy):
    trust = {
        "Version": ASSUME_ROLE_VERSION,
        "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}],
    }
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Coursework role created by SageMaker assignment",
        )
    except iam.exceptions.EntityAlreadyExistsException:
        raise RuntimeError(
            f"Role {role_name} already exists. Use a unique resource prefix; existing roles are never overwritten."
        )
    iam.put_role_policy(RoleName=role_name, PolicyName=policy_name, PolicyDocument=json.dumps(policy))
    return response["Role"]["Arn"]


def provision(args):
    session, ClientError = aws_modules(args.region)
    account, identity = account_info(session, args.region)
    state = load_state()
    if state and not state.get("cleaned_up"):
        raise RuntimeError("cloud_state.json already tracks a deployment. Do not overwrite it; clean up first.")
    bucket = args.bucket
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,55}[a-z0-9]", bucket):
        raise ValueError("Choose a new, DNS-safe, globally unique bucket name")
    if not re.fullmatch(r"[a-z0-9-]{2,16}", args.tag):
        raise ValueError("Tag must be 2-16 lowercase letters/numbers/hyphens")
    print(f"AWS caller: {identity}; region: {args.region}; new bucket: {bucket}")
    s3 = session.client("s3")
    # A unique bucket is required; never modify a bucket the user already created.
    create = {"Bucket": bucket}
    if args.region != "us-east-1":
        create["CreateBucketConfiguration"] = {"LocationConstraint": args.region}
    s3.create_bucket(**create)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    bucket_arn = f"arn:aws:s3:::{bucket}"
    s3.put_bucket_policy(
        Bucket=bucket,
        Policy=json.dumps({
            "Version": ASSUME_ROLE_VERSION,
            "Statement": [{
                "Sid": "DenyPlainHttp",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": [bucket_arn, bucket_arn + "/*"],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
            }],
        }),
    )
    scoped_policy = {
        "Version": ASSUME_ROLE_VERSION,
        "Statement": [
            {
                "Sid": "ListCourseworkBucket",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [bucket_arn],
            },
            {
                "Sid": "AccessCourseworkObjects",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": [bucket_arn + "/breast-cancer/*"],
            },
            {
                "Sid": "AuthenticateToECR",
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": "*",
            },
            {
                "Sid": "ReadSageMakerECRContainer",
                "Effect": "Allow",
                "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
                "Resource": "*",
            },
            {
                "Sid": "WriteSageMakerLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": "*",
            },
        ],
    }
    role_name = f"SageMakerAssignmentRole-{args.tag}"
    role_arn = make_role(
        session.client("iam"), role_name, "sagemaker.amazonaws.com",
        "SageMakerCourseworkAccess", scoped_policy,
    )
    state = {
        "region": args.region,
        "account_id": account,
        "bucket": bucket,
        "prefix": "breast-cancer",
        "role_arn": role_arn,
        "role_name": role_name,
        "tag": args.tag,
        "training_jobs": [],
        "models": [],
        "endpoint_configs": [],
        "cleaned_up": False,
    }
    save_state(state)
    print(f"Provisioned encrypted/private s3://{bucket}; SageMaker role {role_arn}")
    print("AWS IAM role propagation may take a short time before the first training job.")


def require_state():
    state = load_state()
    if not state or state.get("cleaned_up"):
        raise RuntimeError("Run provision first; no active deployment state exists.")
    return state


def upload_training_data(session, state):
    prepare_data(ROOT / "artifacts" / "data")
    bucket, prefix = state["bucket"], state["prefix"]
    s3 = session.client("s3")
    locations = {}
    for channel in ("train", "test"):
        key = f"{prefix}/{channel}/{channel}.csv"
        s3.upload_file(str(ROOT / "artifacts" / "data" / f"{channel}.csv"), bucket, key)
        locations[channel] = f"s3://{bucket}/{prefix}/{channel}/"
    print("Uploaded private data to", locations)
    return locations


def train(args):
    state = require_state()
    session, _ = aws_modules(state["region"])
    import sagemaker
    from sagemaker.sklearn.estimator import SKLearn

    locations = upload_training_data(session, state)
    sm_session = sagemaker.Session(boto_session=session, default_bucket=state["bucket"])
    image_uri = (
        "720646828776.dkr.ecr.ap-south-1.amazonaws.com/"
        "sagemaker-scikit-learn:1.9-0"
    )
    print(f"Selected container {image_uri}; check its support status in your region.")
    for trees, depth in ((100, 4), (200, 8)):
        # SageMaker packages source_dir into a source tarball. Isolate the actual
        # train.py to avoid copying datasets, models, screenshots, and reports.
        source_workspace = tempfile.mkdtemp(prefix="sagemaker-assign-source-")
        shutil.copy2(ROOT / "train.py", Path(source_workspace) / "train.py")
        estimator = SKLearn(
            entry_point="train.py",
            source_dir=source_workspace,
             code_location=f"s3://{state['bucket']}/{state['prefix']}/code",
            role=state["role_arn"],
            instance_type=args.train_instance,
            instance_count=1,
            framework_version=None,
            image_uri=image_uri,
            py_version="py3",
            hyperparameters={"n-estimators": trees, "max-depth": depth},
            metric_definitions=[
                {"Name": f"test:{key}", "Regex": f"test_{key}=([0-9.]+)"}
                for key in ("accuracy", "precision", "recall", "f1", "auc", "malignant_recall")
            ],
            output_path=f"s3://{state['bucket']}/{state['prefix']}/output/",
            sagemaker_session=sm_session,
            base_job_name=f"bc-rf-{state['tag']}",
        )
        job_name = f"bc-rf-{state['tag']}-{trees}-d{depth}-{int(time.time())}"
        print(f"Starting real SageMaker managed training job: {job_name}")
        try:
            estimator.fit(locations, job_name=job_name, wait=True, logs=True)
        finally:
            shutil.rmtree(source_workspace)
        describe = session.client("sagemaker").describe_training_job(TrainingJobName=job_name)
        metrics = {m["MetricName"]: m["Value"] for m in describe.get("FinalMetricDataList", [])}
        record = {
            "job_name": job_name,
            "n_estimators": trees,
            "max_depth": depth,
            "status": describe["TrainingJobStatus"],
            "model_artifact": describe["ModelArtifacts"]["S3ModelArtifacts"],
            "training_seconds": describe.get("TrainingTimeInSeconds"),
            "billable_seconds": describe.get("BillableTimeInSeconds"),
            "metrics": metrics,
        }
        state["training_jobs"].append(record)
        save_state(state)
        print(json.dumps(record, indent=2))
    scored = sorted(
        state["training_jobs"],
        key=lambda j: (j["metrics"].get("test:accuracy", -1), j["metrics"].get("test:auc", -1)),
        reverse=True,
    )
    state["selected_training_job"] = scored[0]["job_name"]
    save_state(state)
    print("Selected best SageMaker job:", state["selected_training_job"])


def deploy(args):
    state = require_state()
    if state.get("endpoint_name"):
        raise RuntimeError("An endpoint is already tracked. Avoid double-charging for instances.")
    if not state.get("selected_training_job"):
        raise RuntimeError("Run train first to produce a managed model artifact.")
    session, _ = aws_modules(state["region"])
    import sagemaker
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.serializers import CSVSerializer
    from sagemaker.deserializers import JSONDeserializer

    sm_session = sagemaker.Session(boto_session=session, default_bucket=state["bucket"])
    estimator = SKLearn.attach(state["selected_training_job"], sagemaker_session=sm_session)
    endpoint_name = f"breast-cancer-rf-{state['tag']}"
    print(f"DEPLOYMENT START: this creates a billable real-time instance ({args.endpoint_instance}).")
    # Preserve the resource name before creation in case the SDK times out
    # after AWS started provisioning the chargeable endpoint.
    state["endpoint_name"] = endpoint_name
    state["endpoint_creation_started_utc"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    estimator.deploy(
        initial_instance_count=1,
        instance_type=args.endpoint_instance,
        endpoint_name=endpoint_name,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )
    description = session.client("sagemaker").describe_endpoint(EndpointName=endpoint_name)
    state["endpoint_config"] = description["EndpointConfigName"]
    state["endpoint_status"] = description["EndpointStatus"]
    state["endpoint_instance"] = args.endpoint_instance
    save_state(state)
    print("Endpoint status:", state["endpoint_status"])
    sample = json.loads((ROOT / "artifacts" / "data" / "sample_request.json").read_text())["features"]
    pred = invoke_endpoint(session, state, sample)
    print("Actual SageMaker real-time inference result:", pred)
    (ROOT / "aws_inference_result.json").write_text(json.dumps(pred, indent=2), encoding="utf-8")
    print("TAKE SCREENSHOTS NOW: Endpoints/InService, model prediction, CloudWatch.")


def invoke_endpoint(session, state, features):
    response = session.client("sagemaker-runtime").invoke_endpoint(
        EndpointName=state["endpoint_name"],
        ContentType="text/csv",
        Accept="application/json",
        Body=",".join(format(float(value), ".16g") for value in features),
    )
    labels = json.loads(response["Body"].read())
    if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in (0, 1):
        raise ValueError("Unexpected inference response: " + repr(labels))
    return {"class_id": int(labels[0]), "prediction": "benign" if int(labels[0]) == 1 else "malignant"}


def publish_api(args):
    state = require_state()
    if not state.get("endpoint_name"):
        raise RuntimeError("Deploy a SageMaker endpoint first.")
    if state.get("rest_api_id"):
        raise RuntimeError("A REST API is already tracked; refusing to create another.")
    session, _ = aws_modules(state["region"])
    iam = session.client("iam")
    account = state["account_id"]
    endpoint_arn = f"arn:aws:sagemaker:{state['region']}:{account}:endpoint/{state['endpoint_name']}"
    role_name = f"SageMakerLambdaInvoke-{state['tag']}"
    role_arn = make_role(
        iam, role_name, "lambda.amazonaws.com", "InvokeOnlyThisSageMakerEndpoint",
        {
            "Version": ASSUME_ROLE_VERSION,
            "Statement": [{
                "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint", "Resource": endpoint_arn,
            }],
        },
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    state["lambda_role_name"] = role_name
    state["lambda_role_arn"] = role_arn
    save_state(state)
    # IAM propagation is eventually consistent.
    time.sleep(12)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "lambda_function.py", arcname="lambda_function.py")
    lam = session.client("lambda")
    function_name = f"breast-cancer-predict-{state['tag']}"
    fn = lam.create_function(
        FunctionName=function_name,
        Runtime="python3.12",
        Handler="lambda_function.lambda_handler",
        Role=role_arn,
        Code={"ZipFile": data.getvalue()},
        Description="Coursework inference proxy to SageMaker; IAM protected at API Gateway",
        Timeout=15,
        MemorySize=128,
        Environment={"Variables": {"ENDPOINT_NAME": state["endpoint_name"]}},
    )
    state["lambda_function"] = function_name
    save_state(state)
    lam.get_waiter("function_active_v2").wait(FunctionName=function_name)

    gateway = session.client("apigateway")
    api = gateway.create_rest_api(
        name=f"breast-cancer-ml-{state['tag']}",
        description="IAM-authenticated student coursework inference API",
        endpointConfiguration={"types": ["REGIONAL"]},
    )
    api_id = api["id"]
    state["rest_api_id"] = api_id
    save_state(state)
    root_id = next(r["id"] for r in gateway.get_resources(restApiId=api_id)["items"] if r["path"] == "/")
    child_id = gateway.create_resource(restApiId=api_id, parentId=root_id, pathPart="predict")["id"]
    gateway.put_method(
        restApiId=api_id,
        resourceId=child_id,
        httpMethod="POST",
        authorizationType="AWS_IAM",
        apiKeyRequired=False,
    )
    uri = (
        f"arn:aws:apigateway:{state['region']}:lambda:path/2015-03-31/functions/"
        f"{fn['FunctionArn']}/invocations"
    )
    gateway.put_integration(
        restApiId=api_id,
        resourceId=child_id,
        httpMethod="POST",
        type="AWS_PROXY",
        integrationHttpMethod="POST",
        uri=uri,
    )
    lam.add_permission(
        FunctionName=function_name,
        StatementId=f"allow-api-gateway-{api_id}",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{state['region']}:{account}:{api_id}/*/POST/predict",
    )
    gateway.create_deployment(restApiId=api_id, stageName="prod", description="ML coursework test")
    gateway.update_stage(
        restApiId=api_id,
        stageName="prod",
        patchOperations=[
            {"op": "replace", "path": "/*/*/throttling/rateLimit", "value": "2"},
            {"op": "replace", "path": "/*/*/throttling/burstLimit", "value": "5"},
        ],
    )
    state["api_url"] = f"https://{api_id}.execute-api.{state['region']}.amazonaws.com/prod/predict"
    save_state(state)
    print("API is AWS_IAM authenticated (SigV4). URL:", state["api_url"])
    print("Use `cloud_pipeline.py test-api` to send an authenticated POST.")


def test_api(args):
    state = require_state()
    if not state.get("api_url"):
        raise RuntimeError("Run publish-api first.")
    session, _ = aws_modules(state["region"])
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    sample = json.loads((ROOT / "artifacts" / "data" / "sample_request.json").read_text())
    payload = json.dumps(sample).encode("utf-8")
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials are not configured")
    signed = AWSRequest(
        method="POST", url=state["api_url"], data=payload,
        headers={"Content-Type": "application/json", "Host": state["api_url"].split("/")[2]},
    )
    SigV4Auth(credentials.get_frozen_credentials(), "execute-api", state["region"]).add_auth(signed)
    req = request.Request(
        state["api_url"], data=payload, method="POST", headers=dict(signed.headers),
    )
    with request.urlopen(req, timeout=30) as response:
        output = {"http_status": response.status, "body": json.loads(response.read())}
    (ROOT / "aws_api_test_result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Actual API Gateway response:", json.dumps(output, indent=2))


def monitor(args):
    state = require_state()
    if not state.get("endpoint_name"):
        raise RuntimeError("Deploy an endpoint first.")
    session, _ = aws_modules(state["region"])
    alarm_name = f"bc-rf-endpoint-5xx-{state['tag']}"
    session.client("cloudwatch").put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription="SageMaker endpoint 5xx errors; investigate endpoint and Lambda logs.",
        Namespace="AWS/SageMaker",
        MetricName="Invocation5XXErrors",
        Dimensions=[
            {"Name": "EndpointName", "Value": state["endpoint_name"]},
            {"Name": "VariantName", "Value": "AllTraffic"},
        ],
        Statistic="Sum",
        Period=60,
        EvaluationPeriods=1,
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        ActionsEnabled=False,  # No notification without a verified SNS recipient.
    )
    state["alarm_name"] = alarm_name
    save_state(state)
    print("CloudWatch alarm created (no notification action):", alarm_name)
    print("Inspect endpoint metrics: Invocations, ModelLatency, OverheadLatency, Invocation4XXErrors, Invocation5XXErrors.")


def metrics(args):
    """Read actual CloudWatch endpoint datapoints; no fabricated timings."""
    state = require_state()
    if not state.get("endpoint_name"):
        raise RuntimeError("Deploy a SageMaker endpoint first.")
    session, _ = aws_modules(state["region"])
    cloudwatch = session.client("cloudwatch")
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=args.minutes)
    output = {
        "endpoint": state["endpoint_name"],
        "range_start_utc": start.isoformat(),
        "range_end_utc": end.isoformat(),
        "metrics": {},
    }
    for name in ("Invocations", "ModelLatency", "OverheadLatency", "Invocation4XXErrors", "Invocation5XXErrors"):
        result = cloudwatch.get_metric_statistics(
            Namespace="AWS/SageMaker",
            MetricName=name,
            Dimensions=[
                {"Name": "EndpointName", "Value": state["endpoint_name"]},
                {"Name": "VariantName", "Value": "AllTraffic"},
            ],
            StartTime=start,
            EndTime=end,
            Period=60,
            Statistics=["Average", "Sum", "Maximum"],
        )
        output["metrics"][name] = sorted(
            result["Datapoints"], key=lambda point: point["Timestamp"]
        )
    (ROOT / "aws_cloudwatch_metrics.json").write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, default=str))
    print("Note: CloudWatch can take several minutes to publish fresh datapoints.")


def status(args):
    state = require_state()
    session, _ = aws_modules(state["region"])
    print("Saved resource state:", json.dumps(state, indent=2))
    if state.get("endpoint_name"):
        current = session.client("sagemaker").describe_endpoint(EndpointName=state["endpoint_name"])
        print("Live endpoint:", current["EndpointStatus"])
    print("For console evidence, capture S3 bucket, training jobs, InService endpoint, API test, CloudWatch, and cleanup.")


def cleanup(args):
    if not args.confirm:
        raise RuntimeError("Explicit --confirm required. Capture console screenshots before cleanup.")
    state = require_state()
    session, ClientError = aws_modules(state["region"])
    print("Cleaning up only resource names tracked in cloud_state.json.")
    if state.get("rest_api_id"):
        session.client("apigateway").delete_rest_api(restApiId=state["rest_api_id"])
    if state.get("lambda_function"):
        session.client("lambda").delete_function(FunctionName=state["lambda_function"])
    if state.get("alarm_name"):
        session.client("cloudwatch").delete_alarms(AlarmNames=[state["alarm_name"]])
    sagemaker = session.client("sagemaker")
    if state.get("endpoint_name"):
        endpoint_name = state["endpoint_name"]
        try:
            description = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") != "ValidationException":
                raise
            print("Endpoint not present; deploy may have failed before creation.")
        else:
            config_name = description["EndpointConfigName"]
            config = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
            models = [v["ModelName"] for v in config["ProductionVariants"]]
            sagemaker.delete_endpoint(EndpointName=endpoint_name)
            sagemaker.get_waiter("endpoint_deleted").wait(
                EndpointName=endpoint_name, WaiterConfig={"Delay": 15, "MaxAttempts": 80}
            )
            sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            for model in models:
                sagemaker.delete_model(ModelName=model)
            state["deleted_endpoint_config"] = config_name
            state["deleted_models"] = models
    # The S3 bucket and model artifacts are intentionally retained for grading.
    # IAM roles and logs also remain so a student can verify provenance later.
    state["cleaned_up"] = True
    state["cleanup_utc"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print("Removed tracked API, Lambda, alarm, endpoint, endpoint config, and model.")
    print("S3, IAM, logs, and old training artifacts remain; remove them manually if no longer needed.")
    print("Cleanup evidence saved:", STATE_FILE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("provision", help="Create encrypted private bucket and SageMaker IAM role")
    p.add_argument("--bucket", required=True, help="Globally unique new bucket, e.g. ty-aids-sagemaker-roll123")
    p.add_argument("--tag", required=True, help="Unique short resource suffix, e.g. roll123")
    p.add_argument("--region", default="ap-south-1")
    p.set_defaults(func=provision)

    p = commands.add_parser("train", help="Upload data and execute two managed training jobs")
    p.add_argument("--train-instance", default="ml.m5.large")
    p.add_argument("--framework-version", default="1.9-0", help="Current AWS image; PDF's 1.2-1 is out of patch")
    p.set_defaults(func=train)

    p = commands.add_parser("deploy", help="Create a BILLABLE always-on SageMaker endpoint")
    p.add_argument("--endpoint-instance", default="ml.m5.large")
    p.set_defaults(func=deploy)
    commands.add_parser("publish-api", help="Create Lambda + IAM-protected REST API").set_defaults(func=publish_api)
    commands.add_parser("test-api", help="Send SigV4-authenticated POST and save actual response").set_defaults(func=test_api)
    commands.add_parser("monitor", help="Create a 5XX CloudWatch alarm").set_defaults(func=monitor)
    p = commands.add_parser("metrics", help="Read real CloudWatch latency/invocation datapoints")
    p.add_argument("--minutes", type=int, default=15)
    p.set_defaults(func=metrics)
    commands.add_parser("status", help="Describe resources and endpoint status").set_defaults(func=status)
    p = commands.add_parser("cleanup", help="Delete billable endpoint and associated API resources")
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

