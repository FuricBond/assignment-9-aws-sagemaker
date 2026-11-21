"""AWS Lambda proxy integration: POST /predict -> SageMaker InvokeEndpoint."""

import base64
import json
import math
import os


def make_response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
        "body": json.dumps(payload),
        "isBase64Encoded": False,
    }


def parse_features(event):
    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body, validate=True).decode("utf-8")
    data = json.loads(body) if isinstance(body, str) else body
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise ValueError("Body must be a JSON object with a features array")
    features = data["features"]
    if len(features) != 30:
        raise ValueError("Expected exactly 30 numeric features")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in features
    ):
        raise ValueError("Every feature must be a finite number")
    return features


def lambda_handler(event, context):
    if event.get("httpMethod", "POST") != "POST":
        return make_response(405, {"error": "POST required"})
    try:
        features = parse_features(event)
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        return make_response(400, {"error": str(error)})
    endpoint_name = os.environ.get("ENDPOINT_NAME")
    if not endpoint_name:
        return make_response(503, {"error": "Endpoint not configured"})
    try:
        import boto3

        runtime = boto3.client("sagemaker-runtime")
        result = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Accept="application/json",
            Body=",".join(format(value, ".16g") for value in features),
        )
        labels = json.loads(result["Body"].read().decode("utf-8"))
        if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in (0, 1):
            raise ValueError("Unexpected SageMaker output")
        label = int(labels[0])
        return make_response(
            200,
            {"prediction": "benign" if label == 1 else "malignant", "class_id": label},
        )
    except Exception:
        # Avoid revealing internal AWS exception details to public API callers.
        return make_response(502, {"error": "Prediction service unavailable"})

