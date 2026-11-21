"""Exercise the real Lambda validation and actual trained model without AWS.

The mock transport only replaces the AWS network service; model prediction,
CSV parsing and Lambda response handling all run locally. Not cloud proof.
"""

import io
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

from lambda_function import lambda_handler
from train import input_fn, model_fn, output_fn, predict_fn


ROOT = Path(__file__).resolve().parent


class LocalRuntime:
    def __init__(self, model_dir):
        self.model = model_fn(model_dir)

    def invoke_endpoint(self, **kwargs):
        features = input_fn(kwargs["Body"], kwargs["ContentType"])
        predicted = predict_fn(features, self.model)
        serialized, _ = output_fn(predicted, kwargs["Accept"])
        return {"Body": io.BytesIO(serialized.encode("utf-8"))}


def main():
    root = ROOT / "artifacts"
    config = json.loads((root / "best_configuration.json").read_text(encoding="utf-8"))
    model_dir = root / config["selected"] / "model"
    runtime = LocalRuntime(model_dir)
    fake_boto3 = types.SimpleNamespace(client=lambda service: runtime)
    event = {
        "httpMethod": "POST",
        "body": (root / "data" / "sample_request.json").read_text(encoding="utf-8"),
    }
    with patch.dict(os.environ, {"ENDPOINT_NAME": "LOCAL_SIMULATION_NOT_AWS"}), patch.dict(
        sys.modules, {"boto3": fake_boto3}
    ):
        result = lambda_handler(event, None)
    summary = {
        "execution": "LOCAL_SIMULATION_NOT_AWS",
        "model_configuration": config["selected"],
        "response": result,
    }
    (root / "local_api_response.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    assert result["statusCode"] == 200


if __name__ == "__main__":
    main()

