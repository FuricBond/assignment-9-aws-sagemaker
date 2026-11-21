"""Stdlib test suite; cloud credentials are not needed."""

import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lambda_function import lambda_handler, parse_features
from prepare_data import prepare_data
from train import input_fn, model_fn, output_fn, predict_fn, run_training


class DataAndModelTests(unittest.TestCase):
    def test_stratified_training_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = prepare_data(root / "data")
            self.assertEqual((meta["train_samples"], meta["test_samples"]), (455, 114))
            metrics = run_training(root / "data", root / "data", root / "model", root / "results", 100, 4)
            self.assertGreater(metrics["accuracy"], 0.90)
            model = model_fn(root / "model")
            sample = json.loads((root / "data" / "sample_request.json").read_text())["features"]
            formatted = input_fn(",".join(str(value) for value in sample), "text/csv")
            self.assertEqual(formatted.shape, (1, 30))
            prediction = predict_fn(formatted, model)
            response, content_type = output_fn(prediction, "application/json")
            self.assertEqual(content_type, "application/json")
            self.assertIn(json.loads(response)[0], [0, 1])

    def test_reject_bad_feature_count(self):
        with self.assertRaises(ValueError):
            input_fn("1,2,3", "text/csv")
        with self.assertRaises(ValueError):
            input_fn(json.dumps({"features": [1] * 29 + [float("inf")]}), "application/json")


class LambdaTests(unittest.TestCase):
    def test_validation_responses(self):
        self.assertEqual(lambda_handler({"httpMethod": "GET", "body": "{}"}, None)["statusCode"], 405)
        self.assertEqual(lambda_handler({"body": "{}"}, None)["statusCode"], 400)
        self.assertEqual(lambda_handler({"body": json.dumps({"features": [0] * 29})}, None)["statusCode"], 400)
        self.assertEqual(lambda_handler({"body": json.dumps({"features": [True] + [0] * 29})}, None)["statusCode"], 400)

    def test_mock_invoke_endpoint(self):
        def client(service_name):
            self.assertEqual(service_name, "sagemaker-runtime")
            return types.SimpleNamespace(invoke_endpoint=lambda **kwargs: {"Body": io.BytesIO(b"[1]")})

        event = {"httpMethod": "POST", "body": json.dumps({"features": [1.5] * 30})}
        fake_boto3 = types.SimpleNamespace(client=client)
        with patch.dict(os.environ, {"ENDPOINT_NAME": "test-endpoint"}), patch.dict(sys.modules, {"boto3": fake_boto3}):
            response = lambda_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"]), {"prediction": "benign", "class_id": 1})

    def test_missing_endpoint_is_not_500(self):
        with patch.dict(os.environ, {}, clear=True):
            event = {"body": json.dumps({"features": [1.5] * 30})}
            self.assertEqual(lambda_handler(event, None)["statusCode"], 503)


if __name__ == "__main__":
    unittest.main()

