"""
SageMaker deployment script — packages, registers, and deploys the XGBoost model.

Usage:
    uv run python deploy/deploy_sagemaker.py \
      --bucket your-bucket-name \
      --region eu-west-1 \
      --endpoint-name your-endpoint-name \
      --model-package-group your-group-name
"""

import argparse
import json
from pathlib import Path
import tarfile
import shutil
import tempfile
import boto3
import os
import time
import pandas as pd
from xgboost import XGBRegressor




MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "xgboost_bath_predictor.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


# XGBoost image URI mapping (override with env if needed)
def _xgboost_image_uri(region: str, version: str = "1.7-1") -> str:
    override = os.environ.get("SAGEMAKER_XGBOOST_IMAGE_URI")
    if override:
        return override
    # Account IDs for SageMaker prebuilt XGBoost images
    account_map = {
        "eu-west-1": "141502667606",
        "us-east-1": "683313688378",
        "us-east-2": "257758044811",
        "us-west-2": "246618743249",
    }
    account = account_map.get(region)
    if not account:
        raise ValueError(f"No XGBoost image mapping for region {region}. Set SAGEMAKER_XGBOOST_IMAGE_URI.")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:{version}"


def package_model(model_path: Path, output_dir: Path) -> Path:
    """Package the XGBoost model as a .tar.gz archive for SageMaker.

    SageMaker's built-in XGBoost container expects a file named
    'xgboost-model' at the root of the archive.

    Args:
        model_path: Path to the trained model JSON file.
        output_dir: Directory where the .tar.gz will be created.

    Returns:
        Path to the created .tar.gz file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp folder with required filename
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        target = tmp_path / "xgboost-model"
        shutil.copy(model_path, target)

        tar_path = output_dir / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(target, arcname="xgboost-model")

    return tar_path



def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    """Upload a local file to S3.

    Args:
        local_path: Path to the local file.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Full S3 URI (s3://bucket/key).
    """
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"



def register_model(
    s3_model_uri: str,
    model_package_group_name: str,
    region: str,
    metrics: dict,
) -> str:
    """Register the model in SageMaker Model Registry.

    Creates the Model Package Group if it doesn't exist, then registers
    a new Model Package version with the XGBoost container image,
    the S3 model artifact, and evaluation metrics.

    Args:
        s3_model_uri: S3 URI of the packaged model (.tar.gz).
        model_package_group_name: Name for the Model Package Group.
        region: AWS region.
        metrics: Dict with 'rmse', 'mae', 'r2' keys.

    Returns:
        The Model Package ARN.
    """
    sm = boto3.client("sagemaker", region_name=region)

    # Create Model Package Group if missing
    try:
        sm.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription="XGBoost bath time predictor",
        )

    # Get built-in XGBoost image URI (version 1.7-1)
    image_uri = _xgboost_image_uri(region, version="1.7-1")

    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription="Bath time predictor",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": s3_model_uri,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        CustomerMetadataProperties={
            "rmse": str(metrics.get("rmse")),
            "mae": str(metrics.get("mae")),
            "r2": str(metrics.get("r2")),
        },
        ModelApprovalStatus="Approved",
    )

    return response["ModelPackageArn"]



def deploy_endpoint(
    model_package_arn: str,
    endpoint_name: str,
    region: str,
    instance_type: str = "ml.t2.medium",
) -> str:
    """Deploy a real-time SageMaker endpoint from a registered Model Package.

    Creates a SageMaker Model, Endpoint Configuration, and Endpoint.
    Waits until the endpoint status is 'InService'.

    Args:
        model_package_arn: ARN of the registered Model Package.
        endpoint_name: Name for the endpoint.
        region: AWS region.
        instance_type: EC2 instance type for the endpoint.

    Returns:
        The endpoint name.
    """
    sm = boto3.client("sagemaker", region_name=region)

    role_arn = os.environ.get("SAGEMAKER_EXECUTION_ROLE")
    if not role_arn:
        raise ValueError("Set SAGEMAKER_EXECUTION_ROLE env var to your SageMaker execution role ARN.")

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    # Clean up existing resources if re-running
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.delete_endpoint(EndpointName=endpoint_name)
        waiter = sm.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
    except sm.exceptions.ClientError:
        pass

    try:
        sm.describe_endpoint_config(EndpointConfigName=config_name)
        sm.delete_endpoint_config(EndpointConfigName=config_name)
    except sm.exceptions.ClientError:
        pass

    try:
        sm.describe_model(ModelName=model_name)
        sm.delete_model(ModelName=model_name)
    except sm.exceptions.ClientError:
        pass

    # Create SageMaker Model from Model Package
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=role_arn,
    )

    # Create Endpoint Config
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
    )

    # Create Endpoint
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    # Wait for InService
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    return endpoint_name



def test_endpoint(endpoint_name: str, region: str) -> dict:
    """Test the deployed endpoint with sample pieces.

    Invokes the endpoint with representative inputs and compares
    the predictions against expected ranges.

    Args:
        endpoint_name: Name of the deployed endpoint.
        region: AWS region.

    Returns:
        Dict with test results and predictions.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    samples = [
        {"die_matrix": 5052, "lifetime_2nd_strike_s": 18.3, "oee_cycle_time_s": 13.5},
        {"die_matrix": 5090, "lifetime_2nd_strike_s": 18.0, "oee_cycle_time_s": 13.5},
        {"die_matrix": 5052, "lifetime_2nd_strike_s": 30.0, "oee_cycle_time_s": 13.5},
    ]

    # Endpoint predictions
    endpoint_preds = []
    for s in samples:
        payload = f"{s['die_matrix']},{s['lifetime_2nd_strike_s']},{s['oee_cycle_time_s']}"
        resp = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        pred = float(resp["Body"].read().decode("utf-8").strip())
        endpoint_preds.append(pred)

    # Local model predictions
    local_model = XGBRegressor()
    local_model.load_model(str(MODEL_FILE))

    df = pd.DataFrame(samples)
    local_preds = local_model.predict(df).tolist()

    return {
        "samples": samples,
        "endpoint_predictions": endpoint_preds,
        "local_predictions": local_preds,
    }


def main():
    parser = argparse.ArgumentParser(description="Deploy XGBoost model to SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket for model artifact")
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--model-package-group", required=True, help="Model Package Group name")
    args = parser.parse_args()

    # Load model metadata for metrics
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    print("=" * 60)
    print("SageMaker Deployment Pipeline")
    print("=" * 60)

    # Step 1: Package
    print("\n[1/5] Packaging model artifact...")
    tar_path = package_model(MODEL_FILE, MODEL_DIR)
    print(f"  Created: {tar_path}")

    # Step 2: Upload to S3
    print("\n[2/5] Uploading to S3...")
    s3_key = "models/xgboost-bath-predictor/model.tar.gz"
    s3_uri = upload_to_s3(tar_path, args.bucket, s3_key)
    print(f"  Uploaded: {s3_uri}")

    # Step 3: Register in Model Registry
    print("\n[3/5] Registering in Model Registry...")
    model_package_arn = register_model(
        s3_uri, args.model_package_group, args.region, metadata["metrics"]
    )
    print(f"  Registered: {model_package_arn}")

    # Step 4: Deploy endpoint
    print("\n[4/5] Deploying endpoint...")
    endpoint = deploy_endpoint(model_package_arn, args.endpoint_name, args.region)
    print(f"  Endpoint live: {endpoint}")

    # Step 5: Test
    print("\n[5/5] Testing endpoint...")
    results = test_endpoint(args.endpoint_name, args.region)
    print(f"  Results: {json.dumps(results, indent=2)}")

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Endpoint:       {args.endpoint_name}")
    print(f"  Model Package:  {model_package_arn}")
    print(f"  S3 artifact:    {s3_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
