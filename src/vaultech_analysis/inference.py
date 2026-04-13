"""
Inference service for predicting total piece travel time.

Loads the trained XGBoost model and provides predictions.

Usage as CLI:
    uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5

Usage as module (for Streamlit):
    from vaultech_analysis.inference import Predictor
    predictor = Predictor()
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3, oee_cycle_time_s=13.5)
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import boto3
from xgboost import XGBRegressor


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "pieces.parquet"


class Predictor:
    """Loads the trained model and provides predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR, gold_file: Path = GOLD_FILE):
        model_path = model_dir / "xgboost_bath_predictor.json"
        metadata_path = model_dir / "model_metadata.json"

        self.model_dir = model_dir
        self.gold_file = gold_file

        # SageMaker endpoint config (optional)
        self.endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
        self.runtime = boto3.client("sagemaker-runtime") if self.endpoint_name else None

        # Load metadata (if available). For SageMaker-only mode, fall back to defaults.
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            self.features = self.metadata["features"]
            self.metrics = self.metadata["metrics"]
        else:
            self.metadata = {
                "features": ["die_matrix", "lifetime_2nd_strike_s", "oee_cycle_time_s"],
                "metrics": {},
            }
            self.features = self.metadata["features"]
            self.metrics = self.metadata["metrics"]

        # Load model (local fallback) if available
        self.model = None
        if model_path.exists():
            self.model = XGBRegressor()
            self.model.load_model(str(model_path))

        # Load gold for valid matrices + OEE median
        gold = pd.read_parquet(gold_file)
        self.die_matrices = sorted(gold["die_matrix"].dropna().astype(int).unique().tolist())
        self.oee_median = gold["oee_cycle_time_s"].dropna().median()

        pass

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict total bath time from early-stage features.

        Returns a dict with predicted_bath_time_s, input values, and model_metrics.
        Returns {"error": "..."} for unknown die_matrix values.
        Missing oee_cycle_time_s should default to the median (~13.8s).
        """
        if die_matrix not in self.die_matrices:
            return {"error": f"Unknown die_matrix: {die_matrix}"}

        oee_used = self.oee_median if oee_cycle_time_s is None else oee_cycle_time_s

        X = pd.DataFrame([{
            "die_matrix": die_matrix,
            "lifetime_2nd_strike_s": lifetime_2nd_strike_s,
            "oee_cycle_time_s": oee_used,
        }])[self.features]

        if self.endpoint_name:
            payload = X.to_csv(index=False, header=False)
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="text/csv",
                Body=payload,
            )
            pred = float(response["Body"].read().decode().strip())
            return {
                "predicted_bath_time_s": pred,
                "die_matrix": die_matrix,
                "lifetime_2nd_strike_s": lifetime_2nd_strike_s,
                "oee_cycle_time_s": oee_used,
                "model_metrics": self.metrics,
                "debug": {
                    "source": "sagemaker",
                    "endpoint": self.endpoint_name,
                    "prediction": pred,
                },
            }

        if self.model is None:
            return {"error": "Local model not available and SageMaker endpoint not configured."}

        pred = float(self.model.predict(X)[0])

        return {
            "predicted_bath_time_s": pred,
            "die_matrix": die_matrix,
            "lifetime_2nd_strike_s": lifetime_2nd_strike_s,
            "oee_cycle_time_s": oee_cycle_time_s,
            "model_metrics": self.metrics,
        }

        

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame of pieces.

        Handle missing oee_cycle_time_s by filling with the median.
        """
        df = df.copy()
        df["oee_cycle_time_s"] = df["oee_cycle_time_s"].fillna(self.oee_median)

        X = df[self.features]
        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index)

       


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--die-matrix", type=int, required=True)
    parser.add_argument("--strike2", type=float, required=True)
    parser.add_argument("--oee", type=float, required=False, default=None)
    args = parser.parse_args()

    predictor = Predictor()
    result = predictor.predict(
        die_matrix=args.die_matrix,
        lifetime_2nd_strike_s=args.strike2,
        oee_cycle_time_s=args.oee,
    )
    print(json.dumps(result, indent=2))



if __name__ == "__main__":
    main()
