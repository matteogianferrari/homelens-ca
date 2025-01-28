"""HomeLens CA Model: Rest API for MLFlow Model microservice for the product HomeLens CA.

This module provides a FastAPI-based REST API for serving predictions
from a deep learning model managed by MLFlow. The API provides an endpoint
to receive input features, process them using a predefined pipeline, and return
the predicted median house value.

The application leverages MLFlow for model and pipeline tracking and deployment,
ensuring seamless integration and scalability.
"""

from contextlib import asynccontextmanager
from typing import Dict
import os

from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import numpy as np
import uvicorn
import mlflow


class InferenceRequest(BaseModel):
    """Request model for the REST API.

    Attributes:
        features: A dictionary representing a single row of input data for the model,
            where keys are feature names and values are their corresponding values.
    """
    features: Dict


class InferenceResponse(BaseModel):
    """Response model for the REST API.

    Attributes:
        prediction: The predicted median house value based on the input features.
    """
    prediction: float


# Model and pipeline objects
MODEL = None
PIPELINE = None


# pylint: disable=W0613
@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    """Manage the lifespan events of the FastAPI application.

    This context manager handles the startup and shutdown events for the FastAPI
    application. It loads the MLFlow model and pipeline during startup and ensures
    they are available for handling prediction requests.

    Args:
        fast_api_app: The FastAPI application instance.
    """
    # Sets the DagsHub MLFlow URI
    mlflow.set_tracking_uri("https://dagshub.com/matteogianferrari/homelens-ca.mlflow")

    # Authenticate to MLFLow
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # Model and pipeline path in MLFlow Model Registry
    model_path = "models:/homelens-ca-model@champion"
    pipeline_path = "models:/homelens-ca-pipeline@champion"

    # Loads the model
    # pylint: disable=W0603
    global MODEL
    try:
        MODEL = mlflow.keras.load_model(model_path)
        print("MLFlow model loaded successfully.")
    # pylint: disable=W0718
    except Exception as e:
        MODEL = None
        print(f"Error loading the MLFlow model: {e}")

    # Loads the pipeline
    # pylint: disable=W0603
    global PIPELINE
    try:
        PIPELINE = mlflow.sklearn.load_model(pipeline_path)
        print("MLFlow pipeline loaded successfully.")
    # pylint: disable=W0718
    except Exception as e:
        PIPELINE = None
        print(f"Error loading the MLFlow pipeline: {e}")

    yield

    print("Shutting down the FastAPI application.")


# FastAPI  application
app = FastAPI(
    lifespan=lifespan,
    title="HomeLens CA Model Microservice",
    description="A REST API for predicting median house values in California using MLFlow models.",
    version="1.1.0",
)


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest) -> InferenceResponse:
    """Handle prediction requests.

    This endpoint receives a JSON payload containing input features, processes the data
    using the MLFlow pipeline, performs inference using the loaded model, and returns
    the predicted median house value.

    Args:
        request: The request payload containing input features.

    Returns:
        The response containing the predicted median house value.

    Raises:
        HTTPException: If the prediction process fails due to missing model or pipeline.
    """
    # Edge case: Empty request
    if not request.features:
        return InferenceResponse(prediction=-1.0)

    # Edge case: Pipeline or model not loaded
    if MODEL is None or PIPELINE is None:
        return InferenceResponse(prediction=-1.0)

    try:
        # Converts the input features to a DataFrame
        input_data = pd.DataFrame([request.features])

        # Applies pre-processing to the data using the MLFlow pipeline
        processed_data = PIPELINE.transform(input_data)

        # Performs model inference
        pred = MODEL.predict(processed_data.to_numpy(dtype=np.float32))

        # Transforms the prediction back to the original scale(exponential minus one)
        pred_transformed = np.expm1(pred)

        return InferenceResponse(prediction=pred_transformed[0][0])
    # pylint: disable=W0718
    except Exception as e:
        print(f"Error during prediction: {e}")
        return InferenceResponse(prediction=-1.0)


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
