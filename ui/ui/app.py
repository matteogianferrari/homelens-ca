"""HomeLens CA UI: A Gradio-based interface microservice for the product HomeLens CA.

This module provides a web interface to allow the user to interface with the model.
Users can input various attributes related to a housing block, and the model returns
the predicted median house value.

The application leverages Gradio for the user interface and communicates with a
model-serving microservice via HTTP requests.
"""

import os

import requests
import gradio as gr


def predict(*features) -> float:
    """Predicts the median house value based on input features.

    This function sends a POST request to the HomeLens CA Model microservice with
    the provided housing features and returns the predicted median house value.

    Args:
        *features: Variable length argument list representing housing features.
            The expected features in order are:
                - ocean_proximity (str): Proximity to the ocean.
                - longitude (float): Geographical longitude.
                - latitude (float): Geographical latitude.
                - housing_median_age (int): Median age of the housing in the block.
                - total_rooms (int): Total number of rooms in the block.
                - total_bedrooms (int): Total number of bedrooms in the block.
                - population (int): Population in the block.
                - households (int): Number of households in the block.
                - median_income (float): Median income in the block (in thousands of $).

    Returns:
        The predicted median house value.

    Raises:
        requests.HTTPError: If the HTTP request to the model service fails.
    """
    # URL used to make POST request to the HomeLens CA Model microservice
    # This variable is obtained from the environment
    # Default value specified for local testing
    os.environ["MODEL_SERVING_URL"] = os.getenv("MODEL_SERVING_URL", "http://0.0.0.0:7860/predict")

    # Creates a dict from the features names and related values
    feature_names = [
        "ocean_proximity", "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population", "households", "median_income"
    ]
    features = dict(zip(feature_names, features))

    # Prepares the payload
    payload = {"features": features}

    # Makes the POST request to the Model microservice
    response = requests.post(os.getenv('MODEL_SERVING_URL'), json=payload, timeout=3)

    # Handles the timeout case
    response.raise_for_status()

    # Extracts the predicted value from the response
    prediction = response.json().get('prediction')

    return prediction


# List of inputs used by the UI
inputs = [
    gr.Radio(
        choices=["near_bay", "<1h_ocean", "inland", "near_ocean"],
        label="Ocean Proximity",
        info="Proximity of the block to the ocean."
    ),
    gr.Slider(
        minimum=-124.3,
        maximum=-114.31,
        label="Longitude",
        info="Geographical longitude of the block location."
    ),
    gr.Slider(
        minimum=32.54,
        maximum=41.95,
        label="Latitude",
        info="Geographical latitude of the block location."
    ),
    gr.Number(
        minimum=1,
        maximum=51,
        precision=0,
        label="Housing Median Age",
        info="Median age of the houses in the block."
    ),
    gr.Number(
        minimum=32,
        maximum=37937,
        precision=0,
        label="Total Rooms",
        info="Total number of rooms in the block."
    ),
    gr.Number(
        minimum=7,
        maximum=5471,
        precision=0,
        label="Total Bedrooms",
        info="Total number of bedrooms in the block."
    ),
    gr.Number(
        minimum=13,
        maximum=16122,
        precision=0,
        label="Population",
        info="Population of the block."
    ),
    gr.Number(
        minimum=6,
        maximum=5189,
        precision=0,
        label="Households",
        info="Number of households in the block."
    ),
    gr.Number(
        minimum=0.49,
        maximum=13.11,
        label="Median Income [k$]",
        info="Median income of the households in the block (in thousands of $)."
    )
]

# List of examples to use for predictions
examples = [
    ['near_bay', -122.23, 37.88, 41, 880, 129, 322, 126, 8.3252],
    ['inland', -119.02, 35.36, 48, 1833, 396, 947, 363, 2.2827],
    ['<1h_ocean', -118.59, 34.14, 19, 1303, 155, 450, 145, 10.5511],
    ['near_ocean', -118.31, 33.73, 49, 1642, 287, 692, 288, 4.1812]
]

# Creates the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    examples=examples,
    outputs=gr.Number(label="Predicted Median House Value [$]", precision=2),
    title="HomeLens CA",
    description=(
        "Predict the median house values in California (1990) given block information. "
        "Provide the necessary housing and geographical features to receive an accurate prediction."
    ),
    theme=gr.themes.Soft()
)


if __name__ == '__main__':
    interface.launch(server_name="0.0.0.0", server_port=8080)
