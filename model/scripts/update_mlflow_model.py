import os

import mlflow
from mlflow.tracking import MlflowClient


def update_champion_model():
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    mlflow.set_tracking_uri("https://dagshub.com/matteogianferrari/homelens-ca.mlflow")
    model_registry_name = 'homelens-ca-model'
    experiment_name = 'HomeLens CA-DL Research'
    client = MlflowClient()

    # Get model info of the champion
    alias_mv = client.get_model_version_by_alias(name=model_registry_name, alias="champion")
    champion_run_id = alias_mv.run_id
    print(f"Champion run id: {champion_run_id}")

    # Get the experiment details
    experiment = client.get_experiment_by_name(experiment_name)

    # Find the best run based on a specific metric
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.test_r2_score > 0",
        order_by=["metrics.test_r2_score DESC"],
        max_results=1,
    )

    # Ensure at least one run exists
    if not runs:
        print("No runs found with the specified metric.")
        return

    current_best_run_id = runs[0].info.run_id
    print(f"Current best run id: {current_best_run_id}")

    # Check if the current best is the champion
    if current_best_run_id == champion_run_id:
        print("The new model doesn't perform better than the champion.")
    else:
        print("Updating the champion model...")

        print("1.Removed champion alias from old best model.")
        # Remove alias @champion from old best model
        client.delete_registered_model_alias(
            name=model_registry_name,
            alias="champion"
        )

        print("2.Updating champion model with new best model.")
        # Register model into MLFlow Model Registry
        mv = mlflow.register_model(
            name=model_registry_name,
            model_uri=f"runs:/{current_best_run_id}/model"
        )

        print("3.Adding champion alias to the best model.")
        # Add alias @champion to new model
        client.set_registered_model_alias(
            name=model_registry_name,
            alias="champion",
            version=mv.version
        )

        print(f"Updated champion model to version {mv.version}.")
