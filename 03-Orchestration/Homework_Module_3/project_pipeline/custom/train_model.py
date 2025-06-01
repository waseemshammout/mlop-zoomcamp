if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from datetime import datetime
import pickle
from pathlib import Path
import mlflow

model_output_dir = Path("models")
tracking_uri = 'http://127.0.0.1:5000'
experiment_name = 'nyc_taxi'

@custom
def transform_custom(data, *args, **kwargs):

    model_output_dir.mkdir(parents=True, exist_ok=True)
    x_tr, x_val, y_tr, y_val, dv = data

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        mlflow.set_tag("Pipeline", "MAGE")
        model = LinearRegression()
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        i = model.intercept_

        mlflow.log_metric("Intercept", i)
        mlflow.log_metric("RMSE", rmse)

        n = str(datetime.now())

        preprocessor_path = model_output_dir / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(preprocessor_path, artifact_path="model_preprocessor")

        mlflow.sklearn.log_model(model, artifact_path="LinReg_model")
        
        with open("C:\\Users\\wsham\\OneDrive\\Desktop\\Data Science\\Playground\\nyc_taxi\\RMSE.txt","+a") as file:
            file.write(f"RMSE: {rmse}, Time: {n} \n")

        return rmse


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
