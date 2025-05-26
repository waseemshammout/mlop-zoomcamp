import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# mlflow.end_run()

mlflow.set_tracking_uri('sqlite:///backend.db')

mlflow.set_experiment('exper_local_server')

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mss = rf.get_params()['min_samples_split']
        mlflow.log_param('max_depth','10')
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('min_samples_split', mss)
        mlflow.set_tag('model','RandomForest')

if __name__ == '__main__':
    run_train()
