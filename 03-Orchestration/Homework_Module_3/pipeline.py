import mlflow.sklearn
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from pathlib import Path
import mlflow

train_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
val_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet"
target_column = "duration"
categorical_features = ["PULocationID", "DOLocationID"]
numerical_features = ["trip_distance"]
experiment_name = "nyc_taxi"
tracking_uri = 'http://127.0.0.1:5000'
model_output_dir = Path("models")

def prepare_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df["duration"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical_features] = df[categorical_features].astype(str)

    return df

def preprocess_data(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple:
    features_for_vectorizer = categorical_features + numerical_features
    train_dicts = df_train[features_for_vectorizer].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    val_dicts = df_val[features_for_vectorizer].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_train = df_train[target_column].values
    y_val = df_val[target_column].values

    return X_train, X_val, y_train, y_val, dv

def train_and_evaluate(X_train, X_val, y_train, y_val, dv: DictVectorizer):
    model_output_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        i = model.intercept_

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("Intercept", i)

        preprocessor_path = model_output_dir / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(preprocessor_path, artifact_path="model_preprocessor")

        mlflow.sklearn.log_model(model, artifact_path="LinReg_model")

if __name__ == "__main__":
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)

    print("--- Preparing data ---")
    df_train = prepare_data(train_data_url)
    df_val = prepare_data(val_data_url)
    print("Data preparation complete.")

    print("--- Preprocessing data ---")
    X_train, X_val, y_train, y_val, dv = preprocess_data(df_train, df_val)
    print("Data preprocessing complete.")

    print("--- Training and evaluating model ---")
    train_and_evaluate(X_train, X_val, y_train, y_val, dv)
    print("Model training and evaluation complete. Check MLflow UI for results.")
