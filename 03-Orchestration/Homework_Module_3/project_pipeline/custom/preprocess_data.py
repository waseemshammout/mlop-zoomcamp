if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer

CATEGORICAL_FEATURES = ['PULocationID', 'DOLocationID']
NUMERICAL_FEATURES = ['trip_distance']
TARGET_COLUMN = 'duration'
features_for_vectorizer = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
dv = DictVectorizer()

@custom
def transform_custom(data, *args, **kwargs):
    
    trn_data = data[0]
    val_data = data[1]

    trn_data['duration'] = (trn_data.tpep_dropoff_datetime - trn_data.tpep_pickup_datetime).dt.total_seconds() / 60
    trn_data = trn_data[(trn_data.duration >= 1) & (trn_data.duration <= 60)].copy()
    trn_data[CATEGORICAL_FEATURES] = trn_data[CATEGORICAL_FEATURES].astype(str)

    val_data['duration'] = (val_data.tpep_dropoff_datetime - val_data.tpep_pickup_datetime).dt.total_seconds() / 60
    val_data = val_data[(val_data.duration >= 1) & (val_data.duration <= 60)].copy()
    val_data[CATEGORICAL_FEATURES] = val_data[CATEGORICAL_FEATURES].astype(str)

    trn_dicts = trn_data[features_for_vectorizer].to_dict(orient='records')
    X_train = dv.fit_transform(trn_dicts)

    val_dicts = val_data[features_for_vectorizer].to_dict(orient='records')
    X_valid = dv.transform(val_dicts)

    y_train = trn_data[TARGET_COLUMN]
    y_valid = val_data[TARGET_COLUMN]

    return X_train, X_valid, y_train, y_valid, dv


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
