if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd

@custom
def transform_custom(*args, **kwargs):
    url_train = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    url_valid = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet'
    
    train = pd.read_parquet(url_train)
    valid = pd.read_parquet(url_valid)

    print('Rows: ', train.shape[0])

    return train, valid


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
