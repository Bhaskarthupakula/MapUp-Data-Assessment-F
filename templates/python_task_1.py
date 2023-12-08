import pandas as pd

def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    car_matrix.values[[range(car_matrix.shape[0])]*2] = 0
    return car_matrix


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    return df['car'].value_counts().to_dict()


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    return df[df['car'] == 'bus'][df['value'] > 2 * df['value'].mean()].index.tolist()


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    avg_truck_values = df.groupby('route')['car'].apply(lambda x: (x == 'truck').sum()).reset_index(name='truck_count')
    return avg_truck_values[avg_truck_values['truck_count'] > 7]['route'].tolist()


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    return matrix.applymap(lambda x: x * 2 if x > 5 else x)


def time_check(df, dataset_2)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.

    Args:
        df (pandas.DataFrame): Main dataset
        dataset_2 (pandas.DataFrame): Shared dataset-2

    Returns:
        pd.Series: Return a boolean series
    """
    merged_df = pd.merge(df, dataset_2, on=['id', 'id_2'])
    time_check_series = (merged_df.groupby(['id', 'id_2'])['timestamp']
                         .apply(lambda x: (x.max() - x.min()).total_seconds() >= (7 * 24 * 60 * 60)).reset_index(name='time_check'))
    return time_check_series.set_index(['id', 'id_2'])['time_check']
