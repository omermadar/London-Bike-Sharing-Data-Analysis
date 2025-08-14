import pandas as pd
from datetime import datetime

from pandas import to_datetime


def load_data(path):
    """
    loads the data from the csv file (given path)
    :param path:
    :return: a pandas df from the given csv file
    """
    df = pd.read_csv(path)
    return df

def season_by_index(index):
    """
    given an index, returns the season name
    :param index: int between 0 and 3
    :return: a string season-name according to the index
    """
    if index == 0:
        return "spring"
    if index == 1:
        return "summer"
    if index == 2:
        return "fall"
    if index == 3:
        return "winter"
    return "unknown"

def weekend_holiday_check(is_weekend, is_holiday):
    """
    Receives is_holiday and is_weekend (o | 1) and returns index num by demand (1 to 4)
    :param is_weekend: int from df
    :param is_holiday: int from df
    :return: an int representing an index according to the given table
    """
    if is_holiday == 0:
        if is_weekend == 0:
            return 1
        return 2

    if is_weekend == 0:
        return 3

    return 4

# Section 2 to 5
def add_new_columns(df):
    """
    return the df after adding new columns, according to the instructions (season_name, timestamp, Hour,
     Day, Month, Year, is_weekend_holiday, t_diff)
    :param df: pandas df
    :return: returns the changed df
    """
    # section 2
    df['season_name'] = df['season'].apply(season_by_index)

    # section 3
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst = True)  # convert once

    df['Hour'] = df['timestamp'].apply(lambda ts: ts.hour)
    df['Day'] = df['timestamp'].apply(lambda ts: ts.day)
    df['Month'] = df['timestamp'].apply(lambda ts: ts.month)
    df['Year'] = df['timestamp'].apply(lambda ts: ts.year)


    # section 4
    df['is_weekend_holiday'] = df.apply(lambda x: weekend_holiday_check(x['is_weekend'],x['is_holiday']), axis = 1)

    # section 5
    df['t_diff'] = df.apply(lambda x:x['t2'] - x['t1'], axis = 1)
    return df



# Section 6 to 8
def data_analysis(df):
    """
    creates a correlation matrix and prints the highest and lowest correlated features,
    prints season t_diff averages
    :param df:
    :return: Nan
    """
    # section 6
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    corr = corr.abs()
    # section 7
    corr_dict = {}
    for i in range(0, len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            corr_dict[(corr.columns[i], corr.columns[j])] = corr.iloc[i, j]

    sorted_corr_dict = sorted(corr_dict.keys(), key = corr_dict.get)

    print()
    print("Highest correlated are: ")
    for i in range(len(sorted_corr_dict) - 1, len(sorted_corr_dict) - 6, -1):
        print(f"{len(sorted_corr_dict) - i}. {sorted_corr_dict[i]} with {corr_dict[sorted_corr_dict[i]]:.6f}")

    print()
    print("Lowest correlated are: ")
    for i in range(5):
        print(f"{i + 1}. {sorted_corr_dict[i]} with {corr_dict[sorted_corr_dict[i]]:.6f}")



    # section 8

    print()
    seasonal_average_t_diff = df.groupby('season_name')['t_diff'].mean()
    for season, avg in seasonal_average_t_diff.items():
        print(f'{season} average t_diff is {avg:.2f}')

    print(f'All average t_diff is {df["t_diff"].mean():.2f}')








