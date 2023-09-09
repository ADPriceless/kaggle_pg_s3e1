import pandas as pd


def read_train_data() -> pd.DataFrame:
    train_data = pd.read_csv('data\\train.csv')
#     train_data.drop('id', axis=1, inplace=True)
    train_data.set_index('id', inplace=True)
    return train_data


def split_X_y(df: pd.DataFrame, y: str):
    return df.drop(y, axis=1), df[y]
