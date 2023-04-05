import pandas as pd
import numpy as np
import os

def get_rating_data(fname):
    df = pd.read_csv(fname)
    data = df.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]
    # center the X
    X -= X.mean(axis=0)
    # normalize the X
    X /= X.std(axis=0)
    return X, Y

def get_dataframe(fname):
    df = pd.read_csv(fname)
    return df

if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.abspath(__file__))
    print(cur_path)
    df = get_dataframe(os.path.join(cur_path, '..\\..', 'Dataset', 'PP_recipes.csv'))  # PP_users.csv
    print(df)

    interaction_df = get_dataframe(os.path.join(cur_path, '..\\..', 'Dataset', 'interactions_test.csv'))
    print(interaction_df)
