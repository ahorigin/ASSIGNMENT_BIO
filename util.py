import pandas as pd

df = pd.read_csv("data1 .txt").iloc[:, 0].str.split(" ", expand=True)


def split_into_train_test(df, ratio):

    # Shuffle your dataset
    shuffle_df = df.sample(frac=1)

    # Define a size for your train set
    train_size = int(ratio* len(df))

    # Split your dataset
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    return (train_set,test_set)
