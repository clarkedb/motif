# motif main

from data import genre_dataframe, generate_genre_dataframe
from features import FeatureProcessor
from os import path

if __name__ == '__main__':
    if not path.exists("../data/genres.csv"):
        generate_genre_dataframe()

    df = genre_dataframe()[:5]
    fp = FeatureProcessor()

    features_df = fp.process_df(df)
    print(features_df)
    print('\n', features_df.columns)
