# motif main

from data import genre_dataframe
from data import generate_genre_dataframe
from features import FeatureProcessor

if __name__ == '__main__':
    df = genre_dataframe()[:5]
    fp = FeatureProcessor()

    features_df = fp.process_df(df)
    print(features_df)
    print('\n', features_df.columns)
