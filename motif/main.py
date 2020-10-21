# motif main

from data import genre_dataframe
from features import FeatureProcessor

if __name__ == '__main__':
    df = genre_dataframe()[:5]
    fp = FeatureProcessor()

    print(fp.process_df(df))
