# motif main

# from features import process_features
from models import logistic_regression

if __name__ == "__main__":
    logistic_regression(plot_matrix=True, test_size=.1, normalize=True)

    """features_df = process_features()

    print(features_df)
    print("\n", features_df.columns)"""
