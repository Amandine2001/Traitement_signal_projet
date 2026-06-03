from model import remap_in_numeric_labels, model_randomforest
from setData import sample_df


def main():
    print("Hello from traitement-signal-projet!")
    df = sample_df("features.csv")
    df = remap_in_numeric_labels(df)

    model_randomforest(df)


if __name__ == "__main__":
    main()
