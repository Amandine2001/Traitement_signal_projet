from extract import create_dataframe, save_dataframe
from chargement_audio import load_audio_files
from model import remap_in_numeric_labels, model_randomforest, model_svm
from setData import sample_df

import os


def main():
    if not os.path.exists("features.csv"):
        datapath = "audio"
        training_data = load_audio_files(datapath)
        df = create_dataframe(training_data)
        save_dataframe(df, "features.csv")

    print(f"Total audio files loaded: {len(training_data)}")

    print("Hello from traitement-signal-projet!")
    df = sample_df("features.csv")
    df = remap_in_numeric_labels(df)

    model_svm(df)
    model_randomforest(df)


if __name__ == "__main__":
    main()
