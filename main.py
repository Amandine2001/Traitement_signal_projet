from classification_anomalie import (
    clustering_pipeline,
    interpret_cluster,
    plot_defect_analysis,
    visualize_clusters,
)
from extract import create_dataframe, save_dataframe
from chargement_audio import load_audio_files
from model import remap_in_numeric_labels, model_randomforest, model_svm
from setData import sample_df
from save_in_output_folder import save_png

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
    print(f"\n Signal caracteristics : \n {df.head()}")

    df_remap = remap_in_numeric_labels(df)
    print(f"\n Dataframe's remap : \n {df_remap.head()}")

    print(f"\n Binary classification (normal/abnormal) : \n")
    model_svm(df=df_remap, target="normalite", columns_to_drop=["name", "normalite"])
    model_randomforest(
        df=df_remap, target="normalite", columns_to_drop=["name", "normalite"]
    )

    print(f"\n Multiple classification \n")
    df_abnormal, kmeans = clustering_pipeline(df=df)
    features = ["centroid", "mfcc_mean", "chroma_mean", "tempo", "zcr"]
    visualize_clusters(df_abnormal, features)
    df_abnormal["type_defaut"] = df_abnormal.apply(interpret_cluster, axis=1)
    df_moteur = df_abnormal[df_abnormal["type_defaut"] == "defaut_moteur"].copy()
    print(df_moteur)
    plot_defect_analysis(df_abnormal, features)

    save_png(original_folder=".", output_folder="output_images")


if __name__ == "__main__":
    main()
