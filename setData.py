import pandas as pd


def sample_df(filepath):
    df = pd.read_csv(filepath)
    df_normal = df[df["normalite"] == "normal"]
    df_anormal = df[df["normalite"] != "normal"]

    df_normal_sampled = df_normal.sample(n=len(df_anormal), random_state=42)
    df_final = pd.concat([df_normal_sampled, df_anormal], ignore_index=True)

    return df_final
