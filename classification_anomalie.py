from model import remap_in_numeric_labels
from setData import sample_df
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def clustering_pipeline(df):

    # Nettoyage robuste
    df["normalite"] = df["normalite"].astype(str).str.strip().str.lower()

    # Gestion numeric / string
    if df["normalite"].isin(["0", "1"]).all():
        df_abnormal = df[df["normalite"] == "1"].copy()
    else:
        df_abnormal = df[df["normalite"] == "abnormal"].copy()

    if df_abnormal.empty:
        raise ValueError("df_abnormal est vide → problème de labels")

    features = ["centroid", "mfcc_mean", "chroma_mean", "tempo", "zcr"]

    X_scaled = StandardScaler().fit_transform(df_abnormal[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_abnormal["cluster"] = kmeans.fit_predict(X_scaled)

    return df_abnormal, kmeans


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def visualize_clusters(df, features, cluster_col="cluster", output_prefix="clusters"):
    """
    Visualise les clusters d'anomalies et sauvegarde les figures :
    - PCA 2D
    - Heatmap des moyennes
    - Boxplots des features
    """

    df = df.copy()

    # =========================
    # 1. PCA 2D
    # =========================
    X_scaled = StandardScaler().fit_transform(df[features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="pca1", y="pca2", hue=cluster_col, palette="viridis")
    plt.title("Clusters visualisés en 2D (PCA)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pca.png", dpi=300)
    plt.close()

    # =========================
    # 2. Heatmap des clusters
    # =========================
    cluster_means = df.groupby(cluster_col)[features].mean()

    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_means, annot=True, cmap="coolwarm")
    plt.title("Profil moyen des clusters")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=300)
    plt.close()

    # =========================
    # 3. Boxplots des features
    # =========================
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=cluster_col, y=feature)
        plt.title(f"{feature} par cluster")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{feature}.png", dpi=300)
        plt.close()


def interpret_cluster(row):
    if row["centroid"] > 1000 and row["zcr"] > 0.03:
        return "defaut_roulement"

    elif row["zcr"] < 0.02:
        return "defaut_moteur"

    else:
        return "defaut_ventilation"


import matplotlib.pyplot as plt
import seaborn as sns


def plot_defect_analysis(df, features, target="type_defaut", output_prefix="defects"):
    """
    Analyse et visualise les types de défauts :
    - Répartition (countplot)
    - Boxplots par feature
    - Moyennes des features par défaut
    """

    df = df.copy()

    # =========================
    # 1. Répartition des défauts
    # =========================
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target, order=df[target].value_counts().index)
    plt.title("Répartition des types de défauts")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_countplot.png")
    plt.close()

    # =========================
    # 2. Boxplots des features
    # =========================
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target, y=feature)
        plt.title(f"{feature} par type de défaut")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_box_{feature}.png")
        plt.close()

    # =========================
    # 3. Moyennes des features
    # =========================
    plt.figure(figsize=(10, 6))
    df.groupby(target)[features].mean().plot(kind="bar")
    plt.title("Moyenne des features par type de défaut")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_mean_features.png")
    plt.close()


if __name__ == "__main__":
    df = sample_df(filepath="features.csv")
    print("Start classification !")
    df_abnormal, kmeans = clustering_pipeline(df=df)
    features = ["centroid", "mfcc_mean", "chroma_mean", "tempo", "zcr"]
    visualize_clusters(df_abnormal, features)
    df_abnormal["type_defaut"] = df_abnormal.apply(interpret_cluster, axis=1)
    df_moteur = df_abnormal[df_abnormal["type_defaut"] == "defaut_moteur"].copy()
    print(df_moteur)
    plot_defect_analysis(df_abnormal, features)
    print("Classification is finished !")
