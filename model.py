from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from setData import sample_df

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def model_randomforest(df: pd.DataFrame, target: str, columns_to_drop: list):

    # Séparer les caractéristiques (X) et les étiquettes (y)
    X = df.drop(columns=columns_to_drop)
    y = df[target]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Créer et entraîner le modèle de forêt aléatoire
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Afficher le rapport de classification et la matrice de confusion
    print("Rapport de classification pour Random Forest :")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # afficher une heatmap de la matrice de confusion et un autre graphique avec l importance des features
    matrice_confusion = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion pour Random Forest :")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrice_confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anormal"],
        yticklabels=["Normal", "Anormal"],
    )
    plt.xlabel("Étiquettes prédites")
    plt.ylabel("Étiquettes réelles")
    plt.title("Matrice de confusion avec Random Forest")
    plt.savefig("matrice_confusion_randomforest.png")
    plt.close()

    # Afficher l'importance des features
    feature_importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=features)
    plt.title("Importance des features avec Random Forest")
    plt.savefig("importance_features_randomforest.png")
    plt.close()

    return model


# Model avec SVM


def model_svm(df: pd.DataFrame, target: str, columns_to_drop: list):

    # Séparer les caractéristiques (X) et les étiquettes (y)
    X = df.drop(columns=columns_to_drop)
    y = df[target]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Créer et entraîner le modèle SVM
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Afficher le rapport de classification et la matrice de confusion
    print("Rapport de classification pour SVM :")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    matrice_confusion = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion pour SVM :")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrice_confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anormal"],
        yticklabels=["Normal", "Anormal"],
    )
    plt.xlabel("Étiquettes prédites")
    plt.ylabel("Étiquettes réelles")
    plt.title("Matrice de confusion avec SVM")
    plt.savefig("matrice_confusion_svm.png")
    plt.close()
    return model


# Test du modèle SVM
def remap_in_numeric_labels(df):
    normalite_mapping = {
        "normal": 0,
        "abnormal": 1,
    }
    mecanisme_mapping = {
        "fan": 0,
        "pump": 1,
        "slider": 2,
        "valve": 3,
    }
    df["normalite"] = df["normalite"].map(normalite_mapping)
    df["mecanisme"] = df["mecanisme"].map(mecanisme_mapping)
    df["modele"] = df["modele"].astype("category").cat.codes
    return df


def test_model_svm(model, df, columns_to_drop, target):
    X = df.drop(columns=columns_to_drop)
    y = df[target]

    y_pred = model.predict(X)

    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    df = sample_df(filepath="features.csv")
    columns_to_drop = ["name", "normalite"]
    target = "normalite"

    df = remap_in_numeric_labels(df)

    model_randomforest(df=df, target=target, columns_to_drop=columns_to_drop)
    model_svm(df=df, target=target, columns_to_drop=columns_to_drop)
