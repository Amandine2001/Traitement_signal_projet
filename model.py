from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC


def model_randomforest(df):

    # Séparer les caractéristiques (X) et les étiquettes (y)
    X = df.drop(columns=["normalite"])
    y = df["normalite"]

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
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model


classes = ["Normal", "Défaut roulement", "Défaut moteur", "Défaut ventilation"]

# Model avec SVM


def model_svm(df):

    # Séparer les caractéristiques (X) et les étiquettes (y)
    X = df.drop(columns=["normalite"])
    y = df["normalite"]

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
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return model


# Test du modèle SVM


def test_model_svm(model, df):
    X = df.drop(columns=["normalite"])
    y = df["normalite"]

    y_pred = model.predict(X)

    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
