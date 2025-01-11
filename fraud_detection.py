# -*- coding: utf-8 -*-
"""Fraud Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1haq3ifCWlXdtutwitJx_QGYh_shf_f0c

# 1. Chargement et Aperçu des Données
"""

# Importer les bibliothèques nécessaires:
import pandas as pd  # Pour manipuler les données
import numpy as np  # Pour les calculs numériques
import matplotlib.pyplot as plt  # Pour les visualisations
import seaborn as sns  # Pour des graphiques avancés

# Charger le jeu de données
data = pd.read_csv("/content/creditcard.csv")  # Charger les données dans un DataFrame Pandas
print("Aperçu des premières lignes du dataset :\n", data.head())  # Aperçu des premières lignes

"""1. Taille et structure du dataset"""

print("\nTaille du dataset (lignes, colonnes) :", data.shape)  # Nombre de lignes et colonnes
print(data.info())  # Types de données et valeurs manquantes

"""2. Statistiques descriptives des données numériques"""

print("\nStatistiques descriptives :")
print(data.describe())  # Moyenne, écart-type, minimum, maximum, etc.

"""3. Vérification des valeurs manquantes"""

missing_values = data.isnull().sum()  # Compter les valeurs manquantes par colonne
print("\nValeurs manquantes par colonne :\n", missing_values)

"""4. Visualisation des distributions des montants"""

# Statistiques descriptives pour la colonne 'Amount'
print(data['Amount'].describe())

# Catégorisation des montants en Faible, Moyen, Élevé et Zéro
def categorize_amount(amount):
    if amount == 0:
        return 'Zéro'
    elif amount <= 10:
        return 'Faible'
    elif 10 < amount <= 100:
        return 'Moyenne'
    else:
        return 'Élevée'

# Appliquer la fonction de catégorisation sur la colonne 'Amount'
data['Amount_Category'] = data['Amount'].apply(categorize_amount)

# Visualisation de la distribution des catégories de montants
plt.figure(figsize=(10, 6))
sns.countplot(x='Amount_Category', data=data, palette=['blue', 'green', 'orange', 'red'])
plt.title("Répartition des montants des transactions")
plt.xlabel("Catégorie de Montant")
plt.ylabel("Nombre de Transactions")
plt.show()

"""5. Matrice de corrélation"""

# Sélectionner uniquement les colonnes numériques
numeric_data = data.select_dtypes(include=[np.number])

# Calculer la matrice de corrélation
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_data.corr()  # Calculer la matrice de corrélation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matrice de corrélation")
plt.show()

"""6. Analyse des outliers
Boîte à moustaches pour détecter les valeurs extrêmes
"""

# Visualisation de la boîte à moustaches (boxplot) pour le montant des transactions
plt.figure(figsize=(10, 6))
sns.boxplot(y=data['Amount'], color='orange')
plt.title("Analyse des outliers pour le montant des transactions")
plt.ylabel("Montant")
plt.show()

"""7. Vérification des transactions par catégorie"""

# Vérification des transactions par catégorie de montant
plt.figure(figsize=(8, 6))
sns.countplot(y='Amount_Category', data=data, order=data['Amount_Category'].value_counts().index)
plt.title("Nombre de transactions par catégorie de montant")
plt.xlabel("Nombre")
plt.ylabel("Catégorie de Montant")
plt.show()

""" 8. Distribution des transactions frauduleuses vs non-frauduleuses"""

# Vérifier la distribution des classes dans la colonne 'Class'
print("Distribution des classes :\n", data['Class'].value_counts())

sns.countplot(x='Class', data=data, palette=['blue', 'red'])
plt.title("Distribution des transactions : Fraudes vs Non-Fraudes")
plt.xlabel("Classe (0 = Non-Fraude, 1 = Fraude)")
plt.ylabel("Nombre de transactions")
plt.xticks(ticks=[0, 1], labels=["Non-Fraude", "Fraude"])
plt.show()

# Pourcentage des classes
class_distribution = data['Class'].value_counts(normalize=True) * 100
print("\nPourcentage des classes :\n", class_distribution)

"""# Nettoyage des données"""

# Importation des bibliothèques nécessaires
from sklearn.impute import SimpleImputer

# Traitement des valeurs manquantes : Remplir les NaN avec la médiane
imputer = SimpleImputer(strategy='median')  # Utilisation de la médiane pour remplacer les valeurs manquantes
data[['V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']] = imputer.fit_transform(data[['V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']])

# Vérification après imputation
missing_values = data.isnull().sum()  # Compter les valeurs manquantes par colonne
print("\nValeurs manquantes par colonne :\n", missing_values)

# Conversion des colonnes mal typées
data['V22'] = data['V22'].astype(float)  # Conversion de 'V22' en float
data['Time'] = data['Time'].astype(int)

# Vérification des types après transformation
print(data.dtypes)

"""# Transformation des données"""

# Importation des bibliothèques nécessaires
from sklearn.preprocessing import StandardScaler

# Standardisation des variables numériques (sauf 'Class')
scaler = StandardScaler()  # Initialisation du standardiseur
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.difference(['Class'])
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])  # Standardisation des variables numériques

# Vérification après standardisation
print(data.head())

"""# Gestion de l'équilibre des classes"""

# Importation de SMOTE et de la fonction de séparation des données
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Séparation des caractéristiques (X) et de la cible (y)
X = data.drop(columns=['Class'])  # Les données sans la colonne 'Class'
y = data['Class']  # La colonne cible 'Class'

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Application de SMOTE pour équilibrer les classes dans l'ensemble d'entraînement
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Vérification des proportions des classes après SMOTE
print("Proportions des classes après SMOTE :")
print(y_train_balanced.value_counts(normalize=True))  # Affichage des proportions des classes dans l'ensemble d'entraînement

# Sauvegarde des données prétraitées
data.to_csv('dataset_prepared.csv', index=False)
print("Les données ont été nettoyées, transformées, et sauvegardées dans 'dataset_prepared.csv'.")

# Graphique avant SMOTE
plt.figure(figsize=(12, 6))

# Distribution avant SMOTE
plt.subplot(1, 2, 1)
sns.countplot(x=y_train, palette=['blue', 'red'])
plt.title("Distribution avant SMOTE : Fraudes vs Non-Fraudes")
plt.xlabel("Classe (0 = Non-Fraude, 1 = Fraude)")
plt.ylabel("Nombre de transactions")
plt.xticks(ticks=[0, 1], labels=["Non-Fraude", "Fraude"])

# Distribution après SMOTE
plt.subplot(1, 2, 2)
sns.countplot(x=y_train_balanced, palette=['blue', 'red'])
plt.title("Distribution après SMOTE : Fraudes vs Non-Fraudes")
plt.xlabel("Classe (0 = Non-Fraude, 1 = Fraude)")
plt.ylabel("Nombre de transactions")
plt.xticks(ticks=[0, 1], labels=["Non-Fraude", "Fraude"])

plt.tight_layout()
plt.show()

"""#  Modélisation : Utilisation d'algorithmes de machine learning"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Initialisation des modèles
log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)

# 2. Entraînement des modèles
log_reg.fit(X_train_balanced, y_train_balanced)
rf_clf.fit(X_train_balanced, y_train_balanced)
xgb_clf.fit(X_train_balanced, y_train_balanced)

# 3. Prédictions sur l'ensemble de test
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)
y_pred_xgb = xgb_clf.predict(X_test)

# 4. Évaluation des modèles
# Logistic Regression
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Random Forest
print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

# XGBoost
print("XGBoost - Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# 5. Affichage des matrices de confusion
plt.figure(figsize=(18, 6))

# Matrix de confusion pour Logistic Regression
plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression - Matrice de confusion")

# Matrix de confusion pour Random Forest
plt.subplot(1, 3, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest - Matrice de confusion")

# Matrix de confusion pour XGBoost
plt.subplot(1, 3, 3)
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues')
plt.title("XGBoost - Matrice de confusion")

plt.tight_layout()
plt.show()

""" # Ajustement de XGBoost : avec RandomizedSearchCV (plus rapide pour une recherche large)"""

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Charger le jeu de données
data = pd.read_csv("dataset_prepared.csv")  # Remplacez par le chemin correct du fichier

# Vérification des premières lignes
print(data.head())

# Séparer les caractéristiques (X) et la cible (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Diviser les données en ensembles d'entraînement et de test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir le modèle RandomForest
rf = RandomForestClassifier(random_state=42)

# Paramètres réduits à tester pour accélérer la recherche
param_dist = {
    'n_estimators': [100, 200],  # Nombre d'arbres
    'max_depth': [5, 7],  # Profondeur maximale des arbres
    'min_samples_split': [5],  # Nombre minimum d'échantillons pour diviser un nœud
    'min_samples_leaf': [1],  # Nombre minimum d'échantillons dans une feuille
    'bootstrap': [True]  # Utiliser ou non l'échantillonnage bootstrap
}

# Recherche aléatoire pour tester différentes combinaisons de paramètres
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                      n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)

# Entraîner la recherche aléatoire
random_search_rf.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print("Meilleurs paramètres : ", random_search_rf.best_params_)

# Utiliser le meilleur modèle pour prédire
best_model_rf = random_search_rf.best_estimator_

# Prédictions sur l'ensemble de test
y_pred_rf = best_model_rf.predict(X_test)

# Afficher le rapport de classification
print("Classification Report pour Random Forest :")
print(classification_report(y_test, y_pred_rf))

"""# Deep Learning avec TensorFlow/Keras"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 1. Préparation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Définition du modèle
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sortie pour classification binaire
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Entraînement
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, validation_split=0.2,
                    epochs=50, batch_size=32, callbacks=[early_stopping])

# 4. Évaluation
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))