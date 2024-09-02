from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
import numpy as np
import shap

# Charger le dataset
california = fetch_california_housing()
# Séparer les caractéristiques et la variable cible
X, y = california.data, california.target
# Créer un DataFrame
df = pd.DataFrame(data=X, columns=california.feature_names)
df['target'] = y
####################################################
##Second STEP : prétraitement des données
print("//PRETRAITEMENT DES DONNEES//\n")
# Vérifier les valeurs manquantes
print("Verification des valeurs manquantes:\n")
print(df.isnull().sum())

# Supprimer les valeurs manquantes
print("Suppression/Remplissage des valeurs manquantes\n")
df.dropna(inplace=True)

# Ou remplir les valeurs manquantes avec la moyenne

df.fillna(df.mean(), inplace=True)

# Normalisation des données
print("Normalisation des données\n")
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df_scaled.drop('target', axis=1), df_scaled['target'], test_size=0.2, random_state=42)

# Définition des modèles
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'RandomForestRegressor': RandomForestRegressor()
}

# Définition des grilles de recherche des hyperparamètres
'''
param_grids = {
    'LinearRegression': {},
    'Lasso': {'alpha': [0.1, 0.5, 1.0, 2.0]},
    'RandomForestRegressor': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
}
'''
param_grids = {
    'LinearRegression': {},
    'Lasso': {'alpha': [0.1, 1.0]},
    'RandomForestRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'bootstrap': [True]
    }
}
# Entraînement des modèles et recherche des meilleurs hyperparamètres
print("ENTRAINEMENT DES DIFFERENTS MODELES ET RECHERCHE DES MEILLEURS HYPERPARAMETRES")
for name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Meilleurs paramètres pour {name}:", grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE pour {name}:", rmse)

# Entraînement des modèles, recherche des meilleurs hyperparamètres et calcul du R^2
for name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Meilleurs paramètres pour {name}:", grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("Calcul et affichage du coefficient de Détermination\n")
    print(f"RMSE pour {name}:", rmse)
    print(f"R^2 pour {name}:", r2)

print("Resultat : le modèle RandomForestRegressor est le meilleur\n")


# Entraîner un modèle de régression Ridge
model = Ridge()
model.fit(X_train, y_train)

# Création de l'explainer
print("Création de l'explainer\n")
explainer = shap.Explainer(model, X_train[:100])

# Calcul des valeurs SHAP
print("Calcul des valeurs SHAP\n")
shap_values = explainer.shap_values(X_test[:100])

# Affichage de l'importance des caractéristiques
print("Affichage de l'importance des caractéristiques\n")
shap.summary_plot(shap_values, X_test[:100], feature_names=california.feature_names, show=False)
plt.savefig('shap_summary_plot.png')


print("OPTIMISATION DES HYPERPARAMETRES\n")
# Définition de la nouvelle grille d'hyperparamètres
'''
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
'''
# Définition de la grille d'hyperparamètres
'''
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}
'''
param_grid = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}


# Création de l'objet GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Entraînement du modèle GridSearchCV
grid_search.fit(X_train, y_train)
# Affichage des meilleurs paramètres GridSearchCV
print("Meilleurs paramètres:", grid_search.best_params_)


# Prédiction sur l'ensemble de test avec le meilleur modèle
y_pred = grid_search.predict(X_test)

# Calcul du RMSE et du R^2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Calcul et affichage du coefficient de Détermination\n")
print("RMSE:", rmse)
print("R^2:", r2)
####################################################

# Calcul des valeurs SHAP pour la première prédiction
'''
print("Calcul des valeurs SHAP pour la première prédiction\n")
shap_values_instance = explainer.shap_values(X_test.iloc[0, :])
print("Affichage des valeurs SHAP\n")

# Affichage des valeurs SHAP
shap.force_plot(explainer.expected_value, shap_values_instance, X_test.iloc[0, :])
'''
####################################################
##FIRST STEP : exploration statistique des données
# Résumé statistique
print("\n")
print("EXPLORATION STATIQUE DES DONNEES\n")
print(df.describe())


# Histogramme
df.hist(bins=50, figsize=(20,15))
plt.savefig('histogram.png')  # Sauvegarde de la figure

# Diagramme de dispersion
sns.scatterplot(x=df['MedInc'], y=df['target'])
plt.savefig('scatterplot.png')  # Sauvegarde de la figure

# Corrélations
print("\n")
print("Coefficients de correlation\n")
corr_matrix = df.corr()
print(corr_matrix['target'].sort_values(ascending=False))
