# ai_exam_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Charger le dataset
data = pd.read_csv("train.csv")

# Sélection des colonnes pertinentes
features = ['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources']
X = data[features]
y = data['Exam_Score']  # Supposons que la colonne de score de l'examen s'appelle 'Exam_Score'

# Séparation en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Liste des modèles à évaluer
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regression": SVR(kernel='rbf')
}

# Critère d'évaluation : pourcentage d'erreur acceptable
error_threshold = 0.05  # 5%

# Entraînement et évaluation des modèles
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcul de l'erreur en pourcentage (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"{model_name} - Mean Absolute Percentage Error: {mape:.2%}")
    
    # Vérification si l'erreur est dans la marge acceptable
    if mape <= error_threshold:
        print(f"{model_name} satisfait le critère avec une erreur de {mape:.2%}\n")
    else:
        print(f"{model_name} ne satisfait pas le critère avec une erreur de {mape:.2%}\n")
