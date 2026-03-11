import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("beer-servings.csv", index_col=0)

# fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# ENCODE COUNTRY
# -----------------------------

le = LabelEncoder()
df["country"] = le.fit_transform(df["country"])

# -----------------------------
# ONE HOT ENCODE CONTINENT
# -----------------------------

df = pd.get_dummies(df, columns=["continent"])

# -----------------------------
# FEATURES & TARGET
# -----------------------------

X = df.drop("total_litres_of_pure_alcohol", axis=1)
y = df["total_litres_of_pure_alcohol"]

feature_columns = X.columns

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL 1: LINEAR REGRESSION
# -----------------------------

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_score = r2_score(y_test, lr_pred)

print("Linear Regression R2:", lr_score)

# -----------------------------
# MODEL 2: RANDOM FOREST
# -----------------------------

rf = RandomForestRegressor()

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[None,10,20],
    "min_samples_split": [2,5]
}

grid = GridSearchCV(rf, param_grid, cv=5)

grid.fit(X_train, y_train)

rf_model = grid.best_estimator_

rf_pred = rf_model.predict(X_test)

rf_score = r2_score(y_test, rf_pred)

print("Random Forest R2:", rf_score)

# -----------------------------
# SELECT BEST MODEL
# -----------------------------

if rf_score > lr_score:
    best_model = rf_model
    best_name = "Random Forest"
    best_score = rf_score
else:
    best_model = lr_model
    best_name = "Linear Regression"
    best_score = lr_score

print("\nBest Model Selected:", best_name)
print("Best R2 Score:", best_score)

# -----------------------------
# SAVE MODEL + ENCODERS
# -----------------------------

pickle.dump(best_model, open("model.pkl","wb"))
pickle.dump(le, open("country_encoder.pkl","wb"))
pickle.dump(feature_columns, open("feature_columns.pkl","wb"))

print("\nModel saved successfully for deployment.")