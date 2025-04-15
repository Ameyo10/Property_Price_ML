import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

train_data = pd.read_csv(r'D:\Python Projects\Python-Projects\Machine_Learning_Projects_Discord\Property_Price\train.csv')
train_df = pd.DataFrame(train_data)

# Separate features and target
x = train_df.drop(columns=["SalePrice"])
y = train_df['SalePrice']

# One-hot encode categorical features
x_encoded = pd.get_dummies(x, drop_first=True)

# Feature selection
mi = mutual_info_regression(x_encoded, y)
mi_series = pd.Series(mi, index=x_encoded.columns)
top_mi_features = mi_series.sort_values(ascending=False).head(5).index.tolist()

df_encoded = x_encoded.copy()
df_encoded['SalePrice'] = y
corr_matrix = df_encoded.corr()
corr_series = corr_matrix['SalePrice'].drop('SalePrice')
top_corr_features = corr_series[abs(corr_series) > 0.5].sort_values(ascending=False).head(5).index.tolist()

important_features = list(dict.fromkeys(top_mi_features + top_corr_features))
X = x_encoded[important_features]

# Pipeline and parameter grid
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

param_grid = [
    {'regressor': [Ridge()], 'regressor__alpha': [0.1, 1, 10]},
    {'regressor': [LinearRegression()]},
    {'regressor': [KNeighborsRegressor()], 'regressor__n_neighbors': [3, 5, 7]},
    {'regressor': [RandomForestRegressor()], 'regressor__n_estimators': [100, 200]}
]

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='r2',
    return_train_score=True,
    n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid.fit(X_train, y_train)

#Predicting the datas
pred = grid.predict(X_test)
r2 = r2_score(y_test, pred)
print(f'R-squared: {r2:.4f}')
print("Best parameters:", grid.best_params_)

# Saving the best model
joblib.dump(grid.best_estimator_, 'property_price_model.pkl')

# Saving the important features (needed for new data processing)
joblib.dump(important_features, 'important_features.pkl')