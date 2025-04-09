import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
dataset = pd.read_csv('d:/lol/GamingPrediction/Model/gaming_industry_trends.csv', encoding='latin-1')
dataset = dataset.rename(columns=lambda x: x.strip().lower())

# Updated column names to match your CSV file
dataset = dataset[['genre', 'platform', 'release year', 'players (millions)',
                  'peak concurrent players', 'metacritic score', 
                  'esports popularity', 'trending status', 'revenue (millions $)']]

# Convert all categorical variables using LabelEncoder
le = LabelEncoder()
categorical_columns = ['genre', 'platform', 'trending status']
for col in categorical_columns:
    dataset[col] = le.fit_transform(dataset[col].astype(str))

# Handle Esports Popularity separately as it might be categorical
dataset['esports popularity'] = pd.to_numeric(dataset['esports popularity'], errors='coerce')
if dataset['esports popularity'].isnull().any():
    # If it's categorical, encode it
    dataset['esports popularity'] = le.fit_transform(dataset['esports popularity'].fillna('None').astype(str))

# Handle other numeric columns
numeric_columns = ['release year', 'players (millions)', 'peak concurrent players', 
                  'metacritic score', 'revenue (millions $)']
for col in numeric_columns:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    dataset[col] = dataset[col].fillna(dataset[col].median())

print("\nFinal data shape:", dataset.shape)
print("\nSample of processed data:")
print(dataset.head())

X = dataset.drop(['revenue (millions $)'], axis=1)
y = dataset['revenue (millions $)']

sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

pickle.dump(rf_model, open("ml_model.sav", "wb"))
pickle.dump(sc, open("scaler.sav", "wb"))