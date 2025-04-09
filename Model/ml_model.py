import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

# Define mappings first
genre_map = {
    'Action': 0, 'Adventure': 1, 'RPG': 2, 'Strategy': 3, 'Sports': 4,
    'Racing': 5, 'Horror': 6, 'Fighting': 7, 'Shooter': 8, 'Simulation': 9
}

platform_map = {
    'PC': 0, 'PlayStation': 1, 'Xbox': 2, 'Nintendo Switch': 3,
    'Mobile': 4, 'Cross-Platform': 5
}

trending_map = {'Rising': 2, 'Stable': 1, 'Declining': 0}

# Load and preprocess data
dataset = pd.read_csv('d:/lol/GamingPrediction/Model/gaming_industry_trends.csv')

# Convert Peak Concurrent Players from millions to thousands
dataset['Peak Concurrent Players'] = dataset['Peak Concurrent Players'] * 1000

# Feature engineering
dataset['Genre'] = dataset['Genre'].map(genre_map)
dataset['Platform'] = dataset['Platform'].map(platform_map)
dataset['Trending Status'] = dataset['Trending Status'].map(trending_map)

# Adjusted player retention calculation for thousands of concurrent players
dataset['Player_Retention'] = (dataset['Peak Concurrent Players'] / (dataset['Players (Millions)'] * 1000000)) * 100
dataset['Recent_Release'] = (2024 - dataset['Release Year']).clip(upper=5)
dataset['High_Rating'] = (dataset['Metacritic Score'] >= 75).astype(int)

# Select features
X = dataset[[
    'Genre', 'Platform', 'Recent_Release',
    'Player_Retention', 'High_Rating', 'Trending Status'
]]

# Target variable
y = (dataset['Esports Popularity'] == 'Yes').astype(int)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use GradientBoostingClassifier with conservative parameters
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

# Evaluate model
train_pred = gb_model.predict(X_train_scaled)
test_pred = gb_model.predict(X_test_scaled)

print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))
print("Precision Score:", precision_score(y_test, test_pred))
print("Recall Score:", recall_score(y_test, test_pred))

# Save model and scaler
pickle.dump(gb_model, open("ml_model.sav", "wb"))
pickle.dump(scaler, open("scaler.sav", "wb"))