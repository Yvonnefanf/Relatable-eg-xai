import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from Approach_ import RelatabilitySolver  # Import lver
from search_strategies import AStarStrategy, GreedyBestFirstStrategy
import time
from utils import explain_prediction

# Load dataxq
pd_data = pd.read_csv('./datasets/boston.csv')

X = pd_data.drop(columns=['B','MEDV'])
y = pd_data['MEDV'] 

X.columns = [
    'Crime Rate',
    '% Residential in Area', 
    '% Business in Area', 
    'River', 
    'Air Pollution',
    '#Rooms',
    '% Pre-1940 Units',
    'Dist to Business District', 
    'Highway Accessibility',
    'Property Tax Rate',
    'Student-Teacher Ratio',
    '% Lower Income in Area',
]

# Identify categorical features
categorical_feature_names = ['River']
categorical_indices = [X.columns.get_loc(col) for col in categorical_feature_names if col in X.columns]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# 训练测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test) 
r2_socre_ = r2_score(y_test, preds)
print("R2 Score:", r2_socre_)
rmse_score = np.sqrt(np.mean((y_test - preds) ** 2))
print("RMSE Score:", rmse_score)

train_preds = rf.predict(X_train)

min_index = np.argmin(train_preds)  # min val index
max_index = np.argmax(train_preds)  # max val index
mean_value = np.mean(train_preds)   # mean val 

closest_mean_index = np.argmin(np.abs(train_preds - mean_value))  # 找到最接近均值的索引

# Define function f(x) using rf.predict.
def f(x):
    x_df = pd.DataFrame([x], columns=X.columns)
    return float(rf.predict(x_df)[0])

np.set_printoptions(suppress=True)  #
prototypes = X_train.iloc[[min_index, closest_mean_index, max_index]].values
prototype_labels = y_train.iloc[[min_index, closest_mean_index, max_index]].values

# Initialize the RelatabilitySolver
solver = RelatabilitySolver(
    f=f,
    prototypes=prototypes,
    prototype_labels=prototype_labels,
    partitions=4,  # number of grid points between prototype and target
    max_steps=5,    # maximum steps in the path
    categorical_indices=categorical_indices # Pass the indices here
)




import sys
if len(sys.argv) > 1:
    index = int(sys.argv[1])
else:
    index = 0

# Example: Explain a prediction for a test house
test_house = X_test.iloc[index].values  # Take first house from test set as example
print(f"=== Target House {index}===")
print(f"Actual price: ${y_test.iloc[index]:.2f}")
print(f"Predicted price: ${f(test_house):.2f}")
print("\nGenerating explanation...")


preds_train = rf.predict(X_train) 
# Find and show explanation
explain_prediction(solver, X_train, y_train, preds_train,test_house, threshold=0.1)


