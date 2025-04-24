import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from Approach_ import RelatabilitySolver  # Import lver

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
    max_steps=5    # maximum steps in the path
)

# Function to explain a prediction and print the path details
def explain_prediction(x_target, threshold=0.1):
    """
    Explain a prediction for a given house with clear step-by-step progression.
    
    Args:
        x_target: Features of the house to explain
        threshold: Sensitivity threshold for feature filtering
    """
    result = solver.find_path(x_target, threshold=threshold)
    
    if result:
        print("\n=== Explanation Path Found ===")
        print("\nStarting Point (Prototype):")
        print(f"Predicted Price: ${result.f_values[0] * 1000:.2f}")
        
        for i in range(1, len(result.path)):
            print(f"\nStep {i}:")
            print("-" * 30)
            
            # Get current and previous points
            current = result.path[i]
            previous = result.path[i-1]
            changes = current - previous
            
            # Show price change
            price_change = (result.f_values[i] - result.f_values[i-1]) * 1000
            print(f"Price: ${result.f_values[i] * 1000:.2f} ({'+' if price_change >= 0 else ''}{price_change:.2f})")
            
            # Show significant feature changes
            significant_changes = [(name, change) for name, change in zip(X.columns, changes) if abs(change) > 0.01]
            if significant_changes:
                print("Changes made:")
                for name, change in significant_changes:
                    direction = "↑" if change > 0 else "↓"
                    print(f"  {name}: {direction} {abs(change):.2f}")
        
        print("\n=== Final Result ===")
        print(f"Total Steps: {len(result.path) - 1}")
        print(f"Path Error: {result.error:.4f}")
        price_diff = (result.f_values[-1] - result.f_values[0]) * 1000
        print(f"Total Price Change: ${'+' if price_diff >= 0 else ''}{price_diff:.2f}")
    else:
        print("Could not find a suitable explanation path")

# Example: Explain a prediction for a test house
test_house = X_test.iloc[1].values  # Take first house from test set as example
print("=== Target House ===")
print(f"Actual price: ${y_test.iloc[1] * 1000:.2f}")
print(f"Predicted price: ${f(test_house) * 1000:.2f}")
print("\nGenerating explanation...")

# Find and show explanation
explain_prediction(test_house, threshold=0.1)



