import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from Approach_ import RelatabilitySolver
from utils import explain_prediction
import sys

# 1. Load data
try:
    pd_data = pd.read_csv('./datasets/salary.csv')
except FileNotFoundError:
    print("Error: salary.csv not found in ./datasets/. Please ensure the file exists.")
    sys.exit(1)

# Handle potential NaN values (example: fill with mode or median, or drop)
# For simplicity, dropping rows with any NaNs for now
pd_data.dropna(inplace=True)
pd_data.reset_index(drop=True, inplace=True)


# 2. Separate Features (X) and Target (y)
X = pd_data.drop(columns=['Salary'])
y = pd_data['Salary']

# 3. Identify categorical and numerical features
categorical_features = ['Gender', 'Education Level', 'Job Title']
numerical_features = ['Age', 'Years of Experience']

# Ensure numerical features are numeric
for col in numerical_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X.dropna(subset=numerical_features, inplace=True) # Drop rows where conversion failed
y = y[X.index] # Align y with X after dropping NaNs

# 4. Create Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # sparse_output=False for dense array
    ],
    remainder='passthrough' # Keep other columns if any (none in this case)
)

# 5. Split data (before preprocessing)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Fit preprocessor on training data
preprocessor.fit(X_train_raw)

# 7. Transform data
X_train_processed = preprocessor.transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

# Get feature names after transformation (useful for debugging/interpretation)
feature_names_processed = preprocessor.get_feature_names_out()
# Convert processed data back to DataFrame for easier handling if needed (optional)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_processed, index=X_train_raw.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_processed, index=X_test_raw.index)


# 8. Train Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_processed, y_train) # Train on processed data

# Evaluate Model
preds_test_processed = rf.predict(X_test_processed)
r2_score_ = r2_score(y_test, preds_test_processed)
print(f"R2 Score: {r2_score_:.4f}")
rmse_score = np.sqrt(np.mean((y_test - preds_test_processed) ** 2))
print(f"RMSE Score: {rmse_score:.2f}")

# Predictions on training data (needed for prototypes/explanation context)
train_preds_processed = rf.predict(X_train_processed)


# 9. Define function f(x) - takes a RAW instance, preprocesses, then predicts
def f(x_raw):
    # Ensure x_raw is in the correct format (e.g., DataFrame row or dict)
    if isinstance(x_raw, np.ndarray):
        # Assuming order matches X_train_raw.columns if it's a numpy array
         x_df = pd.DataFrame([x_raw], columns=X_train_raw.columns)
    elif isinstance(x_raw, dict):
         x_df = pd.DataFrame([x_raw])
    else: # Assume it's already a DataFrame/Series compatible row
        x_df = x_raw if isinstance(x_raw, pd.DataFrame) else pd.DataFrame([x_raw])

    # Handle potential shape issues (e.g., 1D array to 2D)
    if x_df.ndim == 1:
        x_df = x_df.to_frame().T # Convert Series to DataFrame row

    # Ensure columns match the ones preprocessor was trained on
    x_df = x_df[X_train_raw.columns]

    x_processed = preprocessor.transform(x_df)
    return float(rf.predict(x_processed)[0])

# 10. Select Prototypes (using processed data for predictions and values)
min_pred_idx = np.argmin(train_preds_processed)
max_pred_idx = np.argmax(train_preds_processed)
mean_pred_value = np.mean(train_preds_processed)
closest_mean_pred_idx = np.argmin(np.abs(train_preds_processed - mean_pred_value))

# Get the corresponding *processed* feature vectors for the prototypes
prototype_indices_in_train = [min_pred_idx, closest_mean_pred_idx, max_pred_idx]
# Need to map these indices back to the original X_train_processed array indices
# Since train_preds_processed corresponds directly to X_train_processed rows:
prototypes_processed = X_train_processed[prototype_indices_in_train]
# Get the corresponding labels (salary)
prototype_labels = y_train.iloc[prototype_indices_in_train].values


# 11. Initialize the RelatabilitySolver
# Use processed prototypes. No categorical_indices needed as data is preprocessed.
np.set_printoptions(suppress=True)
solver = RelatabilitySolver(
    f=f, # f takes raw data, preprocesses, predicts
    prototypes=prototypes_processed, # Solver works on processed data space
    prototype_labels=prototype_labels,
    partitions=4,
    max_steps=5
    # No categorical_indices here - solver operates on the numerical space after OHE
)

# 12. Explain Prediction
if len(sys.argv) > 1:
    try:
        index = int(sys.argv[1])
        if index < 0 or index >= len(X_test_raw):
             print(f"Error: Index {index} is out of bounds for the test set (size {len(X_test_raw)}). Using index 0.")
             index = 0
    except ValueError:
        print(f"Error: Invalid index '{sys.argv[1]}'. Using index 0.")
        index = 0
else:
    index = 0

# Get the RAW test instance
test_instance_raw = X_test_raw.iloc[index]
# Get the corresponding PROCESSED test instance for the solver
test_instance_processed = X_test_processed[index]
actual_salary = y_test.iloc[index]
predicted_salary = f(test_instance_raw) # Use f() which handles preprocessing

print(f"\n=== Explaining Instance {index} (Raw Features) ===")
print(test_instance_raw.to_string())
print(f"\nActual Salary: ${actual_salary:.2f}")
print(f"Predicted Salary: ${predicted_salary:.2f}")
print("\nGenerating explanation...")

# Explain using PROCESSED data, as the solver operates in that space
# Pass X_train_processed_df so explain_prediction can try to find similar *processed* cases
explain_prediction(
    solver=solver,
    X=X_train_processed_df, # Pass processed train features DataFrame
    y=y_train,
    X_pred=pd.Series(train_preds_processed, index=X_train_processed_df.index), # Pass processed train predictions
    x_target=test_instance_processed, # Pass the processed target instance
    threshold=0.005 # Adjust threshold based on processed data sensitivity if needed
)

# Note: explain_prediction output will show feature indices/names from the *processed* space.
# Further work needed in explain_prediction to map back to original feature names for better readability.


