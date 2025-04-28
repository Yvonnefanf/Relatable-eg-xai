import numpy as np
import pandas as pd
# Grid generation that fixes low-variance features.
def generate_grid_with_filter(x_proto_adj, x_target, partitions, low_variance_features, categorical_indices=None):
    """
    For low-variance features, use a grid with a single point (the target).
    For high-variance categorical features, use a two-point grid [proto, target].
    For high-variance numerical features, generate a grid between x_proto_adj and x_target.
    """
    if categorical_indices is None:
        categorical_indices = []
    grid = []
    n_features = len(x_proto_adj)
    for i in range(n_features):
        if i in low_variance_features:
            # Low variance (numerical or categorical) -> only target value
            grid.append(np.array([x_target[i]]))
        elif i in categorical_indices:
            # High variance categorical -> single step
            grid.append(np.array([x_proto_adj[i], x_target[i]]))
        else:
            # High variance numerical -> multi-step grid
            grid.append(np.linspace(x_proto_adj[i], x_target[i], partitions + 1))
    return grid

def dynamic_feature_filter(f, x_proto, x_target, num_samples=10, threshold=0.1, categorical_indices=None):
    """
    Classify features as low- or high-sensitivity.
    For numerical features: compute variance of predictions when varying the feature.
    For categorical features: classify as low-sensitivity if the value doesn't change, high otherwise.

    Parameters:
      f: Prediction function.
      x_proto: 1D numpy array for the prototype.
      x_target: 1D numpy array for the target.
      num_samples: Number of interpolation points per numerical feature.
      threshold: Sensitivity threshold for numerical features.
      categorical_indices: List of indices for categorical features.

    Returns:
      x_proto_adj: Adjusted prototype where low-sensitivity numerical features are set to target value.
      variances: Array of variance values for numerical features (None for categorical).
      low_variance_features: List of indices of features with low sensitivity.
      high_variance_features: List of indices of features with high sensitivity.
    """
    if categorical_indices is None:
        categorical_indices = []
    x_proto_adj = np.copy(x_proto)
    n_features = len(x_proto)
    variances = np.full(n_features, np.nan)  # Initialize variances with NaN
    low_variance_features = []
    high_variance_features = []

    for i in range(n_features):
        if i in categorical_indices:
            # Handle categorical features
            if x_proto[i] == x_target[i]:
                low_variance_features.append(i)
                # x_proto_adj[i] remains x_proto[i] which is == x_target[i]
            else:
                high_variance_features.append(i)
                # x_proto_adj[i] remains x_proto[i], grid will handle the step
        else:
            # Handle numerical features (existing logic)
            interp_vals = np.linspace(x_proto[i], x_target[i], num_samples)
            preds = []
            for val in interp_vals:
                x_temp = np.copy(x_proto_adj) # Use x_proto_adj for context
                x_temp[i] = val
                preds.append(f(x_temp))
            preds = np.array(preds)
            var_val = np.var(preds)
            variances[i] = var_val # Store variance for numerical features

            if var_val < threshold:
                x_proto_adj[i] = x_target[i] # Adjust prototype for low-variance numerical
                low_variance_features.append(i)
            else:
                high_variance_features.append(i)
                # x_proto_adj[i] remains x_proto[i] for high-variance numerical

    return x_proto_adj, variances, low_variance_features, high_variance_features


# make sure current path is simple enough
def is_unimodal(f_values, tol=1e-6):
    """
    if f_values most complex is unimodal
    can increase than decrease 
    - calculate diff between 
    """
    signs = []
    for i in range(len(f_values) - 1):
        diff = f_values[i+1] - f_values[i]
        if abs(diff) < tol:
            continue
        s = 1 if diff > 0 else -1
        if not signs or s != signs[-1]:
            signs.append(s)
    return len(signs) <= 2

# Function to explain a prediction and print the path details
def explain_prediction(solver, X, y, X_pred, x_target, threshold=0.1):
    """
    Explain a prediction for a given instance with clear step-by-step progression,
    including the most similar case from the dataset for each step.

    Args:
        solver: The pathfinding solver object.
        X: DataFrame or NumPy array of features for the dataset.
        y: Series or NumPy array of target labels for the dataset.
        X_pred: Series or NumPy array of pre-computed predictions for X.
        x_target: Features of the instance to explain.
        threshold: Sensitivity threshold for feature filtering.
    """
    result = solver.find_path(x_target, threshold=threshold)

    if result:
        print("\n=== Explanation Path Found ===")
        print("\nStarting Point (Prototype):")
        print(f"Predicted Price: ${result.f_values[0]:.2f}")

        for i in range(1, len(result.path)):
            print(f"\nStep {i}:")
            print("-" * 30)

            # Get current and previous points
            current = result.path[i]
            previous = result.path[i-1]
            changes = current - previous

            # Show price change
            price_change = (result.f_values[i] - result.f_values[i-1])
            print(f"Price: ${result.f_values[i]:.2f} ({'+' if price_change >= 0 else ''}{price_change:.2f})")

            # Show significant feature changes
            feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'feature_{j}' for j in range(X.shape[1])]
            significant_changes = [(name, change) for name, change in zip(feature_names, changes) if abs(change) > 0.01]
            if significant_changes:
                print("Changes made:")
                for name, change in significant_changes:
                    direction = "↑" if change > 0 else "↓"
                    print(f"  {name}: {direction} {abs(change):.2f}")

            # --- Find and print most similar case ---
            try:
                # Calculate distances (assuming X is DataFrame or numpy array)
                X_values = X.values if isinstance(X, pd.DataFrame) else X
                distances = np.linalg.norm(X_values - current, axis=1)
                most_similar_idx = np.argmin(distances)

                # Retrieve similar case data
                similar_case_y_actual = y.iloc[most_similar_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[most_similar_idx]
                # Retrieve pre-computed prediction for the similar case
                similar_case_y_pred = X_pred.iloc[most_similar_idx] if isinstance(X_pred, (pd.Series, pd.DataFrame)) else X_pred[most_similar_idx]

                print("\nMost Similar Case Found:")
                print(f"  Dataset Index: {most_similar_idx}")
                # You might want to print some features of the similar case here
                # similar_case_X = X.iloc[most_similar_idx] if isinstance(X, pd.DataFrame) else X[most_similar_idx]
                # print(f"  Features: {similar_case_X}")
                print(f"  Actual Label: ${similar_case_y_actual:.2f}")
                print(f"  Predicted Label: ${similar_case_y_pred:.2f}")

            except IndexError:
                 print(f"\nWarning: Index {most_similar_idx} out of bounds for X_pred or y.")
            except Exception as e:
                print(f"\nCould not find or process similar case for step {i}: {e}")
            # --- End similar case ---

        print("\n=== Final Result ===")
        print(f"Total Steps: {len(result.path) - 1}")
        print(f"Path Error: {result.error:.4f}")
        price_diff = (result.f_values[-1] - result.f_values[0])
        print(f"Total Price Change: ${'+' if price_diff >= 0 else ''}{price_diff:.2f}")
    else:
        print("Could not find a suitable explanation path")
