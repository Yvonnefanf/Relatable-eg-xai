import numpy as np
# Grid generation that fixes low-variance features.
def generate_grid_with_filter(x_proto_adj, x_target, partitions, low_variance_features):
    """
    For low-variance features, use a grid with a single point (the target).
    For high-variance features, generate a grid between x_proto_adj and x_target.
    """
    grid = []
    n_features = len(x_proto_adj)
    for i in range(n_features):
        if i in low_variance_features:
            grid.append(np.array([x_target[i]]))
        else:
            grid.append(np.linspace(x_proto_adj[i], x_target[i], partitions + 1))
    return grid

def dynamic_feature_filter(f,x_proto, x_target, num_samples=10, threshold=0.1):
    """
    classify features as low- or high-sensitivity by computing
    the variance of predictions when varying each feature, and then clustering 
    the variances using k-means.
    
    Parameters:
      x_proto: 1D numpy array for the prototype.
      x_target: 1D numpy array for the target.
      num_samples: Number of interpolation points per feature.
    
    Returns:
      variances: Array of variance values for each feature.
      low_variance_features: List of indices of features with low sensitivity.
      high_variance_features: List of indices of features with high sensitivity.
    """
    x_proto_adj = np.copy(x_proto)
    n_features = len(x_proto)
    variances = np.zeros(n_features)
    low_variance_features = []
    high_variance_features = []
    for i in range(n_features):
        # Generate interpolation values for the i-th feature.
        interp_vals = np.linspace(x_proto[i], x_target[i], num_samples)
        preds = []
        for val in interp_vals:
            x_temp = np.copy(x_proto)
            x_temp[i] = val  # Vary only the i-th feature.
            preds.append(f(x_temp))
        preds = np.array(preds)
        var_val = np.var(preds)
        variances[i] = var_val
        if var_val < threshold:
          x_proto_adj[i] = x_target[i]
          low_variance_features.append(i)
        else:
          high_variance_features.append(i)
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
def explain_prediction(solver, X,x_target, threshold=0.1):
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
            significant_changes = [(name, change) for name, change in zip(X.columns, changes) if abs(change) > 0.01]
            if significant_changes:
                print("Changes made:")
                for name, change in significant_changes:
                    direction = "↑" if change > 0 else "↓"
                    print(f"  {name}: {direction} {abs(change):.2f}")
        
        print("\n=== Final Result ===")
        print(f"Total Steps: {len(result.path) - 1}")
        print(f"Path Error: {result.error:.4f}")
        price_diff = (result.f_values[-1] - result.f_values[0])
        print(f"Total Price Change: ${'+' if price_diff >= 0 else ''}{price_diff:.2f}")
    else:
        print("Could not find a suitable explanation path")
