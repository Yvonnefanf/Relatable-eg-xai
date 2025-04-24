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

