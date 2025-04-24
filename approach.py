import numpy as np
import pandas as pd
import heapq
from collections import deque

""" Evaluation metric(accumulate error) """
def linear_approximation_error(f, x_start, x_end, f_start, f_end, num_samples=10):
    """
    compute accumulate absolute error
    """
    t_vals = np.linspace(0, 1, num_samples)
    accumulate_error = 0.0
    for t in t_vals:
        x_interp = x_start + t * (x_end - x_start)
        f_interp = f_start + t * (f_end - f_start)
        f_actual = f(x_interp)
        accumulate_error += abs(f_actual - f_interp)
    return accumulate_error 


""" Baseline one step linear"""
def baseline_one_step(prototypes, f, X_target, avg_abs_pred, num_points=10):
    """
    Baseline 1: Linear Model with Fixed Intercept fitted using inserted data.
    
    This function selects the prototype closest to X_target, then generates a set
    of interpolation points between that prototype (x_start) and X_target (x_end). 
    It fits a linear model with the intercept fixed to f(x_start) (i.e., the function 
    value at the prototype) by estimating the slope using least squares on all inserted points.
    The predicted endpoint is computed as f(x_start) + slope, and the cumulative error 
    is calculated using the linear_approximation_error function.
    
    Args:
        prototypes (np.ndarray): Array of prototype instances.
        f (callable): Function to evaluate.
        X_target (np.ndarray): Target instance.
        avg_abs_pred (float): Average absolute prediction value for normalization.
        num_points (int): Number of interpolation points.
    
    Returns:
        tuple: (raw_cumulative_error, normalized_cumulative_error)
    """
    import numpy as np

    # Select the closest prototype to X_target.
    distances = np.linalg.norm(prototypes - X_target, axis=1)
    closest_proto_index = np.argmin(distances)
    x_start = prototypes[closest_proto_index]
    x_end = X_target
    
    # Generate trajectory of points between X_proto and X_target.
    path = np.array([x_start + (x_end - x_start) * (i / (num_points - 1)) for i in range(num_points)])
    
    # Evaluate the function on the trajectory.
    f_values = np.array([f(x) for x in path])
    
    X_features = path - x_start              # shape: (num_points, d)
    Y_targets = f_values - f_values[0]         # shape: (num_points,)
    w, residuals, rank, s = np.linalg.lstsq(X_features, Y_targets, rcond=None)
    predictions = f_values[0] + np.dot((path - x_start), w)

    # Cumulative absolute error
    raw_cumulative_error = np.sum(np.abs(f_values - predictions))
    normalized_cumulative_error = raw_cumulative_error / avg_abs_pred
    
    errors = np.abs(f_values - predictions)
    
    
    # print("Linear Model Fitted using Inserted Data:")
    # for i, (x_val, actual, pred) in enumerate(zip(path, f_values, predictions)):
    #     print(f"Step {i}: x = {x_val}, f(x) = {actual}, prediction = {pred}")
    print("Raw Cumulative Error:", raw_cumulative_error)
    print("error:",errors)
    print("Normalized Cumulative Error:", normalized_cumulative_error)
    
    return raw_cumulative_error, normalized_cumulative_error

  
""" Baseline gradient based"""   
def baseline_grad(f,grad_f, prototypes, X_target, avg_abs_pred, p, max_steps=5, tol=1e-6):
    """
    Baseline 2: Gradient Descent-Based Search using a finite difference approximation.
    
    This function starts from the prototype that is closest to the target and performs
    gradient descent with an update rule:
    
        x_{t+1} = x_t - (1/p) * grad_f(x_t)
    
    The process stops when the current point is within a tolerance of X_target or after a
    maximum number of steps.
    
    Args:
        prototypes (np.ndarray): Array of prototype instances.
        X_target (np.ndarray): Target instance.
        avg_abs_pred (float): Average absolute prediction value for normalization.
        p (float): Parameter that scales the gradient step (step size is 1/p).
        max_steps (int): Maximum number of gradient descent steps.
        tol (float): Tolerance for convergence toward X_target.
    
    Returns:
        tuple: (raw_cumulative_error, normalized_cumulative_error)
    """
    # Find the prototype closest to the target.
    distances = np.linalg.norm(prototypes - X_target, axis=1)
    closest_proto_index = np.argmin(distances)
    X_proto = prototypes[closest_proto_index]
    
    f_target = f(X_target)
    path = [X_proto.copy()]
    f_values = [f(X_proto)]
    x_current = X_proto.copy()
    
    for step in range(max_steps):
        # Approximate the gradient at the current point.
        grad = grad_f(x_current)
        # Update the current point.
        x_new = x_current - (1 / p) * grad
        path.append(x_new.copy())
        f_values.append(f(x_new))
        
        # Check if the new point is close enough to the target.
        if np.linalg.norm(x_new - X_target) < tol:
            break
        
        x_current = x_new
    
    raw_cumulative_error = sum(abs(val - f_target) for val in f_values)
    normalized_cumulative_error = raw_cumulative_error / avg_abs_pred
    
    print("Gradient Descent-Based Search (Baseline 2) Path:")
    for i, (x_val, fx_val) in enumerate(zip(path, f_values)):
        print(f"Step {i}: x = {x_val}, f(x) = {fx_val}")
    print("Raw Cumulative Error:", raw_cumulative_error)
    print("Normalized Cumulative Error:", normalized_cumulative_error)
    
    return raw_cumulative_error, normalized_cumulative_error
  
# ====================================== our approach ==================================================


def heuristic(f,current_x, target_x, current_f, target_f, num_samples=10):
    """
    A simple heuristic: the local linear error from the current point to the target.
    This is admissible if f is reasonably smooth.
    """
    return linear_approximation_error(f,current_x, target_x, current_f, target_f, num_samples)

def heuristic(f,current_x, target_x, current_f, target_f, num_samples=10):
    """
    A simple heuristic: the local linear error from the current point to the target.
    This is admissible if f is reasonably smooth.
    """
    return linear_approximation_error(f, current_x, target_x, current_f, target_f, num_samples)

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

# def a_star_search(initial_state, initial_x, f_proto, target_x, f_target, grid, max_steps, monotonic_increasing, tol=1e-6):
def a_star_search(initial_state, initial_x, f_proto, target_x, f_target,f, grid, max_steps, monotonic_increasing, tol=1e-6):
    """
    A* search in the discretized feature space.
    """
    # Get the actual number of partitions for each dimension
    partitions = [len(grid[i]) - 1 for i in range(len(grid))]
    d = len(initial_state)
    counter = 0
    
    # Each item in the heap is a tuple: (estimated_total_cost, cumulative_error, state, path, f_values, steps)
    start_item = (heuristic(f,initial_x, target_x, f_proto, f_target), 0.0,counter, initial_state, [initial_x], [f_proto], 0)
    heap = [start_item]
    best_solution = None
    visited = {}  # records the best cumulative error reached for a given state

    while heap:
        est_cost, cum_error, counter, state, path, f_values, steps = heapq.heappop(heap)
        
        # Skip if we've seen this state with a lower cost already
        if state in visited and visited[state] <= cum_error:
            continue
        visited[state] = cum_error
        
        # Check if we've reached the target state
        # print("state")
        if all(state[i] == partitions[i] for i in range(d)):
            best_solution = {'path': path, 'f_values': f_values, 'error': cum_error}
            break
        
        # If maximum steps have been reached, update the best solution if it's better
        if steps >= max_steps:
            # if best_solution is None or cum_error < best_solution['error']:
            #     best_solution = {'path': path, 'f_values': f_values, 'error': cum_error}
            continue

        current_x = np.array([grid[i][state[i]] for i in range(d)])
        current_f = f_values[-1]
        
        # Expand the current state: try to update one feature at a time
        for i in range(d):
            if state[i] < partitions[i]:  # Check against dimension-specific partition size
                # Allow stepping 1 to the remaining number of grid points in dimension i
                for step_size in range(1, partitions[i] - state[i] + 1):
                    new_state = list(state)
                    new_state[i] += step_size
                    new_state = tuple(new_state)
                    
                    new_x = np.array([grid[j][new_state[j]] for j in range(d)])
                    new_f = f(new_x)
                    
                    # Check monotonicity constraints (with tolerance)
                    # if monotonic_increasing and new_f < current_f - tol:
                    #     continue
                    # if not monotonic_increasing and new_f > current_f + tol:
                    #     continue
                    # # Check monotonicity constraints (with tolerance)
                            
                    local_err = linear_approximation_error(f, current_x, new_x, current_f, new_f)
                    new_cum_error = cum_error + local_err
                    
                    # Compute heuristic from new state to target
                    h = heuristic(f,new_x, target_x, new_f, f_target)
                  
                    # new_est_cost = new_cum_error + h
                    new_est_cost = float(new_cum_error + h)
                    new_path = path + [new_x]
                    new_f_values = f_values + [new_f]
                    # print("1",new_est_cost, new_cum_error, new_state, new_path, new_f_values)
                    if not is_unimodal(new_f_values, tol):
                        continue
                    counter += 1
                    # heapq.heappush(heap, (new_est_cost, new_cum_error, counter, new_state, new_path, new_f_values, steps + 1))
                    heapq.heappush(heap, (new_est_cost, new_cum_error, counter, new_state, new_path, new_f_values, steps + 1))    
    return best_solution

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

from sklearn.cluster import KMeans
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


def run_search_path(f,prototypes, prototype_labels, X_target,partitions=2,max_steps=5):
    # Choose a test instance to explain (e.g., the first one)
    f_target = f(X_target)
    # Define multiple prototype candidates
    # Compute distances to X_target
    distances = np.linalg.norm(prototypes - X_target, axis=1)
    # Select the closest prototype
    closest_proto_index = np.argmin(distances)
    X_proto = prototypes[closest_proto_index]
    X_proto_label = prototype_labels[closest_proto_index]
    #### identify noise directions
    x_proto_adj, variances, low_variance_features, high_variance_features = dynamic_feature_filter(f, X_proto, X_target)
    grid = generate_grid_with_filter(x_proto_adj, X_target, partitions, low_variance_features)
    
    
    d = len(X_proto)
    initial_state = tuple([0] * d)
    initial_x = np.array([grid[i][0] for i in range(d)])  # 与 x_proto 应相同
    f_proto = f(initial_x)
    f_target = f(X_target)
    # identify the monotonic direction
    monotonic_increasing = (f_target >= f_proto)

    print("X_proto",X_proto, 'org_proto', f(X_proto), "label", X_proto_label)
    print("X_proto",x_proto_adj, 'pred_prototype adj', f(x_proto_adj))
    print("X_target",X_target,'pred_target', f_target)
    # result = dfs(initial_state, [initial_x], [f_proto], 0.0, grid, max_steps, monotonic_increasing, tol=1e-6)
    result = a_star_search(initial_state, initial_x, f_proto, X_target, f_target, f, grid, max_steps,monotonic_increasing)
    #
    if result is not None:
        print("Found a path:")
        for i, (x_val, fx_val) in enumerate(zip(result['path'], result['f_values'])):
            print(f"Step {i}: x = {x_val}, f(x) = {fx_val}")
        # print(len(result['path']))
        avg_error = result['error'] / (len(result['path']))
        return result['error'], avg_error 
    else:
        print("No path found.")
        return None
    
