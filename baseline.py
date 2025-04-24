import numpy as np
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
  