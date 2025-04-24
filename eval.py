import numpy as np
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
    return accumulate_error / num_samples

