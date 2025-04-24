# -*- coding: utf-8 -*-
import numpy as np
import heapq
from collections import deque
from eval import linear_approximation_error
from utils import dynamic_feature_filter, generate_grid_with_filter, is_unimodal


def heuristic(f,current_x, target_x, current_f, target_f, num_samples=10):
    """
    A simple heuristic: the local linear error from the current point to the target.
    This is admissible if f is reasonably smooth.
    """
    return linear_approximation_error(f, current_x, target_x, current_f, target_f, num_samples)

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
    initial_x = np.array([grid[i][0] for i in range(d)])  #  x_proto 
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
    
