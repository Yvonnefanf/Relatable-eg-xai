import heapq
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from search_base import SearchStrategy
from eval import linear_approximation_error
from utils import is_unimodal

def heuristic(f: callable,
              current_x: np.ndarray,
              target_x: np.ndarray,
              current_f: float,
              target_f: float,
              num_samples: int = 10) -> float:
    """
    A simple heuristic: the local linear error from the current point to the target.
    This is admissible if f is reasonably smooth.
    
    Args:
        f: The function being searched
        current_x: Current point in the search
        target_x: Target point
        current_f: Function value at current point
        target_f: Function value at target point
        num_samples: Number of samples for linear approximation
        
    Returns:
        float: Estimated cost to target
    """
    return linear_approximation_error(f, current_x, target_x, current_f, target_f, num_samples)

class AStarStrategy(SearchStrategy):
    """A* search strategy implementation"""
    
    def search(self, base) -> Optional[Dict[str, Any]]:
        """
        Implements A* search algorithm to find a path from initial state to target.
        
        Args:
            base: SearchPathBase instance containing search configuration
            
        Returns:
            Optional[Dict]: Search results if successful, None otherwise
        """
        f = base.f
        grid = base.grid
        counter = 0
        
        # Initialize search with start state
        # Heap stores: (est_cost, cum_error, counter, state)
        start_cost = heuristic(f, base.initial_x, base.X_target, base.f_proto, base.f_target)
        start_item = (
            start_cost, # estimated cost
            0.0,        # cumulative error
            counter,    # tie-breaker
            base.initial_state
        )
        heap = [start_item]
        
        # Visited maps state -> min_cumulative_error
        visited = {}
        # Path_data maps state -> (path, f_values)
        path_data = {base.initial_state: ([base.initial_x], [base.f_proto])}
        # We need to store steps associated with state as well
        step_count = {base.initial_state: 0}

        while heap:
            est_cost, cum_error, _, state = heapq.heappop(heap) # counter (_) is only for tie-breaking

            # Retrieve path, f_values, and steps for the current state
            path, f_values = path_data[state]
            steps = step_count[state]

            # Skip if we've found a better path to this state already
            if state in visited and visited[state] <= cum_error:
                continue
            visited[state] = cum_error
            
            # Goal check
            if all(state[i] == len(grid[i]) - 1 for i in range(base.d)):
                # Path found, return results from path_data
                return {
                    'path': path,
                    'f_values': f_values,
                    'error': cum_error
                }

            # Max steps check
            if steps >= base.max_steps:
                continue

            current_x = path[-1] # Get current_x from the path
            current_f = f_values[-1]

            # Explore neighbors
            for i in range(base.d):
                if state[i] < len(grid[i]) - 1:
                    for step_size in range(1, len(grid[i]) - state[i]):
                        new_state_list = list(state)
                        new_state_list[i] += step_size
                        new_state = tuple(new_state_list)
                        
                        new_x = np.array([grid[j][new_state[j]] for j in range(base.d)])
                        # --- Debug prints start ---
                        # print(f"DEBUG: Dimension={i}, StepSize={step_size}, NewState={new_state}") 
                        # print(f"DEBUG: new_x = {new_x}")
                        try:
                            new_f = f(new_x)
                            # print(f"DEBUG: f(new_x) = {new_f}")
                        except Exception as e:
                            print(f"ERROR in f(new_x): {e}")
                            print(f"Input new_x was: {new_x}")
                            raise 

                        try:
                            local_err = linear_approximation_error(
                                f, current_x, new_x, current_f, new_f)
                            # print(f"DEBUG: local_err = {local_err}")
                        except Exception as e:
                            print(f"ERROR in linear_approximation_error:")
                            print(f"Inputs: current_x={current_x}, new_x={new_x}, current_f={current_f}, new_f={new_f}")
                            print(f"Error: {e}")
                            raise 
                            
                        new_cum_error = cum_error + local_err
                        
                        try:
                            h = heuristic(f, new_x, base.X_target, new_f, base.f_target)
                            # print(f"DEBUG: heuristic = {h}")
                        except Exception as e:
                            print(f"ERROR in heuristic:")
                            print(f"Inputs: new_x={new_x}, target_x={base.X_target}, new_f={new_f}, target_f={base.f_target}")
                            print(f"Error: {e}")
                            raise 
                            
                        new_est_cost = new_cum_error + h
                        # print(f"DEBUG: new_cum_error={new_cum_error}, h={h}, new_est_cost={new_est_cost}")
                        # --- Debug prints end ---
                        
                        new_path = path + [new_x]
                        new_f_values = f_values + [new_f]
                        new_steps = steps + 1

                        # Check unimodal before considering push
                        try:
                            is_uni = is_unimodal(new_f_values)
                            # print(f"DEBUG: is_unimodal = {is_uni}") # DEBUG
                            if not is_uni:
                                # print(f"DEBUG: Path not unimodal, skipping.") # DEBUG
                                continue
                        except Exception as e:
                             print(f"ERROR in is_unimodal:")
                             print(f"Input f_values = {new_f_values}")
                             print(f"Error: {e}")
                             raise

                        # Check if we already found a better path to new_state
                        if new_state in visited and visited[new_state] <= new_cum_error:
                            continue
                            
                        # If new_state not visited or this path is better, add to heap
                        counter += 1
                        heap_item = (
                            new_est_cost,
                            new_cum_error,
                            counter, # Tie-breaker
                            new_state
                        )
                        # print(f"DEBUG: Pushing to heap: est_cost={heap_item[0]}, cum_error={heap_item[1]}, state={heap_item[3]}") # DEBUG
                        heapq.heappush(heap, heap_item)
                        # Store path and steps data associated with this state exploration
                        path_data[new_state] = (new_path, new_f_values)
                        step_count[new_state] = new_steps
        
        # print("DEBUG: A* search finished without finding a path.") # DEBUG
        return None  # No path found

class GreedyBestFirstStrategy(SearchStrategy):
    """
    Greedy Best-First Search strategy.
    Faster than A* but may not find optimal path.
    Only considers heuristic cost to target, ignoring path cost.
    """
    
    def search(self, base) -> Optional[Dict[str, Any]]:
        f = base.f
        grid = base.grid
        counter = 0
        
        # Initialize with start state - only using heuristic cost
        start_item = (
            heuristic(f, base.initial_x, base.X_target, base.f_proto, base.f_target),
            counter,
            base.initial_state,
            [base.initial_x],
            [base.f_proto],
            0,
            0.0  # cumulative error (kept but not used for decisions)
        )
        
        heap = [start_item]
        visited = set()

        while heap:
            h_cost, counter, state, path, f_values, steps, cum_error = heapq.heappop(heap)
            
            if state in visited:
                continue
            visited.add(state)

            # Check if target reached
            if all(state[i] == len(grid[i]) - 1 for i in range(base.d)):
                return {
                    'path': path,
                    'f_values': f_values,
                    'error': cum_error
                }

            if steps >= base.max_steps:
                continue

            current_x = np.array([grid[i][state[i]] for i in range(base.d)])
            current_f = f_values[-1]

            # Consider larger steps first (opposite of A*)
            for i in range(base.d):
                if state[i] < len(grid[i]) - 1:
                    remaining_steps = len(grid[i]) - state[i]
                    for step_size in range(remaining_steps - 1, 0, -1):
                    
                        new_state = list(state)
                        new_state[i] += step_size
                        new_state = tuple(new_state)
                        
                        new_x = np.array([grid[j][new_state[j]] for j in range(base.d)])
                        new_f = f(new_x)

                        # Calculate error (for return value only)
                        local_err = linear_approximation_error(
                            f, current_x, new_x, current_f, new_f)
                        new_cum_error = cum_error + local_err
                        
                        # Only use heuristic for decisions
                        h = heuristic(f, new_x, base.X_target, new_f, base.f_target)

                        new_path = path + [new_x]
                        new_f_values = f_values + [new_f]
                        
                        if not is_unimodal(new_f_values):
                            continue

                        counter += 1
                        heapq.heappush(heap, (
                            h,  # only heuristic cost
                            counter,
                            new_state,
                            new_path,
                            new_f_values,
                            steps + 1,
                            new_cum_error
                        ))
        return None

