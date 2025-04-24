import heapq
import numpy as np
from typing import Dict, Any, Optional

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
        start_item = (
            heuristic(f, base.initial_x, base.X_target, base.f_proto, base.f_target),
            0.0,  # cumulative error
            counter,
            base.initial_state,
            [base.initial_x],  # path
            [base.f_proto],    # f_values
            0                  # steps
        )
        
        heap = [start_item]
        visited = {}

        while heap:
            est_cost, cum_error, counter, state, path, f_values, steps = heapq.heappop(heap)

            # Skip if we've seen this state with lower error
            if state in visited and visited[state] <= cum_error:
                continue
            visited[state] = cum_error

            # Check if we've reached the target state
            if all(state[i] == len(grid[i]) - 1 for i in range(base.d)):
                return {
                    'path': path,
                    'f_values': f_values,
                    'error': cum_error
                }

            # Check if we've exceeded max steps
            if steps >= base.max_steps:
                continue

            current_x = np.array([grid[i][state[i]] for i in range(base.d)])
            current_f = f_values[-1]

            # Explore each dimension
            for i in range(base.d):
                if state[i] < len(grid[i]) - 1:
                    # Try different step sizes in this dimension
                    for step_size in range(1, len(grid[i]) - state[i]):
                        new_state = list(state)
                        new_state[i] += step_size
                        new_state = tuple(new_state)
                        
                        # Calculate new position and function value
                        new_x = np.array([grid[j][new_state[j]] for j in range(base.d)])
                        new_f = f(new_x)

                        # Calculate errors and estimated cost
                        local_err = linear_approximation_error(
                            f, current_x, new_x, current_f, new_f)
                        new_cum_error = cum_error + local_err
                        h = heuristic(f, new_x, base.X_target, new_f, base.f_target)
                        new_est_cost = new_cum_error + h

                        # Update path and check if it's unimodal
                        new_path = path + [new_x]
                        new_f_values = f_values + [new_f]
                        if not is_unimodal(new_f_values):
                            continue

                        # Add new state to heap
                        counter += 1
                        heapq.heappush(heap, (
                            new_est_cost,
                            new_cum_error,
                            counter,
                            new_state,
                            new_path,
                            new_f_values,
                            steps + 1
                        ))
        
        return None  # No path found 