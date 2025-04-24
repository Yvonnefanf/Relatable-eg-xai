import numpy as np
import heapq
from collections import deque
from eval import linear_approximation_error
from utils import dynamic_feature_filter, generate_grid_with_filter, is_unimodal
from search_base import SearchPathBase
from search_strategies import AStarStrategy

# Default to A* search strategy if none provided
DEFAULT_STRATEGY = AStarStrategy()

class RelatabilitySolver(SearchPathBase):
    """
    Main solver class for finding relatable explanations using path search.
    Inherits from SearchPathBase to leverage the search infrastructure.
    """
    def __init__(self, f, prototypes, prototype_labels, partitions=2, max_steps=5, strategy=None):
        super().__init__(
            f=f,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            partitions=partitions,
            max_steps=max_steps,
            strategy=strategy or DEFAULT_STRATEGY
        )
    
    def find_path(self, X_target, threshold=0.1):
        """
        Find a relatable path from prototype to target.
        
        Args:
            X_target: Target point to explain
            threshold: Sensitivity threshold for feature filtering
            
        Returns:
            SearchResult or None: Search results if successful
        """
        self.prepare_search(X_target, threshold)
        return self.execute_search()

def heuristic(f,current_x, target_x, current_f, target_f, num_samples=10):
    """
    A simple heuristic: the local linear error from the current point to the target.
    This is admissible if f is reasonably smooth.
    """
    return linear_approximation_error(f, current_x, target_x, current_f, target_f, num_samples)

class SearchStrategy:
    def search(self, base):
        """
        Search method to be implemented by each strategy.
        Takes the SearchPathBase instance (`base`) so it can access:
        - base.grid
        - base.f
        - base.X_target
        - base.initial_state
        - base.max_steps
        etc.
        """
        raise NotImplementedError
    
class AStarStrategy(SearchStrategy):
    def search(self, base):
        f = base.f
        grid = base.grid
        counter = 0
        start_item = (
            heuristic(f, base.initial_x, base.X_target, base.f_proto, base.f_target),
            0.0, counter, base.initial_state,
            [base.initial_x], [base.f_proto], 0
        )
        heap = [start_item]
        visited = {}

        while heap:
            est_cost, cum_error, counter, state, path, f_values, steps = heapq.heappop(heap)

            if state in visited and visited[state] <= cum_error:
                continue
            visited[state] = cum_error

            if all(state[i] == len(grid[i]) - 1 for i in range(base.d)):
                return {'path': path, 'f_values': f_values, 'error': cum_error}

            if steps >= base.max_steps:
                continue

            current_x = np.array([grid[i][state[i]] for i in range(base.d)])
            current_f = f_values[-1]

            for i in range(base.d):
                if state[i] < len(grid[i]) - 1:
                    for step_size in range(1, len(grid[i]) - state[i]):
                        new_state = list(state)
                        new_state[i] += step_size
                        new_state = tuple(new_state)
                        new_x = np.array([grid[j][new_state[j]] for j in range(base.d)])
                        new_f = f(new_x)

                        local_err = linear_approximation_error(f, current_x, new_x, current_f, new_f)
                        new_cum_error = cum_error + local_err
                        h = heuristic(f, new_x, base.X_target, new_f, base.f_target)
                        new_est_cost = new_cum_error + h

                        new_path = path + [new_x]
                        new_f_values = f_values + [new_f]
                        if not is_unimodal(new_f_values):
                            continue

                        counter += 1
                        heapq.heappush(heap, (
                            new_est_cost, new_cum_error, counter,
                            new_state, new_path, new_f_values, steps + 1
                        ))
        return None

