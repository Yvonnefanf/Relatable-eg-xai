from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Data class to hold search results"""
    path: List[np.ndarray]
    f_values: List[float]
    error: float
    success: bool = True

class SearchStrategy(ABC):
    """Abstract base class for all search strategies"""
    @abstractmethod
    def search(self, base: 'SearchPathBase') -> Optional[Dict[str, Any]]:
        """
        Search method to be implemented by each strategy.
        
        Args:
            base: SearchPathBase instance containing search configuration and state
            
        Returns:
            Optional[Dict]: Search results including path, f_values, and error if successful
        """
        pass

class SearchPathBase:
    """Base class for search path configuration and state management"""
    def __init__(self, 
                 f: callable,
                 prototypes: np.ndarray,
                 prototype_labels: np.ndarray,
                 partitions: int = 2,
                 max_steps: int = 5,
                 strategy: Optional[SearchStrategy] = None):
        self.f = f
        self.prototypes = prototypes
        self.prototype_labels = prototype_labels
        self.partitions = partitions
        self.max_steps = max_steps
        self.strategy = strategy
        
        # These will be set during prepare_search
        self.X_target = None
        self.f_target = None
        self.X_proto = None
        self.proto_label = None
        self.x_proto_adj = None
        self.variances = None
        self.low_var_idx = None
        self.high_var_idx = None
        self.grid = None
        self.d = None
        self.initial_state = None
        self.initial_x = None
        self.f_proto = None
        self.monotonic_increasing = None

    def prepare_search(self, X_target: np.ndarray, threshold: float = 0.1) -> None:
        """
        Prepare the search configuration for a given target.
        
        Args:
            X_target: Target point to search towards
            threshold: Threshold for feature sensitivity filtering
        """
        from utils import dynamic_feature_filter, generate_grid_with_filter
        
        self.X_target = X_target
        self.f_target = self.f(X_target)

        # Select the closest prototype
        distances = np.linalg.norm(self.prototypes - X_target, axis=1)
        closest_proto_index = np.argmin(distances)
        self.X_proto = self.prototypes[closest_proto_index]
        self.proto_label = self.prototype_labels[closest_proto_index]

        # Adjust prototype for low-sensitivity directions
        self.x_proto_adj, self.variances, self.low_var_idx, self.high_var_idx = \
            dynamic_feature_filter(self.f, self.X_proto, X_target, threshold=threshold)
        
        self.grid = generate_grid_with_filter(
            self.x_proto_adj, self.X_target, self.partitions, self.low_var_idx)

        self.d = len(self.X_proto)
        self.initial_state = tuple([0] * self.d)
        self.initial_x = np.array([self.grid[i][0] for i in range(self.d)])
        self.f_proto = self.f(self.initial_x)
        self.monotonic_increasing = (self.f_target >= self.f_proto)

    def execute_search(self) -> Optional[SearchResult]:
        """Execute the search using the configured strategy"""
        if self.strategy is None:
            raise ValueError("No search strategy configured")
        if self.X_target is None:
            raise ValueError("Search not prepared. Call prepare_search first.")
            
        result = self.strategy.search(self)
        if result is None:
            return None
        return SearchResult(**result) 