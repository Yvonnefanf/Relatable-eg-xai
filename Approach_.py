import numpy as np
import heapq
from collections import deque
from eval import linear_approximation_error
from utils import dynamic_feature_filter, generate_grid_with_filter

class SearchPathBase:
    def __init__(self, f, prototypes, prototype_labels, partitions=2, max_steps=5):
        self.f = f
        self.prototypes = prototypes
        self.prototype_labels = prototype_labels
        self.partitions = partitions
        self.max_steps = max_steps

    def prepare_search(self, X_target, threshold=0.1):
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
        
        self.grid = generate_grid_with_filter(self.x_proto_adj, self.X_target, self.partitions, self.low_var_idx)

        self.d = len(self.X_proto)
        self.initial_state = tuple([0] * self.d)
        self.initial_x = np.array([self.grid[i][0] for i in range(self.d)])
        self.f_proto = self.f(self.initial_x)
        self.monotonic_increasing = (self.f_target >= self.f_proto)

    def search(self):
        raise NotImplementedError("Subclasses should implement this method.")
