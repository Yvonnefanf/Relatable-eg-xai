import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class References:
    """Data class to hold prototype information"""
    data: np.ndarray
    
class ReferenceSampler:
    """Class to sample references from a dataset"""
    def __init__(self, X: np.ndarray, y: np.ndarray, categorical_indices: Optional[List[int]] = None):
        """Initialize the sampler with the dataset and categorical indices"""
        self.X = X
        self.y = y
        self.categorical_indices = categorical_indices

    def sampling_by_ref_val(self, instance: np.ndarray, ref_val: float, f: callable) -> Dict[str, np.ndarray]:
        """
        Samples references based on similarity, typicality, and prediction accuracy.

        Args:
            instance: The input instance to find similar references for.
            ref_val: The reference value associated with the input instance (currently unused in selection).
            f: The prediction function (model.predict or similar).

        Returns:
            A dictionary containing the 'similar', 'typical', and 'accurate' reference instances.
        """
        
        # 1. Most Similar Instance
        distances = np.linalg.norm(self.X - instance, axis=1)
        similar_idx = np.argmin(distances)
        x_similar = self.X[similar_idx]

        # 2. Most Typical Instance
        # Assuming typicality means closest to the median feature values
        median_X = np.median(self.X, axis=0)
        typicality_distances = np.linalg.norm(self.X - median_X, axis=1)
        typical_idx = np.argmin(typicality_distances)
        x_typical = self.X[typical_idx]

        # 3. Most Prediction Accurate Instance
        # Find instance x_i where f(x_i) is closest to y_i
        # Assuming f takes one instance at a time
        pred_errors = np.array([abs(f(x_i) - self.y[i]) for i, x_i in enumerate(self.X)])
        accurate_idx = np.argmin(pred_errors)
        x_accurate = self.X[accurate_idx]

        return {
            'similar': x_similar,
            'typical': x_typical,
            'accurate': x_accurate
        }

# Example Usage (requires data and a prediction function 'model_predict')
# if __name__ == '__main__':
#     # Dummy data
#     X_data = np.random.rand(100, 5)
#     y_data = np.random.rand(100) * 10
#     def model_predict(x): # Dummy prediction function
#         return np.sum(x) * 2 

#     sampler = ReferenceSampler(X_data, y_data)
    
#     test_instance = np.random.rand(5)
#     test_ref_val = model_predict(test_instance) # Example reference value

#     references = sampler.sampling_by_ref_val(test_instance, test_ref_val, model_predict)
    
#     print("Input Instance:", test_instance)
#     print("Reference Value:", test_ref_val)
#     print("\n--- Found References ---")
#     print("Most Similar:", references['similar'])
#     print("Most Typical:", references['typical'])
#     print("Most Accurate Prediction:", references['accurate'])