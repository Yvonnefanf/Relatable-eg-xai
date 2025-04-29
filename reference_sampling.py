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

    def sampling_by_ref_val(self, instance: np.ndarray, ref_val: float, f: callable) -> np.ndarray:
        """
        Samples a single reference instance representing the best trade-off 
        between similarity, typicality, and prediction accuracy.

        Args:
            instance: The input instance to find similar references for.
            ref_val: The reference value associated with the input instance (currently unused in selection).
            f: The prediction function (model.predict or similar).

        Returns:
            The single reference instance from self.X that offers the best trade-off.
        """
        n_samples = self.X.shape[0]
        if n_samples == 0:
            raise ValueError("Dataset X is empty.")

        # 1. Calculate Metrics
        # Similarity distance
        distances = np.linalg.norm(self.X - instance, axis=1)
        
        # Typicality distance (distance to median)
        median_X = np.median(self.X, axis=0)
        typicality_distances = np.linalg.norm(self.X - median_X, axis=1)

        # Prediction error
        # Ensure f is called correctly, assuming it takes one instance
        predictions = np.array([f(x_i) for x_i in self.X])
        pred_errors = np.abs(predictions - self.y)

        # Handle cases where all values are the same to avoid division by zero
        def normalize(values):
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val == min_val:
                return np.zeros_like(values)
            return (values - min_val) / (max_val - min_val)

        # 2. Normalize Metrics (0 = best, 1 = worst)
        norm_dist = normalize(distances)
        norm_typ = normalize(typicality_distances)
        norm_err = normalize(pred_errors)

        # 3. Combine Scores (Equal Weights)
        # Lower score is better
        combined_scores = (norm_dist + norm_typ + norm_err) / 3.0 

        # 4. Find Best Trade-off Instance
        best_tradeoff_idx = np.argmin(combined_scores)
        
        return self.X[best_tradeoff_idx]

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

#     best_reference = sampler.sampling_by_ref_val(test_instance, test_ref_val, model_predict)
    
#     print("Input Instance:", test_instance)
#     print("Reference Value:", test_ref_val)
#     print("\n--- Best Trade-off Reference ---")
#     print(best_reference)