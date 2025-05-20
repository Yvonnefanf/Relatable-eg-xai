import numpy as np
from typing import Callable, List, Optional, Dict, Any

class GradientBasedTrajectoryGenerator:
    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        initial_x: np.ndarray,
        target_x: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        lr: float = 0.01,
        max_steps: int = 100,
        tol: float = 1e-6,
        record_f: bool = True,
        record_loss: bool = True,
    ):
        """
        Args:
            f: The function to evaluate along the trajectory.
            initial_x: Starting point (np.ndarray).
            target_x: Target point (np.ndarray).
            loss_fn: Loss function to minimize (default: squared distance to target).
            lr: Learning rate for gradient descent.
            max_steps: Maximum number of optimization steps.
            tol: Tolerance for stopping criterion.
            record_f: Whether to record f(x) along the trajectory.
            record_loss: Whether to record loss along the trajectory.
        """
        self.f = f
        self.initial_x = np.array(initial_x, dtype=float)
        self.target_x = np.array(target_x, dtype=float)
        self.loss_fn = loss_fn if loss_fn is not None else self.default_loss
        self.lr = lr
        self.max_steps = max_steps
        self.tol = tol
        self.record_f = record_f
        self.record_loss = record_loss

    def default_loss(self, x: np.ndarray, target: np.ndarray) -> float:
        return np.sum((x - target) ** 2)

    def compute_loss_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function with respect to x using finite differences.
        """
        eps = 1e-8
        grad = np.zeros_like(x)
        loss0 = self.loss_fn(x, self.target_x)
        for i in range(len(x)):
            x_eps = np.array(x, dtype=float)
            x_eps[i] += eps
            grad[i] = (self.loss_fn(x_eps, self.target_x) - loss0) / eps
        return grad

    def generate(self) -> Dict[str, Any]:
        x = np.array(self.initial_x, dtype=float)
        trajectory = [x.copy()]
        f_values = [self.f(x)] if self.record_f else []
        loss_values = [self.loss_fn(x, self.target_x)] if self.record_loss else []

        for step in range(self.max_steps):
            grad = self.compute_loss_grad(x)
            x_new = x - self.lr * grad
            loss = self.loss_fn(x_new, self.target_x)
            if self.record_f:
                f_values.append(self.f(x_new))
            if self.record_loss:
                loss_values.append(loss)
            trajectory.append(x_new.copy())
            if np.linalg.norm(x_new - x) < self.tol or loss < self.tol:
                break
            x = x_new

        return {
            'trajectory': trajectory,
            'f_values': f_values,
            'loss_values': loss_values,
        }

# Example usage (to be removed or commented out in production):
if __name__ == "__main__":
    def f(x):
        return np.sum(x ** 2)
    initial_x = np.array([5.0, 5.0])
    target_x = np.array([0.0, 0.0])
    generator = GradientBasedTrajectoryGenerator(f, initial_x, target_x)
    result = generator.generate()
    print("Trajectory:", result['trajectory'])
    print("f values:", result['f_values'])
    print("Loss values:", result['loss_values']) 