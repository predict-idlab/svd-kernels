from typing import *

from .utils import unpack, chi, assembled_gradient, update_svd

import tqdm
import tensorflow as tf
import numpy as np


class SVDOptimizer:
    """Optimizer function for SVD based architectures"""
    def __init__(self, learning_rate: float, nu: float, beta: float = 0.9):
        """Initialize optimizer.

        Parameters
        ----------
        learning_rate: float
            Learning rate for optimizer
        nu: float
            Learning rate for cayley transform
        beta: float
            Momentum parameter
        """
        self.learning_rate = learning_rate
        self.nu = nu
        self.beta = beta

    def train(self, loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
              model: tf.keras.Model, epochs: int, train_data: tf.data.Dataset,
              metrics: Optional[dict] = None, validation_data: Optional[tf.data.Dataset] = None) -> List:
        """Train a model with optimizer.

        Parameters
        ----------
        loss_fn: callable
            Loss function to optimize
        model: tf.Model
            Model to train
        epochs: int
            Number of epochs to train
        train_data: tf.Data.Dataset
            Dataset to iterate for training
        validation_data: tf.data.Dataset
            Dataset to iterate for validation
            (default is None)
        metrics: Optional[dict]
            Dictionary containing metrics
            (default is None)
        Returns
        -------
        train_loss, train_metrics
        """
        # Initialize epsilon
        epsilon = 10e-8
        # Initialize training loss
        train_loss = []
        # Initialize training metrics
        if metrics is not None:
            train_metrics = {key: [] for key in metrics.keys()}
            # Validation metrics if validation data is given
            if validation_data is not None:
                validation_metrics = {key: [] for key in metrics.keys()}
        # Iterate epochs
        for epoch in range(epochs):
            # Initialize epoch loss
            epoch_loss = []
            # Iterate over data
            progress_bar = tqdm.tqdm_notebook(train_data)
            for batch, (inputs, targets) in enumerate(progress_bar):
                # Make padding masks
                with tf.GradientTape() as tape:
                    # Predictions and loss w.r.t inputs and targets
                    predictions = model(inputs)
                    loss = loss_fn(targets, predictions)
                    # Set progress bar information
                    progress_bar.set_postfix({'loss': np.mean(epoch_loss), 'epoch': epoch, 'batch': batch})
                    # Gradients
                    variables = model.trainable_variables
                    gradients = tape.gradient(loss, model.trainable_variables)
                    # Remove nans (Optional)
                    nans = [tf.math.is_nan(g) for g in gradients]
                    gradients = [tf.where(nan, 0.0, g) for nan, g in zip(nans, gradients)]
                    # Initialize after call for first batch of first epoch, can reset after each epoch
                    if (epoch == 0) & (batch == 0):
                        momentum, names = zip(*[
                            (tf.zeros_like(var), var.name) for name, layer in unpack([model]) for var in
                            layer.trainable_variables])
                    # update momentum
                    momentum = [self.beta * mom + grad for grad, mom in zip(gradients, momentum)]

                # Indices of svd variables
                slices = [slice(idx, idx + 3) for idx, name in enumerate(names) if ('cadense' in name) & ('U' in name)]
                # Calculate SVD variables per layer
                for indices in slices:
                    # Get gradients and variables for components
                    u, s, v = variables[indices]
                    du, ds, dv = momentum[indices]
                    # Update svd layer
                    du, ds, dv = update_svd(u, s, v, du, ds, dv, self.nu, self.learning_rate, self.nu, epsilon)
                    # Update orthogonal matrices
                    u.assign_add(du)
                    v.assign_add(dv)
                    # Update singular values
                    s.assign_add(ds)
                # Update remainder
                for idx, (dv, v) in enumerate(zip(momentum, model.trainable_variables)):
                    if idx not in [idx for indices in slices for idx in range(len(variables))[indices]]:
                        v.assign_sub(self.learning_rate * dv)
                # Add batch loss
                train_loss.append(np.mean(loss))

            # # Add metrics
            # if metrics is not None:
            #     # Add train metrics
            #     for key, metric in metrics.items():
            #         train_metrics[key].append(metric(predictions, targets))
            #     # Add validation metrics
            #     if validation_data is not None:
            #         progress_bar = tqdm.tqdm_notebook(validation_data)
            #         for batch, (inputs, targets) in enumerate(progress_bar):
            #             predictions = model(inputs)
            #             for key, metric in metrics.items():
            #                 validation_metrics[key].append(metric(predictions, targets))
        # Outputs
        outputs = [model, train_loss, train_metrics] if metrics is not None else [model, train_loss]
        # Return outputs
        return outputs


class SVDAdamOptimizer:
    """Adam Optimizer function for SVD based architectures"""
    def __init__(self, learning_rate: float, nu: float, beta: float = 0.9, gamma: float = 0.999):
        """Initialize optimizer

        Parameters
        ----------
        learning_rate: float
            Learning rate for optimizer
        nu: float
            Learning rate for cayley transform
        beta: float
            Momentum parameter
        gamma: float
            Velocity parameter
        """
        self.learning_rate = learning_rate
        self.nu = nu
        self.beta = beta
        self.gamma = gamma

    def train(self, loss_fn, model, data, epochs: int, metrics: Optional[dict]) -> List:
        """Train a model

        Parameters
        ----------
        loss_fn: callable
            Loss function to optimize
        model: tf.Model
            Model to train
        data: tf.Data.Dataset
            Dataset to iterate
        epochs: int
            Number of epochs to train
        metrics: Optional[dict]
            Dictionary containing training metrics
        Returns
        -------
        train_loss, train_metrics
        """
        # Initialize epsilon
        epsilon = 10e-8
        # Initialize metrics & loss
        train_loss = []
        train_metrics = {key: [] for key in metrics.keys()}
        # Initialize t
        t = 0
        # Iterate epochs
        for epoch in range(epochs):
            # Initialize epoch loss
            epoch_loss = []
            # Iterate over data
            progress_bar = tqdm.tqdm_notebook(data)
            for batch, (inputs, targets) in enumerate(progress_bar):
                # Make padding masks
                with tf.GradientTape() as tape:
                    # Predictions and loss w.r.t inputs and targets
                    predictions = model(inputs)
                    loss = loss_fn(targets, predictions)
                    # Set progress bar information
                    progress_bar.set_postfix({'loss': np.mean(epoch_loss), 'epoch': epoch, 'batch': batch})
                    # Gradients
                    variables = model.trainable_variables
                    gradients = tape.gradient(loss, model.trainable_variables)
                    # Remove nans (Optional)
                    nans = [tf.math.is_nan(g) for g in gradients]
                    gradients = [tf.where(nan, 0.0, g) for nan, g in zip(nans, gradients)]
                    # Initialize after call for first batch of first epoch, can reset after each epoch
                    if (epoch == 0) & (batch == 0):
                        velocity, momentum, names = zip(*[
                            (tf.zeros_like(var), tf.zeros_like(var), var.name)
                            for name, layer in unpack([model]) for var in layer.trainable_variables
                        ])
                    # update momentum
                    momentum = [self.beta * mom + (1 - self.beta) * grad for grad, mom in zip(gradients, momentum)]
                    # update velocity
                    velocity = [self.gamma * vel + (1 - self.gamma) * grad**2 for grad, vel in zip(gradients, velocity)]

                # Indices of svd variables
                slices = [slice(idx, idx + 3) for idx, name in enumerate(names) if ('cadense' in name) & ('U' in name)]
                # Calculate SVD variables per layer
                for indices in slices:
                    # Get gradients and variables for components
                    u, s, v = variables[indices]
                    du, ds, dv = momentum[indices]
                    lu, ls, lv = velocity[indices]
                    # Calculate scale gradients with momentum bias
                    du /= (1 - self.beta) ** t
                    ds /= (1 - self.beta) ** t
                    dv /= (1 - self.beta) ** t
                    # Calculate adaptive learning rates
                    lu /= (1 - self.gamma) ** t
                    ls /= (1 - self.gamma) ** t
                    lv /= (1 - self.gamma) ** t
                    # Scale gradients with adaptive learning rate
                    du /= tf.sqrt(lu + epsilon)
                    ds /= tf.sqrt(ls + epsilon)
                    dv /= tf.sqrt(lv + epsilon)
                    # Update svd layer
                    du, ds, dv = update_svd(u, s, v, du, ds, dv, self.nu, self.learning_rate, self.nu, epsilon)
                    # Update orthogonal matrices
                    u.assign_add(du)
                    v.assign_add(dv)
                    # Update singular values
                    s.assign_add(ds)

                # Update remainder
                for idx, (lv, dv, v) in enumerate(zip(velocity, momentum, model.trainable_variables)):
                    if idx not in [idx for indices in slices for idx in range(len(variables))[indices]]:
                        # Calculate adaptive learning rate
                        lv /= (1 - self.gamma) ** t
                        # Scale gradients with adaptive learning rate
                        dv /= tf.sqrt(lv + epsilon)
                        # Assign adaptive gradients
                        v.assign_sub(self.learning_rate * dv)
                # Increment t
                t += 1
                # Add batch loss
                train_loss.append(np.mean(loss))
                # # Add metrics
                # if metrics is not None:
                #     # Add train metrics
                #     for key, metric in metrics.items():
                #         train_metrics[key].append(metric(predictions, targets))
                #     # Add validation metrics
                #     if validation_data is not None:
                #         progress_bar = tqdm.tqdm_notebook(validation_data)
                #         for batch, (inputs, targets) in enumerate(progress_bar):
                #             predictions = model(inputs)
                #             for key, metric in metrics.items():
                #                 validation_metrics[key].append(metric(predictions, targets))
        # Outputs
        outputs = [model, train_loss, train_metrics] if metrics is not None else [model, train_loss]
        # Return outputs
        return outputs

# WORK IN PROGRESS


class SVDAdam(tf.keras.optimizers.Optimizer):
    """Adam Optimizer function for SVD based architectures with keras optimizer compaitibility"""
    def __init__(self, learning_rate: float, nu: float, beta: float = 0.9, gamma: float = 0.999,
                 name: Optional[str] = None):
        """Initialize optimizer

        Parameters
        ----------
        learning_rate: float
            Learning rate for optimizer
        nu: float
            Learning rate for cayley transform
        beta: float
            Momentum parameter
        gamma: float
            Velocity parameter
        """
        super(SVDAdam, self).__init__(name=name)
        self.learning_rate = learning_rate
        self.nu = nu
        self.beta = beta
        self.gamma = gamma
        self.epsilon = 10e-8
        # Unpack architecture
        self.names, _ = unpack([self.model])
        # Indices of svd variables
        self.slices = [
            slice(idx, idx + 3) for idx, name in enumerate(self.names) if ('cadense' in name) & ('U' in name)]

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        # Calculate SVD variables per layer
        for indices in self.slices:
            # Get gradients and variables for components
            (u, s, v), (du, ds, dv) = zip(*grads_and_vars[indices])
            # Update svd layer
            du, ds, dv = update_svd(u, s, v, du, ds, dv, self.nu, self.learning_rate, self.nu, self.epsilon)
            # Re-add updated gradients to grads & vars
            grads_and_vars[indices] = [(u, du), (s, ds), (v, dv)]
        # Apply gradients regularly
        return super().apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def _aggregate_gradients(self, grads_and_vars):
        return super()._aggregate_gradients(grads_and_vars)

    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
        return super()._distributed_apply(distribution, grads_and_vars, name, apply_state)

    def get_updates(self, loss, params):
        return super().get_updates(loss, params)

    def _set_hyper(self, name, value):
        super()._set_hyper(name, value)

    def _get_hyper(self, name, dtype=None):
        return super()._get_hyper(name, dtype)

    def _create_slots(self, var_list):
        super()._create_slots(var_list)

    def _create_all_weights(self, var_list):
        super()._create_all_weights(var_list)

    def _resource_apply_dense(self, grad, handle, apply_state):
        pass

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices, **kwargs):
        return super()._resource_apply_sparse_duplicate_indices(grad, handle, indices, **kwargs)

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        pass

    def _resource_scatter_add(self, x, i, v):
        return super()._resource_scatter_add(x, i, v)

    def _resource_scatter_update(self, x, i, v):
        return super()._resource_scatter_update(x, i, v)
