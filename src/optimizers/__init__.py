from typing import *
from src.models.utils import unpack, chi, assembled_gradient

import tqdm
import tensorflow as tf
import numpy as np


class SVDOptimizer:
    """Optimizer function for SVD based architectures"""
    def __init__(self, learning_rate: float, nu: float, beta: float = 0.9):
        """Initialize

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

    def train(self, loss_fn, model, data, epochs: int, metrics: Optional[dict]):
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
        # Initialize metrics & loss
        train_loss = []
        train_metrics = {key: [] for key in metrics.keys()}
        # Initialize momentum
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
                        momentum, names = zip(*[
                            (tf.zeros_like(var), var.name) for name, layer in unpack([model]) for var in
                            layer.trainable_variables])
                    # update momentum
                    momentum = [self.beta * mom + grad for grad, mom in zip(gradients, momentum)]

                # Indices of svd variables
                slices = [slice(idx, idx + 4) for idx, name in enumerate(names) if ('cadense' in name) & ('U' in name)]
                length = range(len(variables))
                svd_indices = [idx for indices in slices for idx in length[indices]]
                # Calculate SVD variables per layer
                for indices in slices:
                    # Get gradients and variables for components
                    u, s, v, w = variables[indices]
                    du, ds, dv, dw = momentum[indices]
                    # Calculate orthogonal update
                    chi_u = chi(u, du, self.nu)
                    chi_v = chi(v, dv, self.nu)
                    u_update = u + chi_u @ u
                    v_update = v + chi_v @ v
                    # Calculate assembled gradient
                    dk = assembled_gradient(u, s, v, du, ds, dv, 10e-8)
                    # Calculate singular value updates
                    psi_u = tf.transpose(u) @ chi_u @ u
                    psi_v = tf.transpose(v) @ chi_v @ v
                    s_matrix = tf.linalg.diag(s)
                    s_update_matrix = psi_u @ s_matrix + (s_matrix + psi_u @ s_matrix) @ tf.transpose(
                        psi_v) - self.learning_rate * (tf.transpose(u_update) @ dk @ v_update)
                    s_update = tf.linalg.diag_part(s_update_matrix)
                    # Update singular values
                    s.assign_add(s_update)
                    # Update orthogonal matrices
                    u.assign_add(chi_u @ u)
                    v.assign_add(chi_v @ v)
                    # Calculate regular update
                    w_update = self.learning_rate * dw
                    # Regular updates
                    w.assign_sub(w_update)

                # Update remainder
                for idx, (dv, v) in enumerate(zip(momentum, model.trainable_variables)):
                    if idx not in svd_indices:
                        v.assign_sub(self.learning_rate * dv)
                # Add batch loss
                train_loss.append(np.mean(loss))
                for key, metric in metrics.items():
                    train_metrics[key].append(metric(predictions, targets))
        return train_loss, train_metrics


class SVDAdamOptimizer:
    """Adam Optimizer function for SVD based architectures"""
    def __init__(self, learning_rate: float, nu: float, beta: float = 0.9, gamma: float = 0.999):
        """Initialize

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

    def train(self, loss_fn, model, data, epochs: int, metrics: Optional[dict]):
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
                slices = [slice(idx, idx + 4) for idx, name in enumerate(names) if ('cadense' in name) & ('U' in name)]
                length = range(len(variables))
                svd_indices = [idx for indices in slices for idx in length[indices]]
                # Calculate SVD variables per layer
                for indices in slices:
                    # Get gradients and variables for components
                    u, s, v, w = variables[indices]
                    du, ds, dv, dw = momentum[indices]
                    lu, ls, lv, lw = velocity[indices]
                    # Calculate adaptive rates on Stieffel manifold
                    left_nu = self.nu
                    right_nu = self.nu
                    # Calculate orthogonal update
                    chi_u = chi(u, du, left_nu)
                    chi_v = chi(v, dv, right_nu)
                    u_update = u + chi_u @ u
                    v_update = v + chi_v @ v
                    # Calculate assembled gradient
                    dk = assembled_gradient(u, s, v, du, ds, dv, 10e-8)
                    # Calculate singular value updates
                    psi_u = tf.transpose(u) @ chi_u @ u
                    psi_v = tf.transpose(v) @ chi_v @ v
                    s_matrix = tf.linalg.diag(s)
                    # Calculate adaptive learning rate
                    ls /= (1 - self.gamma) ** t
                    rate = self.learning_rate * tf.sqrt(ls + 10e-8) ** (-1)
                    # Update singular values
                    s_update_matrix = psi_u @ s_matrix + (s_matrix + psi_u @ s_matrix) @ tf.transpose(
                        psi_v) - rate * (tf.transpose(u_update) @ dk @ v_update)
                    s_update = tf.linalg.diag_part(s_update_matrix)
                    # Update singular values
                    s.assign_add(s_update)
                    # Update orthogonal matrices
                    u.assign_add(chi_u @ u)
                    v.assign_add(chi_v @ v)
                    # Calculate regular update
                    w_update = self.learning_rate * dw
                    # Regular updates
                    w.assign_sub(w_update)

                # Update remainder
                for idx, (lv, dv, v) in enumerate(zip(velocity, momentum, model.trainable_variables)):
                    if idx not in svd_indices:
                        lv /= (1 - self.gamma)**t
                        rate = self.learning_rate * tf.sqrt(lv + 10e-8)**(-1)
                        v.assign_sub(rate * dv)
                # Add batch loss
                train_loss.append(np.mean(loss))
                for key, metric in metrics.items():
                    train_metrics[key].append(metric(predictions, targets))
                # Increment t
                t += 1
        return train_loss, train_metrics
