import tensorflow as tf
from src.callbacks.utils import orthogonality_number, conditioning_number


class OrthogonalityTracker(tf.keras.callbacks.Callback):
    """Track orthogonality of singular value decomposed layers."""
    def __init__(self, on_batch: bool = True):
        """Initialize tracker.

        Parameters
        ----------
        on_batch: bool
            Whether to keep track of orthogonality per batch or per epoch
            (default is True)
        """
        super(OrthogonalityTracker, self).__init__()
        # Bool whether per epoch or per batch
        self.on_batch = on_batch
        # Initialize orthogonality for each SVD layer
        self.kappa = {
            layer.name: {
                'u': None,
                'v': None
            }
            for layer in self.model.layers if 'svd' in layer.name
        }

    def on_train_begin(self, logs=None):
        """Begin tracking at start of training."""
        # Initialize dictionary with initial values
        for layer in self.model.layers:
            if 'svd' in layer.name:
                self.kappa[layer.name]['u'] = 0
                self.kappa[layer.name]['u'] = 0

    def on_epoch_end(self, epoch, logs=None):
        """Add orthogonality per SVD layer at end of epoch"""
        for layer in self.model.layers:
            if 'svd' in layer.name:
                self.kappa[layer.name]['u'].append(orthogonality_number(layer.variables[0]))
                self.kappa[layer.name]['u'].append(orthogonality_number(layer.variables[2]))

    def on_batch_end(self, batch, logs=None):
        """Add orthogonality per SVD layer at end of batch if requested"""
        if not self.on_batch:
            pass
        else:
            for layer in self.model.layers:
                if 'svd' in layer.name:
                    self.kappas[layer.name]['u'].append(orthogonality_number(layer.variables[0]))
                    self.kappas[layer.name]['v'].append(orthogonality_number(layer.variables[2]))


class ConditioningTracker(tf.keras.callbacks.Callback):
    """Track conditioning of svd layers"""
    def __init__(self, on_batch: bool = True):
        """Initialize tracker.

        Parameters
        ----------
        on_batch: bool
            Whether to keep track of orthogonality per batch or per epoch
            (default is True)
        """
        super(ConditioningTracker, self).__init__()
        # Bool whether per epoch or per batch
        self.on_batch = on_batch
        # Initialize conditioning number for each SVD layer
        self.kappa = {
            layer.name: None
            for layer in self.model.layers if 'svd' in layer.name
        }

    def on_train_begin(self, logs=None):
        """Begin tracking at start of training."""
        # Initialize dictionary with initial values
        for layer in self.model.layers:
            if 'svd' in layer.name:
                self.kappa[layer.name] = [conditioning_number(layer.variables[1])]

    def on_epoch_end(self, epoch, logs=None):
        """Add conditioning number per SVD layer at end of epoch"""
        for layer in self.model.layers:
            if 'svd' in layer.name:
                self.kappa[layer.name].append(conditioning_number(layer.variables[1]))

    def on_batch_end(self, batch, logs=None):
        """Add conditioning number per SVD layer at end of batch if requested"""
        if not self.on_batch:
            pass
        else:
            for layer in self.model.layers:
                if 'svd' in layer.name:
                    self.kappa[layer.name].append(conditioning_number(layer.variables[1]))
