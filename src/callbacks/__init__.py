import tensorflow as tf
from src.callbacks.utils import orthogonality_number, conditioning_number


class DecompositionTracker(tf.keras.callbacks.Callback):
    """Track SVD decomposition conditioning and orthogonality of dense layers."""
    def __init__(self, on_batch: bool = True):
        """Initialize tracker.

        Parameters
        ----------
        on_batch: bool
            Whether to keep track of orthogonality per batch or per epoch
            (default is True)
        """
        super(DecompositionTracker, self).__init__()
        # Bool whether per epoch or per batch
        self.on_batch = on_batch
        # Empty dict
        self.kappa = {}

    def on_train_begin(self, logs=None):
        """Begin tracking at start of training."""
        # Initialize orthogonality for each dense layer
        self.kappa = {
            layer.name: {
                'u': [],
                's': [],
                'v': []
            }
            for layer in self.model.layers if 'dense' in layer.name
        }

    def on_epoch_end(self, epoch, logs=None):
        """Add orthogonality per dense layer at end of epoch"""
        for layer in self.model.layers:
            if 'dense' in layer.name:
                u, s, v = tf.linalg.svd(layer.variables[0])
                self.kappa[layer.name]['u'].append(orthogonality_number(u))
                self.kappa[layer.name]['s'].append(conditioning_number(s))
                self.kappa[layer.name]['v'].append(orthogonality_number(v))

    def on_batch_end(self, batch, logs=None):
        """Add orthogonality per dense layer at end of batch if requested"""
        if not self.on_batch:
            pass
        else:
            for layer in self.model.layers:
                if 'dense' in layer.name:
                    u, s, v = tf.linalg.svd(layer.variables[0])
                    self.kappa[layer.name]['u'].append(orthogonality_number(u))
                    self.kappa[layer.name]['s'].append(conditioning_number(s))
                    self.kappa[layer.name]['v'].append(orthogonality_number(v))


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
        # Empty dict
        self.kappa = {}

    def on_train_begin(self, logs=None):
        """Begin tracking at start of training."""
        # Initialize orthogonality for each SVD layer
        self.kappa = {
            layer.name: {
                'u': [],
                'v': []
            }
            for layer in self.model.layers if 'svd' in layer.name
        }

    def on_epoch_end(self, epoch, logs=None):
        """Add orthogonality per SVD layer at end of epoch"""
        for layer in self.model.layers:
            if 'svd' in layer.name:
                self.kappa[layer.name]['u'].append(orthogonality_number(layer.variables[0]))
                self.kappa[layer.name]['v'].append(orthogonality_number(layer.variables[2]))

    def on_batch_end(self, batch, logs=None):
        """Add orthogonality per SVD layer at end of batch if requested"""
        if not self.on_batch:
            pass
        else:
            for layer in self.model.layers:
                if 'svd' in layer.name:
                    self.kappa[layer.name]['u'].append(orthogonality_number(layer.variables[0]))
                    self.kappa[layer.name]['v'].append(orthogonality_number(layer.variables[2]))


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
        # Empty dict
        self.kappa = {}

    def on_train_begin(self, logs=None):
        """Begin tracking at start of training."""
        # Initialize orthogonality for each SVD layer
        self.kappa = {
            layer.name: []
            for layer in self.model.layers if 'svd' in layer.name
        }

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
