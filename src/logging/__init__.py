import tensorflow as tf


class CustomLogger(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(CustomLogger, self).__init__()
        self.name = name

    def set_params(self, params):
        super().set_params(params)

    def set_model(self, model):
        super().set_model(model)

    def on_batch_end(self, batch, logs=None):
        return super().on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        return super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        return super().on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        return super().on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        return super().on_train_batch_end(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        return super().on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        return super().on_test_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        return super().on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        return super().on_predict_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        return super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        return super().on_train_end(logs)

    def on_test_begin(self, logs=None):
        return super().on_test_begin(logs)

    def on_test_end(self, logs=None):
        return super().on_test_end(logs)

    def on_predict_begin(self, logs=None):
        return super().on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        return super().on_predict_end(logs)

    def on_batch_begin(self, batch, logs=None):
        return super().on_batch_begin(logs)
