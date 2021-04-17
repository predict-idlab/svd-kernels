import tensorflow as tf
from src.layers import SVDDense
import time
import tqdm
import os

class Experiment:
    def __init__(self, model, loss_fn, optimizer, model_name, model_dir):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model_name = model_name
        self.model_dir = model_dir

        self._static_optimizer = None
        self._loss = None
        self._kappa = None

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss_fn(targets, self.model(inputs))
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        save_dir = os.path.join(*[self.model_dir, self.model_name, "model.json"])
        print(f'Saving model to {save_dir}')
        with open(save_dir, "w") as json_file:
            json_file.write(model_json)

    def save_weights(self, epoch: int):
        # serialize weights to HDF5
        save_dir = os.path.join(*[self.model_dir, self.model_name, f'epoch_{epoch}.h5'])
        print(f'saving weights on epoch {epoch} to {save_dir}')
        self.model.save_weights(save_dir)

    def load_model(self):
        # serialize model to JSON
        load_dir = os.path.join(*[self.model_dir, self.model_name, "model.json"])
        print(f'Loading model from {load_dir}')
        json_file = open('models/2layerMLP_SVD/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json, custom_objects={'SVDDense': SVDDense})

    def load_weights(self, epoch: int):
        # serialize weights to HDF5
        load_dir = os.path.join(*[self.model_dir, self.model_name, f'epoch_{epoch}.h5'])
        print(f'Loading weights from epoch {epoch} from {load_dir}')
        self.model.load_weights(save_dir)

    def train(self, train_ds, n_epochs, learning_rate, save: bool = False, save_epoch: int = 5, adagrad: bool = False):
        if save:
            full_dir = os.path.join(self.model_dir, self.model_name)
            if not os.path.exists(full_dir):
                print('Making model directory')
                os.makedirs(full_dir)
            self.save_model()
        # Keep results for plotting
        self._loss = []
        static_optimizer = tf.keras.optimizers.Adagrad() if adagrad else tf.keras.optimizers.SGD()
        optimizer = self.optimizer(learning_rate)
        for epoch in range(n_epochs):
            print(f'Epoch: {epoch + 1}')
            start = time.time()
            # Training loop - using batches of 32
            for idx, (x, y) in enumerate(tqdm.notebook.tqdm(train_ds)):
                # Calculate gradients
                loss_value, grads = self.grad(x, y)
                # create variables
                trainable_vars = self.model.trainable_variables
                grad_vars = list(zip(grads, trainable_vars))
                # optimize seperatly for (U, V) & others
                condition = lambda var: ('U:' in var.name or 'V:' in var.name)
                static_optimizer.apply_gradients(
                  grad_vars[i] for i, var in enumerate(trainable_vars) if condition(var)
                )
                optimizer.apply_gradients(
                  grad_vars[i] for i, var in enumerate(trainable_vars) if not condition(var)
                )
                # End epoch
                self._loss.append(loss_value.numpy())
            if save & (epoch + 1 % save_epoch == 0):
                self.save_weights(epoch + 1)

    def predict(self, test_ds, argmax=False):
        prediction = tf.concat([self.model(x) for x, y in test_ds], axis=0)
        return np.argmax(prediction.numpy(), axis=-1) if argmax else prediction.numpy()

    def evaluate(evaluate_ds, metrics):
        y_pred = self.predict(evaluate_ds) 
        return {
            key: metric(y_true, y_pred)
            for key, metric in metrics.items()
        }