# SVD-kernels
This repository contains the code for the paper: <br>
"Parameter-efficient neural networks with singular value decomposed kernels"

# Code
All the code is written in the newest stable tensorflow version v2.5.0. <br>
The code is designed with to be compatible with the keras functional, sequantial & model API.

# Project tree
* Src
  * Optimizers
  * Layers
  * Models
  * Callbacks
  * Initializers
* Notebooks

# Example
The following is a sample for code usage.

```
from src.layers import SVDDense
from src.optimizrs import SVDAdam

# Create a dataset
data = tf.data.Dataset.from_generator(...)
# Create a model
inputs = tf.keras.layers.Inputs(...)
hidden = SVDDense(...)(inputs)
... # Add more complicated architecture
outputs = tf.keras.layers.Dense(...)(hidden)
# Make model
model = tf.keras.Models(outputs=outputs, inputs=inputs)
# Create optimizer --> Needs a model to be created
optimizer = SVDAdam(model, ...)
# Create loss object
loss_fn = tf.keras.losses.MeanSquaredError()
# Compile model
Model.compile(optimizer, loss_fn, ...)
# train model
model.fit(data, ...)
```

More detailed examples, corresponding to the experiment section in the paper, can be found in the notebooks directory.
