# THIS IS ANOTHER TUTORIAL BY TENSORFLOW
# PLEASE WORK THIS TIME
# Edit: it worked this time
#
# This document will essentially serve as a way to learn how to save
# and import various models in HDF5 format. This is imperative to our
# success.
#
# If a comment has no space before the token, assume that it's code commented out for demo
#
# Remember that everything I said about tensorflow errors and imports still applies
#  - Ruby

# best to import keras from my experience
import os
import glob
import tensorflow as tf
import keras

print(tf.version.VERSION)

# Getting MNIST again, gonna run it through 1000 examples for demo
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0

# Building a sequential model for demo
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

# Creating a basic instance
model = create_model()

# Display the model's architecture
model.summary()

# You can use a model without having to retrain it, just save checkpoints
# use tf.keras.callbacks.ModelCheckpoint for that to save during and at the end
# NOTE: The tutorial might have the file extension ".cpkt"
# KERAS NEEDS ".weights.h5" TO WORK PROPERLY
checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
# Might make the model shit itself about saving the state of the optimizer
# Just ignore it, tensorflow things.
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback]) # Pass callback to training

os.listdir(checkpoint_dir)

# As long as two models share the same architecture you can share weights between them

# Rebuilding a fresh model, untrained so ~10% acc
model = create_model()

# Evaluate the new model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Load the weights
model.load_weights(checkpoint_path)

# Re-evaluate after loading weights
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored mode, accuracy: {:5.2f}%".format(100*acc)) # Way better

# Different callback options:

# Include epochs in the file name (uses 'str.format')
# CREATE FOLDER MANUALLY
checkpoint_path = "training_2/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Calculate batches per epoch
import math
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)   # round to nearest whole int

# Create a callback that saves every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*n_batches)

# Create a new model instance
model = create_model()

# Save the weights using 'checkpoints_path' format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with new callback
model.fit(train_images,
          train_labels,
          epochs=50,
          batch_size=batch_size,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# Review the checkpoints and choose latest
print(os.listdir(checkpoint_dir))

# NOTE: TensorFlow format saves only the 5 most recent checkpoints
# NOTE: Using a different library than specified for compatibility reasons (plus globglob sounsd funny)
latest = max(glob.glob("training_2/*.h5"))


# Testing (resetting the model and loading from latest)
model = create_model()

# Load
model.load_weights(latest)

# Re-evaluate to check
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# Here we shall save entire models .keras format
# This allows to skip creating models and loading checkpoints every time
#
# Keras v3 .keras format
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model
model.save('my_model.keras')

# Load saved model from .keras zip
new_model = tf.keras.models.load_model('my_model.keras')

# Show architecture
new_model.summary()

# Evaluate restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100* acc))

print(new_model.predict(test_images).shape)