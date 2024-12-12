
# This here file will just serve as a training area to check
# How under and overfilling affects models and prediction accuracy
# This will be most useful while training EVDRA
# BTW THE HIGGS SET TAKES LIKE 5 HOURS TO DOWNLOAD SO SETTING THIS ASIDE

# imports, remember tensor shits itself and pycharm does not see keras

import tensorflow as tf
# tensor.keras freaks the feck out so just do keras
from keras import layers
from keras import regularizers
from keras.src.ops import BinaryCrossentropy

#checking ver
print(tf.__version__)

# might error out, do
# pip install git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# higgs dataset, 11 MILLION examples with 28 features
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28

# tf.data.experimental.CsvDataset reads from gz archives without intermediate steps
ds = tf.data.experimenta.CvsDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

# returns a list of scalars for each record, repacks it into (feature_vector,label) pair
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:],1)
    return features, label

# batch, apply pack_row to each batch, then split into individual records
packed_ds = ds.batch(10000).map(pack_row).unbatch()

# not perfect normalization but this is just a demonstration
for features,label in packed_ds.batch(10000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins = 101)

# just to keep it short, use first 1k for validation, next 10k for training innit
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN/BATCH_SIZE

# setting cache to make sure loader does not shit itself and try to read whole set every time
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

# train_ds returns individual examples
# use Dataset.batch to create caches of a certain size, and Dataset.shuffle/repeat on the training set
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# you lowkey have to experiment to find this shite out
# start with relatively few layers and increase until you see
# diminishing returns in validation loss

# training procedure
# many models work better if you gradually reduce learning rate during training


# this sets InverseTimeDecay to hyperbolically decrease learning rate to 1/2 of the base rate at 1000 epochs
# 1/3 at 2000 epochs and so on
lr_schedule = tf.keras.optimizers.chedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


step = np.lindspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')

# setting up callbacks as every demonstration will use the same config
# use tfdocs.EpochDots to reduce noise
# it prints . for each epoch and full set every 100th
def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacs.TensorBoard(logdir/name),
    ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    # I HATE INDENTATION FOCUSED SCOPE IT FUCKING SUCKS GIMME BACK MY end OR {}
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.metrics.BinaryCrossentropy(
                          from_logits=True, name='binary_crossentropy'),
                      'accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = max_epochs,
        validation_data = validate_ds,
        callbacks = get_callbacks(name),
        verbose = 0)
    return history


# compiling and analyzing a tiny model
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std = 10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
