# This document will use the following tutorial
# https://www.tensorflow.org/tfmodels/vision/object_detection
#
# I pray to every god. Please let this go smoothly.
#
# As per usual, this is intended for ues with an Anaconda environment.
# Upon completion, the relevant yml file shall be included in the repo

# imports and sacrificing virtual virgins to the gods of AI
import os
import io
import pprint
import tempfile
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from six import BytesIO
from IPython import display
from urllib.request import urlopen

# tensorflow imports and praying for my liver as completing this will require
# a steady diet of caffeine and alcohol
import orbit
import tensorflow_models as tfm

from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib
from official.vision.ops.preprocess_ops import normalize_image
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder

pp = pprint.PrettyPrinter(indent=4) # setting some pretty printer to please the higher beings
print(tf.__version__) # check the version innit


# preparing the custom dataset
# RUN THIS IN CLI

# TRAIN_DATA_DIR='./BCC.v1-bccd.coco/train'
# TRAIN_ANNOTATION_FILE_DIR='./BCC.v1-bccd.coco/train/_annotations.coco.json'
# OUTPUT_TFRECORD_TRAIN='./bccd_coco_tfrecords/train'

# What needs to be provided
    # 1. image_dir: where the images are present
    # 2. object_annotations_file: where the annotations are listed IN JSON
    # 3 output_file_prefix: where to write output converted TFRecords files

# python -m official.vision.data.crate_coco_tf_record --logtostderr \
# --image_dir=${TRAIN_DATA_DIR} \
# --object_annotations_file=${TRAIN_ANNOTATION_FILE_DIR} \
# --output_file_prefix=$OUTPUT_TFRECORD_TRAIN \
# --num_shards=1

# Configure Retinanet for custom dataset

train_data_input_path = './bccd_coco_tfrecords/train-00000-of-00001.tfrecord'
valid_data_input_path = '/.bccd_coco_tfrecords/valid-00000-of-00001.tfrecord'
test_data_input_path = './bccd_coco_tfrecords/test-00000-of-00001.tfrecord'
model_dir='./trained_model'
export_dir='./exported_model'

exp_config = exp_factory.get_exp_config('retinanet_resnetfpn_coco')

# adjusting stuff so it works with custom dataset, here BCCD
batch_size = 8
num_classes = 3

HEIGHT, WIDTH = 256, 256
IMG_SIZE = [HEIGHT, WIDTH, 3]

# Backbone config
exp_config.task.freeze_backbone = False
exp_config.task.annotation_file = ''

# Model config
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.num_classes = num_classes + 1
exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

# Training data config
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.train_data.parser.aug_scale_max = 1.0
exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation data config
exp_config.task.validation_data.input_path = valid_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = batch_size

# Adjusting for GPU training if any of youse wanna do this at home (why?)
logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

if 'GPU' in ''.join(logical_device_names):
    print('This may be broken') # when even the guide says so it's gg please help me
    device='GPU'
elif 'TPU' in ''.join(logical_device_names):
    print('This may also be broken') # help
    device='TPU'
else:
    print('Running on CPU, good luck, this will take a sec')
    device='CPU'

train_steps=1000
exp_config.trainer.steps_per_loop = 100 # steps_per_loop = num_of_training_examples // train_batch_size

exp_config.trainer.summary_interval = 100
exp_config.trainer.checkpoint_interval = 100
exp_config.trainer.validation_interval = 100
exp_config.trainer.validation_steps = 100 # validation_steps = num_of_validation_examples // eval_batch_size
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps # no clue what this means
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

pp.pprint(exp_config.as_dict())
display.Javascript('google.colab.output.setIframeHeight("500px");')


