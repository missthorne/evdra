# This document will use the following tutorial
# https://www.tensorflow.org/tfmodels/vision/object_detection
#
# I pray to every god. Please let this go smoothly.
#
# As per usual, this is intended for use with an Anaconda environment.
# Upon completion, the relevant yml file shall be included in the repo
#
# imports and sacrificing virtual virgins to the gods of AI
import os
import io
import pprint
import tempfile
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("QtAgg")
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

# Configure RetinaNet for custom dataset

train_data_input_path = './bccd_coco_tfrecords/train-00000-of-00001.tfrecord'
valid_data_input_path = './bccd_coco_tfrecords/valid-00000-of-00001.tfrecord'
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

if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

if 'GPU' in ''.join(logical_device_names):
    distribution_strategy = tf.distribute.MirroredStrategy()
elif 'TPU' in ''.join(logical_device_names):
    tf.tpu.experimental.initialize_tpu_system()
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
    distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    print('Grab some popcorn, pop in the extended version of The Hobbit and relax, this will take a while')
    distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

print('Starting model training')
# panic starts herec

# Task object handles building the dataset
with distribution_strategy.scope():
    task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)

for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
    print()
    print(f'images.shape: {str(images.shape):16} images.dtype: {images.dtype!r}')
    print(f'labels.keys: {labels.keys()}')

# Creating index with different object that can be detected (categories)
category_index={
    1: {
        'id' : 1,
        'name' : 'Platelets'
    },
    2: {
        'id' : 2,
        'name' : 'RBC'
    },
    3: {
        'id' : 3,
        'name' : 'WBC'
    }
}

tf_ex_decoder = TfExampleDecoder()

def show_batch(raw_records, num_of_examples):
  plt.figure(figsize=(20, 20))
  use_normalized_coordinates=True
  min_score_thresh = 0.30
  for i, serialized_example in enumerate(raw_records):
    plt.subplot(1, 3, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = decoded_tensors['image'].numpy().astype('uint8')
    scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        decoded_tensors['groundtruth_boxes'].numpy(),
        decoded_tensors['groundtruth_classes'].numpy().astype('int'),
        scores,
        category_index=category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Image-{i+1}')
  # plt.show()
  # PLT.SH OW SHITS ITSELF EVENTHOUGH I EXPLICITLY CALL FOR QT
  # I HATE PYTHON I HATE PYTHON I HATE PYTHON I HATE PYTHON I HATE PYTHON I HATE PYTHON I HATE PYTHON I HATE PYTHON I HATE PYTHON
  plt.savefig('batch.png')


# Creating the two components of the bounding box
buffer_size = 20
num_of_examples = 3

raw_records = tf.data.TFRecordDataset(
    exp_config.task.train_data.input_path).shuffle(
        buffer_size=buffer_size).take(num_of_examples)
show_batch(raw_records, num_of_examples)

model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True)

# Exporting model directly as I am realising this will be hell
export_saved_model_lib.export_inference_graph(
    input_type='image_tensor',
    batch_size=1,
    input_image_size=[HEIGHT, WIDTH],
    params=exp_config,
    checkpoint_path=tf.train.latest_checkpoint(model_dir),
    export_dir=export_dir)


# inference
def load_image_into_numpy_array(path):
    """ Load an image into a numpy array
    
    It feeds it into the tensorflow graph
    By convention we put it into a numpy array with shape
    (height, width, channels) where channels=3 for RGB.
    
    Args:
        path: the file path to the image
        
    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if(path.startswith('http')):
        response=urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8
    )

def build_inputs_for_object_detection(image, input_image_size):
    # Builds Object Detection model inputs for serving
    image, _ = resize_and_crop_image(
        image,
        input_image_size,
        padded_size=input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0
    )
    return image
num_of_examples = 3

test_ds = tf.data.TFRecordDataset(
    './bccd_coco_tfrecords/test-00000-of-00001.tfrecord').take(
        num_of_examples)
show_batch(test_ds, num_of_examples)

# Trying to import model
imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']

# Visualizing predictions
input_image_size = (HEIGHT, WIDTH)
plt.figure(figsize=(20, 20))
min_score_thresh = 0.30 # change minimum score to see all bounding boxes confidence

for i, serialized_example in enumerate(test_ds):
    plt.subplot(1, 3, i+1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = build_inputs_for_object_detection(decoded_tensors['image'], input_image_size)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype = tf.uint8)
    image_np = image[0].numpy()
    result = model_fn(image)
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        result['detection_boxes'][0].numpy(),
        result['detection_classes'][0].numpy().astype(int),
        result['detection_scores'][0].numpy(),
        category_index=category_index,
        use_normalized_coordinates=False,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4
    )
    plt.imshow(image_np)
    plt.axis('off')
plt.savefig('predictions.png')