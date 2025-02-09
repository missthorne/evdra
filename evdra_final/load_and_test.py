# This file serves as a test and a study on how to load models
import os
import io
import pprint
import tempfile
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from six import BytesIO
from IPython import display
from urllib.request import urlopen

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

# Exported model dir
export_dir='./exported_model'
# Test data dir
test_data_input_path = './bccd_coco_tfrecords/test-00000-of-00001.tfrecord'

# Trying to import model
imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']

# Needs height and width for sample image
HEIGHT, WIDTH = 256, 256
IMG_SIZE = [HEIGHT, WIDTH, 3]

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

# Creating decoder
tf_ex_decoder = TfExampleDecoder()

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

# Test dataset load
num_of_examples = 5
test_ds = tf.data.TFRecordDataset(
    './bccd_coco_tfrecords/test-00000-of-00001.tfrecord').take(
        num_of_examples)
# show_batch(test_ds, num_of_examples) # might need it?


# Visualizing predictions
input_image_size = (HEIGHT, WIDTH)
plt.figure(figsize=(20, 20))
min_score_thresh = 0.30 # change minimum score to see all bounding boxes confidence

for i, serialized_example in enumerate(test_ds):
    plt.subplot(1, num_of_examples, i+1)
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
plt.savefig('test_predictions.png')