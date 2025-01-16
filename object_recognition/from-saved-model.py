# This will follow the TensorFlow 2 Object Detection API Tutorial
# In order to achieve correct functionality, please set up a conda venv
# According to the instructions on:
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

# imports imports...
import os
# make tensorflow shut up for once
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib
import tensorflow as tf
# Once more trying to make it shut up
tf.get_logger().setLevel('ERROR')

# Enable GPU (for those with CUDA support etc)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

def download_images():
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []
    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename,
                                            origin=base_url + filename,
                                            untar=False)
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))
    return image_paths

IMAGE_PATHS = download_images()

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

# Downloading labels
def download_labels(filename):
    base_url= 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)

# imports for loading the model
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Loading labels for plotting
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
# The code below will load an image, put it through the model and then
# visualize the results, including keypoints

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
# Make matplotlib stop crying
warnings.filterwarnings('ignore')

# Load an image into a numpy array, returns array with shape (img_height, img_width, 3) (3 is RGB)
def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

for image_path in IMAGE_PATHS:
    print('Running inference for {}...'.format(image_path), end='')
    image_np = load_image_into_numpy_array(image_path)

    # Below will be different things you can try to check the detection
    # just uncomment relevant code and see where it fecks up
    #
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert to greyscale
    # image_np = np.title(
    #       np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The inputs needs to be a tensor, so convert it using tf.convert_to_tensor
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch, so add an axis with tf.newaxis (good for expandability)
    input_tensor = input_tensor[tf.newaxis, ...]

    # No clue what commented code below does, gonna experiment later
    # input_tensor = np.expand_dims, image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs will be batches, convert to numpy and take index [0] to remove the batch dimension
    # As we are only interested in the first num detection
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints (essentially the categories in which the object can be put into
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    # VISUAL DEMONSTRATION BROKEN, PRINTING INSTEAD
    # EVERYTHING ELSE WORKS THO SO IGNORE
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print(detections['detection_classes'])
    print(detections['detection_scores'])
    print('Done!')

plt.show()

# PLEASE WORK PLEASE WORK PLEASE WORK PLEASE WORK