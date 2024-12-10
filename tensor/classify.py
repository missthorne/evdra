# this is yet one more tutorial I am trying out
# idfk why the other one just does not work


# importing tf, keras and helper libraries
# tensorflow will shit itself with warnings if you do not have cuda support
# simply ignore them for now, even if they are annoying.
# another quirk is that pycharm will freak out and say keras is not imported
# the code will work as expected, though
# either deal with it or use another editor

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# will be using fashion MNIST, kinda hello world of ML world
# 60k greyscale images with fashion items
# test set has 10k images
fashion_mnist = tf.keras.datasets.fashion_mnist

# loading data into some vars
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# train_images and train_labels are a training set
# they are tested against a test set, that being test_images and test_labels
# more info: https://www.tensorflow.org/tutorials/keras/classification

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt','Sneaker', 'Bag', 'Ankle boot']

# preprocessing the data
# what does it do? idk but they both have to be preprocessed the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# displaying the first 25 for verification

# plt.figure(figsize=(10,10))
# for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
# plt.show()

# building a layers, it's essentially extracting representation from data
# you kinda chain them together

model = tf.keras.Sequential([
    # change images into two-dimensional layers, 28x28 pixels
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    # this means that every image belongs in one of the 10 classes
    tf.keras.layers.Dense(10)
])

# compiling the model
# optimizer kinda handles how the model is updated and its loss function
# loss function - measures how accurate it is during training. minimize to "steer" the model
# metrics - monitoring training and testing steps
# this here uses accuracy meaning it'll return the fraction of the images that are correctly classified
# throws another CUDA warning, not sure how to get rid of that
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# feeding the model
model.fit(train_images, train_labels, epochs=10)

# setting some stat variables
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\n Test accuracy: ', test_acc)

# converting linear outputs to probabilities
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# making predictions for each image in the test set
predictions = probability_model.predict(test_images)

# to look at first one do
# predictions[0]
# this will print out an array of 10 numbers
# showing probability of the specimen belonging to one of the classes


# this will print the label with the highest confidence/probability value
# np.argmax(predictions[0])

# defining functions to graph a full set of 10 class predictions

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color ='red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# plot the first x test images, predicted label and true label
# correct in blue, incorrect in red

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()