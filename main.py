import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5

nb_filters = 64
nb_classes = 10

model = Sequential([
    Conv2D(nb_filters, 8, 2, "same", activation='relu', input_shape= (28, 28, 1)),
    Conv2D(nb_filters * 2, 6, 2, "valid", activation='relu'),
    Conv2D(nb_filters * 2, 5, 1, "valid", activation='relu'),
    Flatten(),
    Dense(nb_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=500, epochs=6)

model.evaluate(test_images, test_labels)
