import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# Load the MNIST data from Keras datasets API
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Pre-process the data
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5

# Specify parameters
nb_filters = 64
nb_classes = 10
learning_rate=1e-3
batch_size=500
epochs=6


# Define the Keras model
model = tf.keras.Sequential([
    Conv2D(nb_filters, 8, 2, "same", activation='relu', name = 'conv2d', input_shape= (28, 28, 1)),
    Conv2D(nb_filters * 2, 6, 2, "valid", name = 'conv2d_1', activation='relu'),
    Conv2D(nb_filters * 2, 5, 1, "valid", name = 'conv2d_2', activation='relu'),
    Flatten(),
    Dense(nb_classes, name = 'dense', activation='softmax')
])

# Print a summary of the network
model.summary()

# Configure the model for trainging
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, batch_size, epochs)

# Return the loss value & metrics for the model in test mode
model.evaluate(test_images, test_labels)

# Save the model to a single HDF5 file
model.save('mnist.h5')
