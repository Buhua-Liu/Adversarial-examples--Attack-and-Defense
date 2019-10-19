import tensorflow as tf
import numpy as np
from dknn import DkNNModel
from fast_gradient_method import fast_gradient_method

def main():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    test_images = (test_images - 127.5) / 127.5

    #use a holdout of the test data to simulate calibration data for the DkNN
    nb_cali = 750
    train_data = train_images
    cali_data = test_images[:nb_cali]
    y_cali = test_labels[:nb_cali]
    test_data = test_images[nb_cali:]
    y_test = test_labels[nb_cali:]

    #Define callable that returns a dictionary of all activations for a dataset
    def get_activations(data):
        data_activations = {}
        for layer in layers:
            output = model.get_layer(layer).output
            extractor = tf.keras.Model(model.input,output)
            print(output.shape)
            data_activations[layer] = extractor(data)
        return data_activations

    #load the trained model
    model = tf.keras.models.load_model('mnist.h5')

    # Extract representations for the training and calibration data at each layer of interest to the DkNN.
    layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'dense']


    neighbors = 75
    nb_classes = 10
    number_bits =17

    dknn_model = DkNNModel(
            neighbors,
            layers,
            get_activations,
            train_data,
            train_labels,
            nb_classes,
            number_bits = number_bits
    )

    dknn_model.calibrate(cali_data, cali_labels)

    #Generate adversarial examples
    adv = fast_gradient_method(
            model,
            test_data,
            eps = 0.25,
            norm = np.inf
            clip_min = -1.,
            clip_max = 1.,
    )

    #Test the DkNN on clean test data and FGSM test data
    for data_in, fname in zip([test_data, adv], ['test', 'adv'):
        dknn_preds = dknn.fprop_np(data_in)
        print(dknn_preds.shape)
        print(np.mean(np.argmax(dknn_preds, axis=1) == np.argmax(y_test, axis=1)))
        DkNNModel.plot_reliability_diagram(dknn_preds, np.argmax(
                y_test, axis=1), '/tmp/dknn_' + fname + '.pdf')

if __name__ == '__main__':
    main()
