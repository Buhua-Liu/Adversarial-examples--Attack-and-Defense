import tensorflow as tf
import numpy as np
import os
import time
from dknn import DkNNModel, plot_reliability_diagram
from attack.fgsm import FGSM


def main():
    # Loading and pre-processing data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    test_images = (test_images - 127.5) / 127.5

    # Use a holdout of the test data to simulate calibration data for the DkNN
    nb_cali = 750
    train_data = train_images
    y_train = train_labels
    cali_data = test_images[:nb_cali]
    y_cali = test_labels[:nb_cali]
    test_data = test_images[nb_cali:]
    y_test = test_labels[nb_cali:]

    #Define callable that returns a dictionary of all activations for a dataset
    flatten = tf.keras.layers.Flatten()
    def get_activations(data):
        data_activations = {}
        for layer in layers:
            output = model.get_layer(layer).output
            extractor = tf.keras.Model(model.input, output)
            data_activations[layer] = flatten(extractor(data)).numpy()
        return data_activations

    # Load the trained model
    model = tf.keras.models.load_model('mnist.h5')
    _,acc = model.evaluate(test_images, test_labels, verbose=0)
    print('The accuracy of the trained model on test data is:', acc)

    # Extract representations for the training and calibration data at each layer of interest to the DkNN.
    layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'dense']
    neighbors = 75
    nb_classes = 10
    number_bits =17

    #  Instantiate the DkNNModel
    dknn_model = DkNNModel(
            neighbors,
            layers,
            get_activations,
            train_data,
            y_train,
            nb_classes,
            number_bits = number_bits
    )

    # Calibrate the credibility metric
    dknn_model.calibrate(cali_data, y_cali)

    # Generate adversarial examples
    print('================================================================')
    print('Generating adversarial examples with FGSM...')
    start = time.time()
    adv = FGSM(
            model,
            test_data,
            eps = 0.25,
            norm = np.inf,
            clip_min = -1.,
            clip_max = 1.,
    )
    end = time.time()
    print('Generation completed! Time cost:', end-start, 's.')

    # Test the original DNN and the corresponding DkNN
    #   on clean test data and adversarial test data
    for data_in, fname in zip([test_data, adv], ['clean', 'adversarial']):
        print('================================================================')
        print('Testing the DNN and DkNN on {} data...'.format(fname))
        start = time.time()
        preds = model(data_in).numpy()
        print('For DNN, accuracy on', fname, 'data:',
                np.mean(np.argmax(preds, axis=1) == y_test))
        dknn_preds = dknn_model.fprop_np(data_in)
        print('For DkNN, accuracy on', fname, 'data:',
                np.mean(np.argmax(dknn_preds, axis=1) == y_test))
        if not os.path.exists('diagram'):
            os.mkdir('diagram')
        plot_reliability_diagram(dknn_preds,
                y_test, 'diagram/dknn_' + fname + '.png')
        plot_reliability_diagram(preds,
                y_test, 'diagram/dnn_' + fname + '.png')
        end = time.time()
        print('Test on {} completed! Time cost: {}s'.format(fname, end-start))

if __name__ == '__main__':
    main()
