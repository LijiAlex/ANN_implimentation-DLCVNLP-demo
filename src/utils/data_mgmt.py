import tensorflow as tf

X_test, y_test = None, None

def get_data(validation_datasize, logger):
    global X_test
    global y_test
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    logger.info("MNIST data download")
    # normalizing and redusing the pixel value to be between 0-1
    X_val, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_val, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]
    X_test = X_test / 255.
    logger.info("Create validation data, training data and test data")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_test_data():
    return (X_test, y_test)