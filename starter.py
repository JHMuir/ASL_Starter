import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging noise

import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns

# Use keras directly (works with both standalone Keras 3 and tf.keras)
import keras
# from keras import layers, models, callbacks, regularizers

print(f"Keras version: {keras.__version__}")

def load_and_preprocess_data(train_path, test_path):
    """
    Load the Sign MNIST dataset and prepare it for training.
    
    The dataset uses labels 0-24, but skips 9 (letter J) because J requires motion.
    This gives us 24 classes total for static hand signs.
    """
    # Load CSV files
    train_df = pandas.read_csv(train_path)
    test_df = pandas.read_csv(test_path)
    
    # Separate labels from pixel data
    y_train = train_df['???'].values
    y_test = test_df['???'].values
    x_train = train_df.drop('???', axis=1).values
    x_test = test_df.drop('???', axis=1).values
    
    # Normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / ???
    x_test = x_test.astype('float32') / ???
    
    # Reshape to (samples, height, width, channels) for CNN input
    x_train = x_train.reshape(-1, ??, ??, 1)
    x_test = x_test.reshape(-1, ??, ??, 1)
    
    # One-hot encode labels
    ??? = sklearn.???.???()
    y_train = ???.fit_transform(y_train)
    y_test = ???.transform(y_test)  # Use transform, not fit_transform!
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Number of classes: {y_train.shape[1]}")
    
    return x_train, y_train, x_test, y_test, ???

if __name__ == "__main__":
    # File paths - adjust these to match your dataset location
    TRAIN_PATH = "dataset/sign_mnist_train.csv"
    TEST_PATH = "dataset/sign_mnist_test.csv"
    
    # Load and preprocess data
    print("Loading dataset...")
    x_train, y_train, x_test, y_test, label_binarizer = load_and_preprocess_data(
        TRAIN_PATH, TEST_PATH
    )