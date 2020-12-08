import numpy as np
import pandas as pd


def load_training_data(path='data/'):
    # Read into pandas dataframes
    df_train_eeg1 = pd.read_csv(path + 'training_eeg1.csv')
    df_train_eeg2 = pd.read_csv(path + 'training_eeg2.csv')
    df_train_emg = pd.read_csv(path + 'training_emg.csv')

    # Read directly to numpy array
    y_train = pd.read_csv(path + 'training_labels.csv').y.to_numpy()
    y_train -= 1

    # Convert to numpy arrays, stripping the ID column
    x_train_eeg1 = df_train_eeg1.filter(regex='x.*').to_numpy()
    x_train_eeg2 = df_train_eeg2.filter(regex='x.*').to_numpy()
    x_train_emg = df_train_emg.filter(regex='x.*').to_numpy()

    # Scale each dataset to lie between 0 and 1
    x_train_eeg1 = (x_train_eeg1 - x_train_eeg1.min()) / (x_train_eeg1.max() - x_train_eeg1.min())
    x_train_eeg2 = (x_train_eeg2 - x_train_eeg2.min()) / (x_train_eeg2.max() - x_train_eeg2.min())
    x_train_emg = (x_train_emg - x_train_emg.min()) / (x_train_emg.max() - x_train_emg.min())

    # Combine into single array
    # Reshape data for LSTM
    x_train_eeg1 = x_train_eeg1.reshape(x_train_eeg1.shape[0], x_train_eeg1.shape[1], 1)
    x_train_eeg2 = x_train_eeg2.reshape(x_train_eeg2.shape[0], x_train_eeg2.shape[1], 1)
    x_train_emg = x_train_emg.reshape(x_train_emg.shape[0], x_train_emg.shape[1], 1)

    # Combine into single array
    x_train = np.concatenate((x_train_eeg1, x_train_eeg2, x_train_emg), axis=2)

    return x_train, y_train


def load_testing_data(path='data/'):
    # Read into pandas dataframes
    df_test_eeg1 = pd.read_csv(path + 'testing_eeg1.csv')
    df_test_eeg2 = pd.read_csv(path + 'testing_eeg2.csv')
    df_test_emg = pd.read_csv(path + 'testing_emg.csv')

    # Convert to numpy arrays, stripping the ID column
    x_test_eeg1 = df_test_eeg1.filter(regex='x.*').to_numpy()
    x_test_eeg2 = df_test_eeg2.filter(regex='x.*').to_numpy()
    x_test_emg = df_test_emg.filter(regex='x.*').to_numpy()

    # Scale each dataset to lie between 0 and 1
    x_test_eeg1 = (x_test_eeg1 - x_test_eeg1.min()) / (x_test_eeg1.max() - x_test_eeg1.min())
    x_test_eeg2 = (x_test_eeg2 - x_test_eeg2.min()) / (x_test_eeg2.max() - x_test_eeg2.min())
    x_test_emg = (x_test_emg - x_test_emg.min()) / (x_test_emg.max() - x_test_emg.min())

    # Combine into single array
    # Reshape data for LSTM
    x_test_eeg1 = x_test_eeg1.reshape(x_test_eeg1.shape[0], x_test_eeg1.shape[1], 1)
    x_test_eeg2 = x_test_eeg2.reshape(x_test_eeg2.shape[0], x_test_eeg2.shape[1], 1)
    x_test_emg = x_test_emg.reshape(x_test_emg.shape[0], x_test_emg.shape[1], 1)

    # Combine into single array
    x_test = np.concatenate((x_test_eeg1, x_test_eeg2, x_test_emg), axis=2)

    return x_test


def generate_submission(y_pred, filename):
    y_classes = y_pred.argmax(axis=1)
    df = pd.DataFrame(y_classes, columns=['y'])
    df.to_csv('submissions/' + filename + '.csv', index_label='Id')
