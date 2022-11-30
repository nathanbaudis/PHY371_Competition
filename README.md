# Machine Learning for the Sciences Competition

Notebooks used to generate submissions in the in-class competition in Machine Learning for the Sciences, UZH fall semester 2020. 

The competition consists of classifying EEG and EMG data into three sleep categories. The metric used is the macro F1 score.

There are three approaches, using a dense and deep neural network (DeepNN), a convolutional neural network (CNN) and a long short-term memory network (LSTM).

For the competition, the deep neural network was used, achieving first place. It extracts features using the open source module [PyEEG](https://www.hindawi.com/journals/cin/2011/406391/) for EEG/MEG feature extraction, followed by a dense neural network. Class weights were used during training due to the imbalanced classes in the training data, and dropout layers were used for regularization. 

The other networks were produced after the competition, with less hyperparameter tuning.  They achieve the following scores:

| Network      | Macro F1 Score |
| ----------- | ----------- |
| DeepNN      | 0.905       |
| CNN   | 0.926        |
| LSTM  | 0.890        |

Further details:
* The data CSV files (not uploaded) are placed in a top-level 'data/' folder
* The 'utils' module provides functions for loading the data into a numpy array, as well as generating submission CSV files in the correct format
