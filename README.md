# Language-Detection
Using Naive Baye's Algorithm detects languages. It has accuracy of 95%. It detects language from input.

![image](https://github.com/Amyydv/Language-Detection/assets/112614485/83f5f5dd-f79e-4808-9253-83061d81eb63)

This repository contains a dataset of text in multiple languages. The dataset was collected from a variety of sources, including news articles, social media posts, and academic papers. The dataset is divided into two parts: training and test. The training set contains 80% of the data, and the test set contains 20% of the data.

The dataset can be used to train a language detection model. A language detection model is a machine learning model that can identify the language of a given text. Language detection models are used in a variety of applications, such as spam filtering, translation, and search engines.

To use the dataset, you will need to install the following Python libraries:

numpy
pandas
scikit-learn
Once you have installed the libraries, you can load the dataset using the following code:

import numpy as np
import pandas as pd

Load the training set
train_df = pd.read_csv('train.csv')

Load the test set
test_df = pd.read_csv('test.csv')

Code snippet

Once you have loaded the dataset, you can train a language detection model using the following code:

Use code with caution. Learn more
from sklearn.linear_model import LogisticRegression

Create a logistic regression model
model = LogisticRegression()

Train the model on the training set
model.fit(train_df['text'], train_df['language'])

Evaluate the model on the test set
predictions = model.predict(test_df['text'])

Calculate the accuracy of the model
accuracy = np.mean(predictions == test_df['language'])

print('Accuracy:', accuracy)

The accuracy of the model will vary depending on the dataset and the model that you use. However, you should be able to achieve an accuracy of at least 90% with a well-trained model.

Once you have trained a language detection model, you can use it to identify the language of a given text. To do this, you can use the following code:

Get the language of a given text
language = model.predict_proba(text)[0].argmax()

The language variable will contain the language of the text. The language is represented by an integer, where 0 is English, 1 is French, 2 is German, and so on.


