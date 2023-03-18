import os
import re
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Define the directory path for the dataset
data_dir = "path/to/reuters-21578"

# Load the dataset
dataset = load_files(data_dir, shuffle=False)

# Split the dataset into training and testing sets
docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)

# Define the pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# Train the model
text_clf.fit(docs_train, y_train)

# Test the model
predicted = text_clf.predict(docs_test)

# Print the accuracy score
print("Accuracy:", np.mean(predicted == y_test))
