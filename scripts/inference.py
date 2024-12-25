# inference.py
import sys
import os
import pandas as pd
import re
from keras_preprocessing import sequence
sys.path.append("../")

from utils.config import Config
from model.train_classifier import BiLSTMClassifier

text_infer = "𡖵𣌉咏咏𤿰"

# Initialize the classifier
classifier = BiLSTMClassifier()

# Load the pre-trained model
model_path = "../checkpoint/bilstm_classifier_model.h5"  # Update this path if necessary
classifier.load_model(model_path)

# Create the tokenizer using the same training data
config = Config()
last_i = config.num_corpus - 1
last_j = config.num_folds - 1

training_data_path = f"../data/folds/polysen_corpus_{last_i}/{last_j}/train_{last_j}.csv"  # Update to your actual training data path
df, corpus = classifier.load_and_preprocess_data(training_data_path)
sequences = classifier.create_tokenizer(corpus)

# load model
model = classifier.load_model(model_path)
text_seq = classifier.tokenizer.texts_to_sequences([text_infer])
text_seq = sequence.pad_sequences(text_seq, maxlen=classifier.config.maxlen)

# Make predictions
predictions = model.predict(text_seq)
print(predictions)
# Print the predictions
if predictions[0][0] > 0.5:
    print("Predicted label: Nom")
else:
    print("Predicted label: Chinese")