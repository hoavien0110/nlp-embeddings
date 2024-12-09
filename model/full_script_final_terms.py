
import pandas as pd
from datetime import date, timedelta
import re
from nltk.tokenize import word_tokenize
from huggingface_hub import hf_hub_download
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from preprocessor import *
from embedding_model import *
from dictionary import *
import re
from keras_preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Bidirectional
from sklearn.model_selection import train_test_split

class Config:
    embedding_size = 300 # embedding size of word embedding
    maxlen = 100 # maximum length of a sentence
    LSTM_output_size = 64 # dimensionality of the output space.
    loss = 'binary_crossentropy'

config = Config()

embedding_model = EmbeddingModel()
embedding_model.load("embedding.model", verbose=True)
print(embedding_model)

dictionary = Dictionary()
dictionary.load_from_excel("quocngu_sinonom.xlsx", verbose=True)

df = pd.read_excel("data/data_collection.xlsx")
df.rename(columns={"Character": "Label"}, inplace=True)
df.rename(columns={"Text": "sentence"}, inplace=True)

data_sentences = list(df['sentence'].values)
data_sentences = [x for x in data_sentences if type(x) == str]
data_sentences = [re.sub(r'\s+', '', item) for item in data_sentences]
data_sentences = [re.sub(r'。', '', item) for item in data_sentences]

corpus = [list(sentence) for sentence in data_sentences]

oov_token = "<UNK>"

tokenizer = Tokenizer(char_level=True, oov_token=oov_token)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index
print(word_index)
vocab_size = len(word_index) + 1  # Plus 1 because indices start from 1

# Initialize the embedding matrix with zeros
embedding_matrix = np.zeros((vocab_size, config.embedding_size))

# Fill the embedding matrix
for word, i in word_index.items():
    try:
        # Get the FastText vector for the word
        embedding_vector = embedding_model.model.wv[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        # If no embedding is found for a word, leave the vector as zero
        embedding_matrix[i] = np.zeros(config.embedding_size)

classifierModel = Sequential()
classifierModel.add(Embedding(vocab_size, config.embedding_size, input_length=config.maxlen, weights=[embedding_matrix], trainable=False))
classifierModel.add(Bidirectional(LSTM(config.LSTM_output_size, return_sequences=True, input_shape=(config.maxlen, ))))
classifierModel.add(Flatten())
classifierModel.add(Dense(1, activation='sigmoid'))
classifierModel.compile(optimizer='adam', loss=config.loss, metrics=['acc'])
classifierModel.summary()


df['BinaryLabel'] = df['Label'].apply(lambda x: 1 if x == 'Nom' else 0)
df = df[df['sentence'].apply(lambda x: isinstance(x, str))]

def clean_text(text):
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'。', '', text)
    return text

df['sentence_cleaned'] = df['sentence'].apply(lambda x: clean_text(x))

sequences = tokenizer.texts_to_sequences(df['sentence_cleaned'],)
X_data = sequence.pad_sequences(sequences, maxlen=config.maxlen)
y_data = np.array(df['BinaryLabel'])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=101)
classifierModel.fit(X_train, y_train, epochs=2, batch_size=32)

predictions = classifierModel.predict(X_test)
predictions = np.round(predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


