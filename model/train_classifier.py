import os
import sys
import re
import numpy as np
import pandas as pd

sys.path.append(os.path.join('../'))

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from utils.dataframe_preprocessing import load_and_clean_data
from utils.tokenizer import create_tokenizer
from utils.config import Config
from embedding.embedding_model import EmbeddingModel

from huggingface_hub import hf_hub_download
from keras_preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Embedding, Flatten, Dense, LSTM, Bidirectional

# Ensure path for relative imports

class BiLSTMClassifier:
    def __init__(self, config_path="../utils/config.py", embedding_path="../checkpoint/embedding.model"):
        """
        Initialize the BiLSTM Classifier with configuration and embedding model.
        
        Args:
            config_path (str): Path to the configuration file.
            embedding_path (str): Path to the embedding model file.
        """
        self.config = Config()
        self.embedding_model = EmbeddingModel()
        self.embedding_model.load(embedding_path, verbose=True)
        self.tokenizer = None
        self.embedding_matrix = None
        self.vocab_size = None
        self.model = None

    def load_and_preprocess_data(self, file_path):
        """
        Load and clean the data from the specified file.
        
        Args:
            file_path (str): Path to the data file.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = load_and_clean_data(file_path)
        data_sentences = df['sentence'].dropna().astype(str).apply(lambda x: re.sub(r'\s+|。', '', x)).tolist()
        corpus = [list(sentence) for sentence in data_sentences]
        return df, corpus

    def create_tokenizer(self, corpus, oov_token="<UNK>"):
        """
        Create and fit a tokenizer on the corpus.
        
        Args:
            corpus (list): List of tokenized sentences.
            oov_token (str): Token for out-of-vocabulary words.
        """
        self.tokenizer = create_tokenizer(corpus, oov_token)
        sequences = self.tokenizer.texts_to_sequences(corpus)
        self.vocab_size = len(self.tokenizer.word_index) + 1  # Add 1 since indices start from 1
        return sequences

    def create_embedding_matrix(self):
        """
        Create an embedding matrix from the embedding model.
        
        Returns:
            np.array: Embedding matrix.
        """
        embedding_matrix = np.zeros((self.vocab_size, self.config.embedding_size))
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = self.embedding_model.model.wv[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                pass
        self.embedding_matrix = embedding_matrix
        return embedding_matrix

    def prepare_data(self, sequences, df):
        """
        Prepare training and testing data for model training.
        
        Args:
            sequences (list): List of tokenized sequences.
            df (pd.DataFrame): DataFrame containing the labels.
        
        Returns:
            tuple: Split datasets (X_train, X_test, y_train, y_test).
        """
        X = sequence.pad_sequences(sequences, maxlen=self.config.maxlen)
        y = df['BinaryLabel'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def build_model(self):
        """
        Build the BiLSTM classification model.
        
        Returns:
            Sequential: Compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.config.embedding_size, weights=[self.embedding_matrix], 
                            input_length=self.config.maxlen, trainable=False))
        model.add(Bidirectional(LSTM(self.config.LSTM_output_size)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss=self.config.loss, metrics=['acc'])
        self.model = model
        model.summary()
        return model

    def train_model(self, X_train, y_train, epochs=2, batch_size=32):
        """
        Train the BiLSTM classification model.
        
        Args:
            X_train (np.array): Training feature set.
            y_train (np.array): Training labels.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def save_model(self, save_path="../checkpoint/bilstm_classifier_model.h5"):
        """
        Save the trained model to the specified file path.
        
        Args:
            save_path (str): Path to save the trained model.
        """
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

    def run_pipeline(self, data_path="../data/data_collection.xlsx", save_model_path="../checkpoint/bilstm_classifier_model.h5"):
        """
        Run the complete pipeline from data loading, processing, model training to model saving.
        
        Args:
            data_path (str): Path to the input data file.
            save_model_path (str): Path to save the trained model.
        """
        print("Step 1: Loading and cleaning data...")
        df, corpus = self.load_and_preprocess_data(data_path)
        
        print("Step 2: Creating tokenizer...")
        sequences = self.create_tokenizer(corpus)
        
        print("Step 3: Creating embedding matrix...")
        self.create_embedding_matrix()
        
        print("Step 4: Preparing data for training...")
        X_train, X_test, y_train, y_test = self.prepare_data(sequences, df)
        
        print("Step 5: Building the BiLSTM model...")
        self.build_model()
        
        print("Step 6: Training the model...")
        self.train_model(X_train, y_train)
        
        print("Step 7: Saving the trained model...")
        self.save_model(save_model_path)
        
        print("Pipeline complete! Model saved to", save_model_path)


if __name__ == "__main__":
    classifier = BiLSTMClassifier()
    classifier.run_pipeline()
