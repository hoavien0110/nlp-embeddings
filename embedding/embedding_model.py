import os
import sys
sys.path.append(os.path.join('./'))

from gensim.models import *
from tqdm import tqdm

class EmbeddingModel:
    def __init__(self):
        self.model = None
        
        
    def __str__(self):
        result = ""
        if self.model:
            result += "Model type: " + str(type(self.model)) + "\n"
            result += "Model size: " + str(len(self.model.wv)) + "\n"
            result += "Vector size: " + str(self.model.vector_size) + "\n"
        else:
            result += "No model loaded"
        return result
            
            
    def train(self, 
                        tokenized_lines,
                        vector_size=100, 
                        window=5, 
                        min_count=2, 
                        sg=1,
                        epochs=10,
                        model_type="FastText",
                        verbose=False):
        
        if model_type not in ["FastText", "Word2Vec"]:
            raise ValueError("model_type must be either 'FastText' or 'Word2Vec'")
        
        if verbose:
            print("Training model with", len(tokenized_lines), "lines")
        
        if model_type == "FastText":
            model = FastText(vector_size=vector_size,
                             window=window, 
                             min_count=min_count, 
                             sg=sg)
        else:
            model = Word2Vec(vector_size=vector_size,
                             window=window, 
                             min_count=min_count, 
                             sg=sg)
        
        model.build_vocab(corpus_iterable=tokenized_lines)
        
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            model.train(total_examples=len(tokenized_lines), 
                        epochs=1,
                        corpus_iterable=tokenized_lines)
        
        if verbose:
            print("Model trained with", len(model.wv), "words")
        self.model = model
        return model
    
    
    def save(self, model_path="embedding.model", verbose=False):
        """Saves the model to a file."""
        if verbose:
            print("Saving model to", model_path)
        self.model.save(model_path)
        if verbose:
            print("Model saved")


    def load(self, model_path="embedding.model", verbose=False):
        """Loads a pre-trained model."""
        if verbose:
            print("Loading model from", model_path)
        self.model = Word2Vec.load(model_path)
        return self.model
    

    # def get_most_similar(self, word, topn=5, dictionary=None, source_col=None, target_col=None):
    #     similar_words = self.model.wv.most_similar(word, topn=topn)
        
    #     if dictionary and source_col and target_col:
    #         result = []
    #         for word, similarity in similar_words:
    #             translation = dictionary.lookup(word, source_col, target_col)
    #             result.append((word, similarity, translation))
    #         return result
    #     return similar_words

    