import os
import sys

sys.path.append("../")
sys.path.append("./")

from embedding.preprocessor import PreProcessor
from embedding.embedding_model import *

# Step 1: Preprocessing
def train_embedding(file_path: str): 
    
    preproc = PreProcessor()
    corpus = preproc.read_csv_corpus(file_path, verbose=True)

    lines = corpus["sentence"].tolist()
    lines = preproc.remove_invalid_lines(lines, verbose=True)

    tokenized_lines = [preproc.tokenize(line) for line in lines]

    # Step 2: Training the embedding
    embedding_model = EmbeddingModel()
    embedding_model.train(tokenized_lines = tokenized_lines,
                                window = 10,
                                vector_size = 300,
                                min_count = 1,
                                sg=0,
                                epochs=50,
                                model_type="FastText",
                                verbose=True
    )
                                

    # Step 3: Saving the model
    embedding_model.save("../checkpoint/embedding.model", verbose=True)
