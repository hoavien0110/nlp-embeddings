from preprocessor import *
from embedding_model import *

# Step 1: Preprocessing
file_path = "data_collection.xlsx"
preproc = PreProcessor()
corpus = preproc.read_excel_corpus(file_path, verbose=True)

lines = corpus["Text"].tolist()
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
