class Config:
    embedding_size = 300  # embedding size of word embedding
    maxlen = 100  # maximum length of a sentence
    LSTM_output_size = 64  # dimensionality of the output space.
    loss = 'binary_crossentropy'
    size_ratio = 0.2  # ratio of the size of the training set to the size of the test set
    num_folds = 5  # number of folds
    num_corpus = 5
    mu = 10
    max_size = 20