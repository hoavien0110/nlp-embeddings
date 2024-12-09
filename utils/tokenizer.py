from keras_preprocessing.text import Tokenizer

def create_tokenizer(corpus, char_level=True, oov_token="<UNK>"):
    """Create and fit a tokenizer."""
    tokenizer = Tokenizer(char_level=char_level, oov_token=oov_token)
    tokenizer.fit_on_texts(corpus)
    return tokenizer
