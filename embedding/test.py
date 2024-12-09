from preprocessor import *
from embedding_model import *
from dictionary import *
emb = EmbeddingModel()

dictionary = Dictionary()
dictionary.load_from_excel("QuocNgu_SinoNom.xlsx", verbose=True)

emb.load("embedding.model", verbose=True)
print(emb)

# ans = emb.get_most_similar("𡮈",topn=10, dictionary=dictionary, source_col="SinoNom", target_col="QuocNgu")
# for line in ans:
#     print(line)
    
    
ans = emb.model.wv.similarity("㦨", "㦨")
print(ans)