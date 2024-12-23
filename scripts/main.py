import sys
import os
import pandas as pd
sys.path.append("../")

from utils.config import Config
from utils.partitioner import Partitioner
from sklearn.model_selection import KFold
partitioner = Partitioner()
config = Config()
# sources = sorted(random.choices(["A", "B", "C", "D"], k=100))
# # sentences of format {source}_{index witihin corresponding source}
# sentences = []
# for i, src in enumerate(sources):
#     if i == 0 or src != sources[i - 1]:
#         sentences.append(f"{src}-0")
#     else:
#         sentences.append(f"{src}-{int(sentences[-1].split('-')[1]) + 1}")
# df = pd.DataFrame({"source": sources, "sentence": sentences})

# print(df)

# df = pd.read_excel("../data/data_collection.xlsx")
df = pd.read_csv("../data/data_sentences.csv")

for i in range(config.num_corpus - 1):
    result = partitioner.generate_polysen_corpus(df, "source", "sentence", size_ratio=config.size_ratio, weight_method="lognormal", separator = "")

    # get label of each source from df and add to result
    result = result.merge(df[["source", "label"]].drop_duplicates(), on="source", how="left")

    # split to 5 folds and save 5 folds to csv
    kf = KFold(n_splits=config.num_folds, shuffle=True)
    # create fold_i folder if not exist
    if not os.path.exists(f"../data/folds/polysen_corpus_{i}"):
        os.makedirs(f"../data/folds/polysen_corpus_{i}")

    for j, (train_index, test_index) in enumerate(kf.split(result)):
        train = result.iloc[train_index]
        test = result.iloc[test_index]
        if not os.path.exists(f"../data/folds/polysen_corpus_{i}/{j}"):
            os.makedirs(f"../data/folds/polysen_corpus_{i}/{j}")
        train.to_csv(f"../data/folds/polysen_corpus_{i}/{j}/train_{j}.csv", index=False)
        test.to_csv(f"../data/folds/polysen_corpus_{i}/{j}/test_{j}.csv", index=False)

# print(result)

