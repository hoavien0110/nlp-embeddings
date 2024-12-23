import sys
import os
import pandas as pd
sys.path.append("../")

from utils.config import Config
from utils.partitioner import Partitioner

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

from embedding.train import train_embedding
from embedding.preprocessor import PreProcessor

from model.train_classifier import BiLSTMClassifier

partitioner = Partitioner()
config = Config()

def create_data():
    df = pd.read_csv("../data/data_sentences.csv")

    for i in range(config.num_corpus):
        result = partitioner.generate_polysen_corpus(df, "source", "sentence", size_ratio=config.size_ratio, weight_method="lognormal", separator = "")
        result = result.merge(df[["source", "label"]].drop_duplicates(), on="source", how="left")

        kf = GroupKFold(n_splits=config.num_folds)
        if not os.path.exists(f"../data/folds/polysen_corpus_{i}"):
            os.makedirs(f"../data/folds/polysen_corpus_{i}")

        for j, (train_index, test_index) in enumerate(kf.split(result, groups=result["source"])):
            train = result.iloc[train_index]
            test = result.iloc[test_index]
            if not os.path.exists(f"../data/folds/polysen_corpus_{i}/{j}"):
                os.makedirs(f"../data/folds/polysen_corpus_{i}/{j}")
            train.to_csv(f"../data/folds/polysen_corpus_{i}/{j}/train_{j}.csv", index=False)
            test.to_csv(f"../data/folds/polysen_corpus_{i}/{j}/test_{j}.csv", index=False)


if __name__ == "__main__":
    create_data()

    for i in range(config.num_corpus):
        for j in range(config.num_folds):
            train_embedding(f"folds/polysen_corpus_{i}/{j}/train_{j}.csv")
            classifier = BiLSTMClassifier()
            classifier.run_pipeline(
                train_path=f"../data/folds/polysen_corpus_{i}/{j}/train_{j}.csv",
                test_path=f"../data/folds/polysen_corpus_{i}/{j}/test_{j}.csv"
            )


