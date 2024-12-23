import pandas as pd
import numpy as np
import random
import sys

sys.path.append("../")

class Partitioner:
    def __init__(self):
        pass
    
    def random_list_summing_to(self, total_sum, value_list, weight_list=None):
        """
        Generate a random list of values from `value_list` that sum up to `total_sum`.
        Parameters:
        total_sum: int
            The target sum that the returned list should add up to.
        value_list: list of int
            A list of positive integers to choose from.
        weight_list: list of float, optional
            A list of non-negative weights corresponding to the likelihood of each value in `value_list` being picked. 
            Must be the same length as `value_list`.
        Returns:
        list of int
            A list of integers from `value_list` that sum up to `total_sum`.
        Raises:
        ValueError
            If `value_list` is empty, contains non-positive values, or if `total_sum` is less than the smallest value in `value_list`.
            If `weight_list` is provided and is not the same length as `value_list`, contains negative weights, or all weights are zero.
            If it is impossible to form `total_sum` from the values in `value_list`.
        """
        

        # if not value_list:
            # raise ValueError("`value_list` must not be empty.")
        if any(x <= 0 for x in value_list):
            raise ValueError("All elements in `value_list` must be strictly positive.")
        if total_sum < min(value_list):
            raise ValueError(
                f"Impossible to form total_sum={total_sum} from smallest value={min(value_list)}."
            )
        
        # If weight_list is provided, some validations:
        if weight_list is not None:
            if len(weight_list) != len(value_list):
                raise ValueError("`weight_list` must have the same length as `value_list`.")
            if any(w < 0 for w in weight_list):
                raise ValueError("Weights must be non-negative.")
            # If all weights are zero, that would be invalid
            if all(w == 0 for w in weight_list):
                raise ValueError("All weights are zero; cannot sample from `value_list`.")

        result = []
        leftover = total_sum

        # Keep picking while leftover isn't exactly one of our possible values
        while leftover not in value_list:
            # Determine which values are feasible (i.e., do not exceed leftover)
            feasible_indices = [
                i for i, val in enumerate(value_list) 
                if val <= leftover
            ]
            if not feasible_indices:
                # No valid pick can be made to continue
                raise ValueError(
                    f"Cannot proceed further. leftover={leftover} cannot be formed from {value_list}."
                )
            
            # If weight_list is provided, filter it by feasible_indices
            if weight_list is not None:
                feasible_values = [value_list[i] for i in feasible_indices]
                feasible_weights = [weight_list[i] for i in feasible_indices]
                # random.choices was introduced in Python 3.6
                # k=1 means we pick exactly 1 item
                pick = random.choices(feasible_values, weights=feasible_weights, k=1)[0]
            else:
                # Otherwise pick uniformly from the feasible subset
                pick = random.choice([value_list[i] for i in feasible_indices])

            result.append(pick)
            leftover -= pick
            
            # If leftover hits 0, we are done
            if leftover == 0:
                return result

            # If leftover is below the smallest possible value but not zero, we can't proceed
            if leftover < min(value_list) and leftover != 0:
                raise ValueError(
                    f"Stuck with leftover={leftover}, which is less than the minimum {min(value_list)}."
                )

        # leftover is exactly in value_list, so we can just add it as the final element
        result.append(leftover)
        
        # shuffle
        random.shuffle(result)
        return result

    
    def random_cut_points(self, length, value_list, weight_list=None):
        """
        Generates a list of cumulative cut points based on random values summing to a specified length.
        Parameters:
        length: int
            The total length to which the random values should sum.
        value_list: list
            A list of values to be used for generating random values.
        weight_list: list, optional
            A list of weights corresponding to the values in value_list. If not provided, values are equally weighted.
        Returns:
        list
            A list of cumulative cut points starting from 0.
        """
        
        
        random_list = self.random_list_summing_to(length, value_list, weight_list)
        cumulative_list = np.cumsum(random_list).tolist()
        cmul_list = [0] + cumulative_list
        return cmul_list
    


    def generate_weight_list(self, size_list, method="uniform"):
        n = len(size_list)
        if method == "uniform":
            return [1] * n
        if method == "normal":
            mean = np.mean(size_list)
            std_dev = np.std(size_list)
            # sampling
            samples = np.random.normal(mean, std_dev, n)
            # Ensure no negative or zero weights
            samples = np.maximum(samples, 1e-6)
            return samples.tolist()
        if method == "lognormal":
            mean = np.mean(np.log(size_list))
            std_dev = np.std(np.log(size_list))
            # sampling
            samples = np.random.lognormal(mean, std_dev, n)
            # Ensure no negative or zero weights
            samples = np.maximum(samples, 1e-6)
            return samples.tolist()
        if method == "exponential":
            scale = 1.0 / np.mean(size_list)
            samples = np.random.exponential(scale, n)
            # Ensure no negative or zero weights
            samples = np.maximum(samples, 1e-6)
            return samples.tolist()
        
    
    def generate_polysen_df(self, df: pd.DataFrame, text_column: str, size_list, weight_list=None, separator=None):
        """
        Generates a DataFrame by partitioning the input DataFrame into segments and merging the text values from each segment.
        Parameters:
        df: pd.DataFrame
            The input DataFrame to be partitioned.
        text_column: str
            The name of the column containing text values to be merged.
        size_list: list
            A list of integers specifying the sizes of each segment.
        weight_list: list, optional
            A list of weights to be used for weighted random partitioning. Default is None.
        separator: str, optional
            A string used to join the text values in each segment. If None, values are merged as a list. Default is None.
        Returns:
        pd.DataFrame
            A DataFrame with the merged text values from each segment.
        """
        
        # if there are values in size_list larger than the length of the dataframe, raise an error
        
        
        # 1. Generate cut indices
        cuts = self.random_cut_points(len(df), size_list, weight_list)
        
        merged_chunks = []
        
        # 2. Iterate over cut segments
        for i in range(len(cuts) - 1):
            start_idx = cuts[i]
            end_idx = cuts[i + 1]  # next cut point
            
            # Slice the DataFrame
            chunk = df.iloc[start_idx:end_idx]
            
            # 3. Collect the values from `value_column` 
            values = chunk[text_column].tolist()
            
            # Merge as list or string
            if separator is None:
                merged_chunks.append(values)
            else:
                # Join by the specified separator
                merged_chunks.append(separator.join(str(v) for v in values))
        
        # convert to dataframe
        return pd.DataFrame({text_column: merged_chunks})
        
    
    def generate_polysen_corpus(self, df: pd.DataFrame, source_column, sentence_column, size_ratio=0.25, weight_method="uniform", separator=None):
        """
        Generate a polysynthetic corpus from a DataFrame by grouping and merging sentences.
        Parameters:
        df: pd.DataFrame
            The input DataFrame containing the data.
        source_column: str
            The name of the column to group by.
        sentence_column: str
            The name of the column containing sentences to be merged.
        size_ratio: float, optional
            The ratio of the subset size to be used for merging (default is 0.25).
        weight_method: str, optional
            The method to generate weights for merging ("uniform", "normal", "log-normal")
        separator: str, optional
            The separator to use when merging sentences (default is None).
        Returns:
        pd.DataFrame
            A DataFrame containing the merged sentences grouped by the source column.
        """
        
        # 1. Sort inplace by the source column
        df_sorted = df.sort_values(by=source_column, kind='mergesort', ignore_index=False)
        
        # 2. Prepare a dataframe
        results = []
        
        # 3. Loop over each unique value in group_column
        for val in df_sorted[source_column].unique():
            # Select all rows for the current 'val'
            subset = df_sorted[df_sorted[source_column] == val].copy()
            
            # Generate the size list and weight list
            size_list = list(range(1, min(len(subset), max(1, int(size_ratio * len(subset))) + 1)))
            weight_list = self.generate_weight_list(size_list, "lognormal")
            
            # Cut and merge the subset
            merged_chunk = self.generate_polysen_df(subset, sentence_column, size_list, weight_list, separator)
            merged_chunk[source_column] = val
            
            # Append the processed subset to the results list
            results.append(merged_chunk)
            
            
        # 5. Concatenate all processed subsets into one DataFrame
        result_df = pd.concat(results, ignore_index=True)
        # swap sentence and source column
        result_df = result_df[[source_column, sentence_column]]
        return result_df
        

class FoldSplitter:
    def __init__ (self, model):
        self.model = model
    
    
    def split_folds(self, df, num_folds=5, shuffle=True, categorical_column=None, fold_column=None):
        """
        Split a DataFrame into multiple folds, ensuring that each category is represented in each fold.
        Parameters:
        
        df: pd.DataFrame
            The input DataFrame to be split.
            
        num_folds: int
            The number of folds to split the DataFrame into.
            
        shuffle: bool
            Whether to shuffle the DataFrame before splitting. Default is True.
            
        categorical_column: str, optional
            The name of the column containing the categorical values to be balanced across folds.
            If None, the DataFrame is split randomly without considering any category.
            
        fold_column: str, optional
            The name of the column to store the fold number. If None, return list of DataFrames.
        
        Returns:
            List or dataframe 
        """
        
        # Shuffle the DataFrame if needed
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        # If no categorical column is provided, split randomly
        if categorical_column is None:
            # Calculate the number of samples per fold
            samples_per_fold = len(df) // num_folds
            # Split the DataFrame into num_folds parts
            folds = [df.iloc[i * samples_per_fold: (i + 1) * samples_per_fold] for i in range(num_folds)]
            
        else:
            # Group the DataFrame by the categorical column
            grouped = df.groupby(categorical_column)
            # Initialize an empty list to store the folds
            folds = [pd.DataFrame() for _ in range(num_folds)]
            
            # Iterate over the groups
            for name, group in grouped:
                # Calculate the number of samples per fold
                samples_per_fold = len(group) // num_folds
                # Split the group into num_folds parts
                group_folds = [group.iloc[i * samples_per_fold: (i + 1) * samples_per_fold] for i in range(num_folds)]
                
                # Assign each part to the corresponding fold
                for i in range(num_folds):
                    folds[i] = pd.concat([folds[i], group_folds[i]])
        
        # If fold_column is provided, add the fold number to the DataFrame
        if fold_column is not None:
            for i, fold in enumerate(folds):
                fold[fold_column] = i
                # append 
            return pd.concat(folds, ignore_index=True)
        else:
            return folds

# ví dụ sử dụng


# partitioner = Partitioner()

# # sources = sorted(random.choices(["A", "B", "C", "D"], k=100))
# # # sentences of format {source}_{index witihin corresponding source}
# # sentences = []
# # for i, src in enumerate(sources):
# #     if i == 0 or src != sources[i - 1]:
# #         sentences.append(f"{src}-0")
# #     else:
# #         sentences.append(f"{src}-{int(sentences[-1].split('-')[1]) + 1}")
# # df = pd.DataFrame({"source": sources, "sentence": sentences})

# # print(df)

# # df = pd.read_excel("../data/data_collection.xlsx")
# df = pd.read_csv("../data/data_sentences.csv")


# result = partitioner.generate_polysen_corpus(df, "source", "sentence", size_ratio=0.2, weight_method="lognormal", separator = "")

# # get label of each source from df and add to result
# result = result.merge(df[["source", "label"]].drop_duplicates(), on="source", how="left")

# result.to_csv("../data/polysen_corpus_2.csv", index=False)
