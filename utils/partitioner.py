import pandas as pd
import numpy as np
import random
import sys
import scipy.stats as stats
import random
from visualizer import plot_discrete_pdf
import matplotlib.pyplot as plt
import json


sys.path.append("../")




class Partitioner:
    def __init__(self):
        self.PROBABILITY_DENISTY_FUNCTIONS = {
            "uniform":
                {
                    "pdf": stats.uniform.pdf,
                    "params": {}
                },
            "exponential":
                {
                    "pdf": stats.expon.pdf,
                    "params": {"scale": 1}
                },
            "normal":
                {
                    "pdf": stats.norm.pdf,
                    "params": {"loc": 0, "scale": 1}
                },
            "gamma":
                {
                    "pdf": stats.gamma.pdf,
                    "params": {"a": 1}
                },
            "lognormal":
                {
                    "pdf": stats.lognorm.pdf,
                    "params": {"s": 1}
                },
            "poisson":
                {
                    "pdf": stats.poisson.pmf,
                    "params": {"mu": 1}
                }
        }



    
    
    def calculate_discrete_pdfs(self, values, **kwargs):
        """
        Calculate the discrete probability density function (PDF) for a list of values.
        Parameters:
        values: list
            A list of values for which the PDF will be calculated.
        pdf: function
            The PDF function to be used for calculation.
        params: dict
            A dictionary of parameters to be passed to the PDF function.
        Returns:
        list
            A list of PDF values corresponding to the input value list.
        
        """
        assert 'pdf' in kwargs, "pdf function must be provided"
        assert 'params' in kwargs, "pdf parameters must be provided"

        pdf_func = kwargs['pdf']
        pdf_params = kwargs['params']
        return [pdf_func(value, **pdf_params) for value in values]


    def generate_size_weights(self, sizes, pdf_func=stats.uniform.pdf, **kwargs):
        # Calculate the PDF values for the size list
        pdf_values = self.calculate_discrete_pdfs(sizes, pdf=pdf_func, params=kwargs)
        
        # Normalize the PDF values
        pdf_sum = sum(pdf_values)
        weights = [pdf_value / pdf_sum for pdf_value in pdf_values]
        return weights


    def generate_integer_partition(self, target, values, weights):
        remaining = target
        result = []
        while remaining > 0:
            value = random.choices(values, weights=weights)[0]
            if value > remaining:
                result.append(remaining)
                break
            result.append(value)
            remaining -= value
        return result
    

    def generate_random_cut_points(self, length, values, weights):
        """
        Generates a list of cumulative cut points based on random values summing to a specified length.
        Parameters:
        length: int
            The total length to which the random values should sum.
        values: list
            A list of values to be used for generating random values.
        weights: list, optional
            A list of weights corresponding to the values in value_list. If not provided, values are equally weighted.
        Returns:
        list
            A list of cumulative cut points starting from 0.
        """
        integer_partition = self.generate_integer_partition(length, values, weights)
        cumulative_list = np.cumsum(integer_partition).tolist()
        cmul_list = [0] + cumulative_list
        return cmul_list
    



    def generate_polysen_df(self, df: pd.DataFrame, text_column: str, sizes, size_weights=None, separator=None):
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
        cuts = self.generate_random_cut_points(len(df), sizes, size_weights)
        
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
        
    
    def generate_polysen_corpus(self, df: pd.DataFrame, source_column, sentence_column, sizes, size_weights=None, separator=None):
        # 1. Sort inplace by the source column
        df_sorted = df.sort_values(by=source_column, kind='mergesort', ignore_index=False)
        
        # 2. Prepare a dataframe
        results = []
        
        # 3. Loop over each unique value in group_column
        for val in df_sorted[source_column].unique():
            # Select all rows for the current 'val'
            subset = df_sorted[df_sorted[source_column] == val].copy()
            
            # Cut and merge the subset
            merged_chunk = self.generate_polysen_df(subset, sentence_column, sizes, size_weights, separator)
            merged_chunk[source_column] = val
            
            # Append the processed subset to the results list
            results.append(merged_chunk)
            
            
        # 5. Concatenate all processed subsets into one DataFrame
        result_df = pd.concat(results, ignore_index=True)
        # swap sentence and source column
        result_df = result_df[[source_column, sentence_column]]
        return result_df
        

    def generate_polysen_corpus_with_random_size_weights(self, df: pd.DataFrame, source_column, sentence_column, separator=None, distribution="uniform", custom_params=None, max_size=10, draw_distribution=False, verbose=False):
        """
        Generates a polysen corpus with random size weights based on a specified distribution.

        PARAMETERS:
        `df`: `pd.DataFrame`  
            The input dataframe containing the data.

        `source_column`: `str`  
            The column name in the dataframe that contains the source identifiers.

        `sentence_column`: `str`  
            The column name in the dataframe that contains the sentences.

        `separator`: `str, optional`  
            The separator to use between sentences. Defaults to None.

        `distribution`: `str, optional`  
            The type of distribution to use for generating size weights. Defaults to "uniform".

        `custom_params`: `dict, optional`  
            Custom parameters for the distribution. Defaults to None.

        `max_size`: `int, optional`  
            The maximum size of the generated polysen corpus. Defaults to 10.

        `draw_distribution`: `bool, optional`  
            Whether to draw the distribution plot. Defaults to False.

        `verbose`: `bool, optional`  
            Whether to print detailed information. Defaults to False.

        RETURNS:
        `pd.DataFrame`  
            A dataframe containing the generated polysen corpus.

        RAISES:
        `ValueError`  
            If the specified distribution is not valid.
        """
        if distribution not in self.PROBABILITY_DENISTY_FUNCTIONS.keys():
            raise ValueError(f"Invalid distribution: {distribution}, must be one of {list(self.PROBABILITY_DENISTY_FUNCTIONS.keys())}")
        params = custom_params if custom_params is not None else self.PROBABILITY_DENISTY_FUNCTIONS[distribution]["params"]
        
        
        # 1. Sort inplace by the source column
        df_sorted = df.sort_values(by=source_column, kind='mergesort', ignore_index=False)
        # Get the maximum number of sentences for each source
        max_size = min(max_size, df_sorted.groupby(source_column).size().max())

        
        # Generate sizes
        sizes = list(range(1, max_size + 1))
        size_weights = self.generate_size_weights(sizes, pdf_func=self.PROBABILITY_DENISTY_FUNCTIONS[distribution]["pdf"], **params)
    
        if verbose:
            print("Distribution type:", distribution)
            print("Distribution parameters:" )
            print(json.dumps(params, indent=4))
            print("Max size:", max_size)
            for i in range(len(sizes)):
                print(f"Size: {sizes[i]}, Weight: {size_weights[i].round(3)}")
            
        
        # draw plot
        if draw_distribution:
            plt.gca().set_title(f"{distribution} distribution for paragraph size")
            plt.gca().set_xlabel("Size")
            plt.gca().set_ylabel("Probability")
            # label on top of bar
            plt.gca().set_xticks(sizes)
            
            plot_discrete_pdf(self.PROBABILITY_DENISTY_FUNCTIONS[distribution]["pdf"], 
                params, 
                sizes, 
                ax=plt.gca(),
                )
            
        # Generate the polysen corpus using the provided sizes and weights
        return self.generate_polysen_corpus(df, source_column, sentence_column, sizes, size_weights, separator)