import pandas as pd
import numpy as np

class Partitioner:
    def __init__(self):
        pass
    
    
    
    def generate_positive_integer_series_with_sum(self, n, m, mean=0, std_dev=1):
        # Step 1: Generate raw normal distribution
        X_raw = np.random.lognormal(mean=mean, sigma=std_dev, size=m)
        
        # Step 2: Shift to ensure positivity
        shift = abs(min(X_raw)) + 1  # Ensure all values are positive
        X_shifted = X_raw + shift

        # Step 3: Scale to sum to n
        X_scaled = (n / np.sum(X_shifted)) * X_shifted

        # Step 4: Round to integers
        X_rounded = np.round(X_scaled).astype(int)

        # Step 5: Adjust for sum to n
        delta = n - np.sum(X_rounded)
        while delta != 0:
            idx = np.random.choice(range(m))  # Randomly pick an index
            if delta > 0 and X_rounded[idx] > 0:
                X_rounded[idx] += 1
                delta -= 1
            elif delta < 0 and X_rounded[idx] > 1:
                X_rounded[idx] -= 1
                delta += 1

        return X_rounded

    def assign_continuous_partition_values(self, df, n_paritions, partition_column, ratio=None) -> pd.DataFrame:
        """
        Assigns continuous partition values to a DataFrame.
        
        This function partitions a DataFrame into a specified number of partitions or 
        based on a partition ratio. The partition values are assigned to a new column 
        in the DataFrame.
        
        Parameters:
        - `df (pd.DataFrame)`: The DataFrame to be partitioned.
        - `n_paritions (int)`: The number of partitions to create.
        - `partition_column (str)`: The name of the column to store partition values.
        - `ratio (float, optional)`: The ratio of the DataFrame to be partitioned. 
                                           If provided, it overrides n_paritions.
        
        Returns:
        `pd.DataFrame`: The DataFrame with an additional column containing partition values.
        Raises:
        AssertionError: If n_paritions is not between 1 and the length of the DataFrame.
        """

        if ratio is not None:
            n_paritions = int(len(df) * ratio)
            
        assert 0 < n_paritions <= len(df), "The number of partitions must be between 1 and the length of the dataframe."
        cut_points = [0] + sorted(np.random.choice(range(1, len(df)- 1), n_paritions - 1, replace=False)) + [len(df)]
        
        
        df[partition_column] = pd.cut(df.index, bins=cut_points, labels=range(n_paritions))
        return df


    def group_partitions(self, df, group_column, value_column, delimiter):
        """
        Group the values in the value_column by the group_column.
        
        Parameters:
        - `df (pandas.DataFrame)`: The DataFrame containing the data to be grouped.
        - `group_column (str)`: The column name to group by.
        - `value_column (str)`: The column name whose values will be grouped.
        - `delimiter (str or None)`: The delimiter to use for joining the grouped values. 
                                 If None, the values will be grouped into a list.
        Returns:
        `pandas.DataFrame`: A DataFrame with the grouped values.
        """
        if delimiter is None:
            grouped = df.groupby(group_column)[value_column].apply(list).reset_index()
        else:
            grouped = df.groupby(group_column)[value_column].apply(lambda x: delimiter.join(map(str, x))).reset_index()
        return grouped


    def partition_and_group(self, df, value_column, partition_column, ratio=0.5, delimiter=None):
        n_partitions = int(len(df) * ratio)
        df = self.assign_continuous_partition_values(df, n_partitions, partition_column)
        df = self.group_partitions(df, partition_column, value_column, delimiter=delimiter)
        return df


    def parition_and_group_for_each_category(self, df, value_column, category_column, partition_column="partition", catwise_ratio=0.5, delimiter=None):
        result = pd.DataFrame(columns=[category_column, value_column])
        for category, sub_df in df.groupby(category_column):
            sub_df = sub_df.reset_index(drop=True)
            sub_df = self.partition_and_group(sub_df, value_column, partition_column, catwise_ratio, delimiter)
            sub_df[category_column] = category
            sub_df.drop(partition_column, axis=1, inplace=True)
            result = pd.concat([result, sub_df], ignore_index=True)
        return result
    
    
# test
categories = sorted(np.random.choice(["A", "B", "C", "D", "E"], 200))
values = list(range(200))
df = pd.DataFrame({"category": categories, "value": values})
partitioner = Partitioner()
result = partitioner.parition_and_group_for_each_category(df, "value", "category", catwise_ratio=0.26)
print(result)