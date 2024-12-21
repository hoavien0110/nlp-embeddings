import pandas as pd
import numpy as np

class Partitioner:
    def __init__(self):
        pass

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


    def parition_and_group_for_each_category(self, df, value_column, category_column, partition_column="partition", ratio=0.5, delimiter=None):
        result = pd.DataFrame(columns=[category_column, value_column])
        for category, sub_df in df.groupby(category_column):
            sub_df = sub_df.reset_index(drop=True)
            sub_df = self.partition_and_group(sub_df, value_column, partition_column, ratio, delimiter)
            sub_df[category_column] = category
            result = pd.concat([result, sub_df], ignore_index=True)
        return result
    
    
    