import pandas as pd

DATA_PREFIX = 'dict/'

class Dictionary:
    def __init__(self, data_prefix=DATA_PREFIX):
        self.data_prefix = data_prefix
        self.df = None
        self.name = None
        
        
    def __str__(self):
        result = ""
        if self.df is not None:
            result += "Dictionary name: " + self.name + "\n"
            result += "Dictionary size: " + str(len(self.df)) + "\n"
            result += "Dictionary columns: " + str(self.df.columns) + "\n"
        else:
            result += "No dictionary loaded"
        return result
        
        
    def load_from_excel(self, file_name, verbose=False):
        self.df = pd.read_excel(DATA_PREFIX + file_name)
        self.name = file_name
        if verbose:
            print(self.__str__())
        return self.df
    
    
    def lookup(self, key, source_col, target_col):
        if self.df is None:
            raise ValueError("No dictionary loaded")
        result = self.df[self.df[source_col] == key][target_col].tolist()
        if len(result) == 0:
            return None
        return result

