import re
import pandas as pd

DATA_PREFIX = 'data/'


class PreProcessor:
    def __init__(self, data_prefix=DATA_PREFIX):
        self.data_prefix = data_prefix 
        
    
    def read_excel_corpus(self, file_name, verbose=False):
        result = pd.read_excel(DATA_PREFIX + file_name)
        if verbose:
            print("Read", len(result), "lines from", file_name)
        return result
    
    
    def remove_invalid_lines(self, lines, verbose=False):
        valid_lines = [line for line in lines if type(line) == str]
        if verbose:
            print("Removed", len(lines) - len(valid_lines), "invalid lines")
        return valid_lines  
    
    
    def remove_non_CJK(self, text):
        split_text = text.split()
        cleaned_text = ""
        for word in split_text:
            for char in word:
                if self.is_CJK(char):
                    cleaned_text += char
        return cleaned_text
        
    
    
    def is_CJK(self, char):
        is_standard = 0x4E00 <= ord(char) <= 0x9FFF
        if (is_standard):
            return True
        is_ext_A = 0x3400 <= ord(char) <= 0x4DBF
        if (is_ext_A):
            return True
        is_ext_B = 0x20000 <= ord(char) <= 0x2A6DF
        if (is_ext_B):
            return True
        is_ext_C = 0x2A700 <= ord(char) <= 0x2B73F
        if (is_ext_C):
            return True
        is_ext_D = 0x2B740 <= ord(char) <= 0x2B81F
        if (is_ext_D):
            return True
        is_ext_E = 0x2B820 <= ord(char) <= 0x2CEAF
        if (is_ext_E):
            return True
        is_ext_F = 0x2CEB0 <= ord(char) <= 0x2EBEF
        if (is_ext_F):
            return True
        is_ext_G = 0x30000 <= ord(char) <= 0x3134F
        if (is_ext_G):
            return True
        is_ext_H = 0x31350 <= ord(char) <= 0x323AF
        if (is_ext_H):
            return True
        return False


    def tokenize(self, line):
        cleaned_line = self.remove_non_CJK(line)
        return list(cleaned_line)