import re
import logging
import sys
import json
from os.path import dirname, abspath

parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from linguistic_style_transfer_pytorch.config import GeneralConfig

config = GeneralConfig()


class Preprocessor():
    """
    Preprocessor class
    """

    def __init__(self):

        print("Preprocessor instantiated")

    def _clean_text(self, string):
        """
        Clean the raw text file
        """
        string = string.replace(".", "")
        string = string.replace(".", "")
        string = string.replace("\n", " ")
        string = string.replace(" 's", " is")
        string = string.replace("'m", " am")
        string = string.replace("'ve", " have")
        string = string.replace("n't", " not")
        string = string.replace("'re", " are")
        string = string.replace("'d", " would")
        string = string.replace("'ll", " will")
        string = string.replace("\r", " ")
        string = string.replace("\n", " ")
        string = re.sub(r'\d+', "number", string)
        string = ''.join(x for x in string if x.isalnum() or x == " ")
        string = re.sub(r'\s{2,}', " ", string)
        string = string.strip().lower()

        return string

    def preprocess(self):
        """
        Preprocesses the train text data 
        """
        print("Preprocessing started")
        with open(config.train_text_file_path, 'w') as text_file, open(config.train_labels_file_path, 'w') as labels_file:
            with open(config.train_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    line = self._clean_text(line)
                    if len(line) > 0:
                        text_file.write(line + "\n")
                        labels_file.write("pos" + "\n")
            with open(config.train_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    line = self._clean_text(line)
                    if len(line) > 0:
                        text_file.write(line + "\n")
                        labels_file.write("neg" + "\n")
        with open(config.l2i_file_path, 'w') as label2index_file:
            label_dict = {"neg": [0, 1], "pos": [1, 0]}
            json.dump(label_dict, label2index_file)
        print("Processing complete ")

if  __name__ == '__main__':
    prepro = Preprocessor()
    prepro.preprocess()
