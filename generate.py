import torch
import os
import argparse
import numpy as np
import pickle
from linguistic_style_transfer_pytorch.config import GeneralConfig
from linguistic_style_transfer_pytorch.model import AdversarialVAE
import json

gconfig = GeneralConfig()

# load word embeddings
weights = torch.FloatTensor(np.load(gconfig.word_embedding_path))
# load checkpoint
model_checkpoint = torch.load('linguistic_style_transfer_pytorch/checkpoints/model_epoch_20.pt')
# Load model
model = AdversarialVAE(weights)
model.load_state_dict(model_checkpoint)
model.eval()
# Load average style embeddings
with open(gconfig.avg_style_emb_path, 'rb') as f:
    avg_style_embeddings = pickle.load(f)
# set avg_style_emb attribute of the model
model.avg_style_emb = avg_style_embeddings
# load word2index
with open(gconfig.w2i_file_path) as f:
    word2index = json.load(f)
# load index2word
with open(gconfig.i2w_file_path) as f:
    index2word = json.load(f)
# load label2index
with open(gconfig.l2i_file_path) as f:
    label2index  = json.load(f)

# Read input sentence
source_sentence = input("Enter the source sentence: ")
target_style = input("Enter the target style: pos or neg: ")
# Get token ids
source_sentence += "<eos>"
token_ids = [word2index.get(word, gconfig.unk_token)
             for word in source_sentence.split()]
token_ids = torch.LongTensor(token_ids)
target_style_id = label2index[target_style].index(1)
# Get transfered sentence token ids
target_tokenids = model.transfer_style(token_ids, target_style_id)
target_sentence = " ".join([index2word[str(int(idx))] for idx in target_tokenids])
print("Style transfered sentence: {}".format(target_sentence))
