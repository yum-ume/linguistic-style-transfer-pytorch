# Linguistic Style Transfer 
Implementation of the paper `Disentangled Representation Learning for Non-Parallel Text Style Transfer`[(link)](https://www.aclweb.org/anthology/P19-1041.pdf) in Pytorch.
The original repository is [link](https://github.com/h3lio5/linguistic-style-transfer-pytorch) .
Basically, the file structure or function names have not been changed.

## Requirements
```
python=3.9.12
pytorch=1.11.0
scikit-learn=1.0.1
nltk=3.7
gensim= 4.1.2
```
## Note
- Run all the commands from the root directory.   
- The parameters for your experiment are all set by defualt. But you are free to set them on your own by editing the `config.py` file.

## 0. Preprocess
### 1. Preparing train data
```
python preprocess.py
```

### 2. Training word embeddings by Word2Vec
```
python linguistic_style_transfer_pytorch/utils/train_w2v.py
```

### 3. Preparing vocabulary files
```
python linguistic_style_transfer_pytorch/utils/vocab.py
```

## 1. Training model from scratch
```
python train.py
```
## 2. Transfering Text Style from Trained Models
```
python generate.py
```
