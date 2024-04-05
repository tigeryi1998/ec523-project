# ec523-project: Re-implementing data2vec

This project is from Rohan Kumar, Tiger Yi, Hao Chen, Shi Gu for EC523 Deep Learning, Spring 2024. 

> NOTE: the data2vec directory contains the code from the original implementation (i.e. Meta's data2vec)

## Implemented as of 04/04/2024

1. Finding datasets and writing scripts to pull datasets from system
2. Image (Rohan and Tiger):
    1. Dataset wrapper class for DataLoader usage later on, applies all transforms needed, in image/data/data_util.py
    1. Embedding System in image/data/data_util.py
    2. Masking Generator in image/data/data_util.py
    3. Testing of both of the above systems, in the notebook called image/data/testing_data.ipynb
    4. Testing of BEiT model on some sample data, in image/data/transformers.ipynb
3. Text (Hao Chen):
    1. Dataset wrapper class, applies transforms needed in text/dataset.py
    2. Found text tokenizer (RoBERTa) in text/dataset.py
    3. Masking function of the dataset wrapper in text/dataset.py
    4. Trying out both of the above systems

