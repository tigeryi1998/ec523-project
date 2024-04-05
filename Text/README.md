Text

Data should be preprocessed following the language modeling format, i.e. each document should be separated by an empty line (only useful with --sample-break-mode complete_doc). Lines will be concatenated as a 1D text stream during training.

We'll use the WikiText-103 dataset to demonstrate how to preprocess raw text data with the GPT-2 BPE. Of course this dataset is quite small, so the resulting pretrained model will perform poorly, but it gives the general idea.
