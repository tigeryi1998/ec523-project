# ec523-project: Re-implementing data2vec

This project is from Rohan Kumar, Sicheng (Tiger) Yi, Hao Chen, Shi Gu for EC523 Deep Learning, Spring 2024. 


## Prerequisite

Python3

Install the requirements:
```bash
pip install -r requirements.txt
```

## Modules
### **Text**

Before training, change the configuration in `text/configs/config.yaml`

Run the following code to start training,
```bash
python train.py --config text/configs/config.yaml 
```

For evaluating the trained model's performance on GLUE tasks, run the glue_test.py by entering the path of the model file and selecting the task you want to test.
```bash
python glue_test.py 
```

In the file, we prepared five tasks:
1. sst2 (Stanford Sentiment Treebank)
2. qnli (Question Natural Language Inference)
3. mrpc (Microsoft Research Paraphrase Corpus)
4. qqp (Quora Question Pairs)
5. cola (Corpus of Linguistic Acceptability)

You will need to change the selected_task variable to select task.

## References
[data2vec examples](https://github.com/arxyzan/data2vec-pytorch)

[data2vec official repo](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)




