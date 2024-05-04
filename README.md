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

## References
(data2vec examples)[https://github.com/arxyzan/data2vec-pytorch]

(data2vec official)[https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec]




