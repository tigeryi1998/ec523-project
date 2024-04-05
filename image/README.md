
## Computer Vision

This model works as follows:
1) We are given some sample of images
2) For each image, we embed them as a series of 16x16 "blocks" using ViT-based method
3) The teacher model is given these embeddings and generates a representation
4) We mask the embeddings using block-wise masking
5) The student model is given the masked embeddings and generates a representation
6) We try to minimize the difference between the teacher and student model (i.e. the student "learns" from the teacher)
    1) the teacher's parameters are updated using EMA (exponential moving average)
    2) targets are built as the top k blocks of the teacher model for time-steps that are masked in student mode
