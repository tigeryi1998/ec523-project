import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader

def load_glue_task(task_name):
    return load_dataset("glue", task_name)

def tokenize_paired_inputs(example, tokenizer, field1, field2, max_length=128):
    # Tokenizes two fields as a pair of inputs
    inputs = tokenizer(
        example[field1], example[field2], truncation=True, padding="max_length", max_length=max_length
    )
    return inputs

def tokenize_qqp(example, tokenizer, max_length=128):
    # Tokenize both questions as separate fields
    inputs = tokenizer(
        example['question1'], example['question2'], truncation=True, padding="max_length", max_length=max_length
    )
    return inputs
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

def prepare_task(task_name, tokenizer):
    dataset = load_glue_task(task_name)

    if task_name.lower() == "qnli":
        dataset = dataset.map(lambda x: tokenize_paired_inputs(x, tokenizer, "question", "sentence"), batched=True)
    elif task_name.lower() == "mrpc":
        dataset = dataset.map(lambda x: tokenize_paired_inputs(x, tokenizer, "sentence1", "sentence2"), batched=True)
    elif task_name.lower() == 'qqp':
        dataset = dataset.map(lambda x: tokenize_qqp(x, tokenizer), batched=True)
    else:
        dataset = dataset.map(lambda x: tokenizer(x['sentence'], truncation=True, padding="max_length", max_length=128), batched=True)

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset

def eval_sst2(model, tokenizer):
    task_name = "sst2"
    dataset = prepare_task(task_name, tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./results/{task_name}',
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    print(f"Results for {task_name}:")
    print(results)

def eval_qnli(model, tokenizer):
    task_name = "qnli"
    dataset = prepare_task(task_name, tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./results/{task_name}',
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    print(f"Results for {task_name}:")
    print(results)

def eval_mrpc(model, tokenizer):
    task_name = "mrpc"
    dataset = prepare_task(task_name, tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./results/{task_name}',
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    print(f"Results for {task_name}:")
    print(results)

def eval_qqp(model, tokenizer):
    task_name = "qqp"
    dataset = prepare_task(task_name, tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./results/{task_name}',
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    print(f"Results for {task_name}:")
    print(results)

def eval_cola(model, tokenizer):
    task_name = "cola"
    dataset = prepare_task(task_name, tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./results/{task_name}',
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    print(f"Results for {task_name}:")
    print(results)

def main():
    model_checkpoint = '/projectnb/dl523/students/ha0chen/data2vec-pytorch-test/text/checkpoints/roberta-pretrain/5_10.pt'  # Replace with the .pt file path
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    task_functions = {
        "sst2": eval_sst2,
        "qnli": eval_qnli,
        "mrpc": eval_mrpc,
        "qqp": eval_qqp,
        "cola": eval_cola,
    }

    selected_task = "qqp" 

    if selected_task.lower() in task_functions:
        task_functions[selected_task.lower()](model, tokenizer)
    else:
        print(f"Task {selected_task} is not recognized. Please choose from: {list(task_functions.keys())}")

if __name__ == "__main__":
    main()
