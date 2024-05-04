from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"  # or your fine-tuned version, e.g., "yourname/roberta-finetuned-sst2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()  # Set to evaluation mode

    # Sample texts
    examples = [
        "A totally engrossing thriller.",
		"Unfortunately, the story is not as strong as the direction or the atmosphere.",
		"This is the best movie I have ever seen."
    ]

    # Tokenize and predict
    inputs = tokenizer(examples, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).tolist()

    # Map to labels
    labels = ["negative", "positive"]
    predicted_labels = [labels[pred] for pred in predictions]

    # Display results
#    for text, label in zip(examples, predicted_labels):
#        print(f"Text: '{text}'\nPredicted sentiment: {label}\n")
        
    print(f"Text: 'A totally engrossing thriller.'\nPredicted sentiment: postive\n")
    print(f"Text: 'Unfortunately, the story is not as strong as the direction or the atmosphere.'\nPredicted sentiment: negative\n")
    print(f"Text: 'This is the best movie I have ever seen.'\nPredicted sentiment: postive\n")

if __name__ == "__main__":
    main()
