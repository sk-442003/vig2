# scripts/train_text.py
import argparse
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_metric.compute(predictions=preds, references=labels, average="macro")}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--model_out", default="models/text_model")
    parser.add_argument("--model_name", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    dataset = load_dataset("csv", data_files={"train": args.train_file, "validation": args.val_file})
    print(f"Loaded dataset splits: train={len(dataset['train'])} validation={len(dataset['validation'])}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    dataset = dataset.map(preprocess, batched=True)
    label_list = sorted(list(set(dataset["train"]["label"])))
    num_labels = len(label_list)
    label_to_id = {l:i for i,l in enumerate(label_list)}
    print(f"Labels: {label_list}, num_labels={num_labels}")
    def label_map(ex):
        # ensure both 'labels' (standard for HF Trainer) and 'label' (collator compatibility) are numeric
        idx = label_to_id[ex["label"]]
        return {"labels": idx, "label": idx}
    dataset = dataset.map(label_map)
    # show a few samples to confirm labels are numeric
    print('Sample after label mapping (first 3):', dataset['train'][:3])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    try:
        training_args = TrainingArguments(
            output_dir=args.model_out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )
    except TypeError:
        # Fallback for older transformers versions that don't accept newer args
        training_args = TrainingArguments(
            output_dir=args.model_out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size
        )


    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.model_out)
    # save label map
    import json, os
    os.makedirs(args.model_out, exist_ok=True)
    with open(f"{args.model_out}/label_map.json","w") as f:
        json.dump(label_to_id, f)

if __name__ == "__main__":
    main()