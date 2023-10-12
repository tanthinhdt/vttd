import pandas as pd 
import pyarrow as pa
import pyarrow.dataset as ds
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


train = pd.read_csv(r"data\processed\dataset_without_stopwords\train_without_stopwords.csv")
dev = pd.read_csv(r"data\processed\dataset_without_stopwords\dev_without_stopwords.csv")
test = pd.read_csv(r"data\processed\dataset_without_stopwords\test_without_stopwords.csv")


train.columns = ["text", "label"]
dev.columns = ["text", "label"]
test.columns = ["text", "label"]


train = Dataset(pa.Table.from_pandas(train))
dev = Dataset(pa.Table.from_pandas(dev))
test = Dataset(pa.Table.from_pandas(test))


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base",token = "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")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train = train.map(preprocess_function, batched=True)
tokenized_dev = dev.map(preprocess_function, batched=True)
tokenized_test = test.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    metric = evaluate.combine([
        evaluate.load("f1", average="micro"),
        evaluate.load("precision", average="micro"),
        evaluate.load("recall", average="micro")
    ])
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis = -1)
    return metric.compute(predictions=preds, references = labels)


id2label = {0: "CLEAN", 1: "TOXIC"}
label2id = {"CLEAN": 0, "TOXIC": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",token = "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", num_labels=2, id2label=id2label, label2id=label2id
)


training_args = TrainingArguments(
    output_dir="my_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()