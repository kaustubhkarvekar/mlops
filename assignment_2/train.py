import torch, wandb
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from data import prepare_data
from utils import MyDataset, create_label_maps, compute_metrics

model_name = "distilbert-base-cased"
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_encodings, test_encodings, train_labels, test_labels = prepare_data(model_name, max_length)
label2id, id2label = create_label_maps(train_labels)

train_labels_encoded = [label2id[y] for y in train_labels]
test_labels_encoded  = [label2id[y] for y in test_labels]

train_dataset = MyDataset(train_encodings, train_labels_encoded)
test_dataset  = MyDataset(test_encodings, test_labels_encoded)

model = DistilBertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
).to(device)

wandb.init(project="mlops-assignment2", name="distilbert-run-1")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="distilbert-run-1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

