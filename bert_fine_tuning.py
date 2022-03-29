import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer, DataCollatorWithPadding
from transformers import get_scheduler
import json
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AdamW
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gan_bert import comment_preprocessing
import csv

def tokenize_function(example):
    return tokenizer(example["messages"], max_length=64, padding="max_length", truncation=True)

def load_data():
    with open('data/fb_comments_labelled_all.json') as f:
        data = json.load(f)

    data_dict = {'messages': [], 'labels': []}
    for d in data:
        if d['sentiment_label'] != '0':
            data_dict['messages'].append("".join(comment_preprocessing(d['message'])))
            label = 0 if float(d['sentiment_label'].replace("−", "-")) < 0 else 1
            data_dict['labels'].append(label)

    X, y = np.array(data_dict['messages']), np.array(data_dict['labels'])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    train_dict = {
        "messages": X_train,
        "labels": y_train
    }

    test_dict = {
        "messages": X_test,
        "labels": y_test
    }

    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    return tokenized_train, tokenized_test

def data_formatting(tokenized_train, tokenized_test):
    tokenized_train = tokenized_train.remove_columns(["messages"])
    tokenized_train.set_format("torch")

    tokenized_test = tokenized_test.remove_columns(["messages"])
    tokenized_test.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_train, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_test, batch_size=batch_size, collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader


def training_loop():
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.train()
    for epoch in tqdm(range(num_epochs)):
        loss_epoch = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_epoch += loss.item()

        bert_loss.append(loss_epoch/len(train_dataloader))
        print("\nEvaluation for epoch %d"%epoch)
        eval_loop()

def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)

    print("Acc = %.2f, F1 = %s, recall = %s, precision = %s, b_loss = %.2f"
    % (accuracy, str(f1), str(recall), str(precision), bert_loss[-1]))
    print("%.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f" % (
    accuracy, f1[1], f1[0], recall[1], recall[0], precision[1], precision[0]))
    accuracy_t.append(accuracy)
    recall_t.append(recall)
    precision_t.append(precision)
    metrics.append([accuracy, f1, recall, precision])

def eval_loop():
    y_pred = []
    y_true = []
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_pred += list(predictions.cpu().detach().numpy())
        y_true += list(batch["labels"].cpu().detach().numpy())
    print_metrics(y_true, y_pred)

def log_training():
    with open('eval_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics)

if __name__ == "__main__":
    batch_size = 8
    num_epochs = 6
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    bert_loss = []
    accuracy_t = []
    precision_t = []
    recall_t = []
    metrics = [["accuracy", "f1", "recall", "precision"]]

    checkpoint = "KB/bert-base-swedish-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_train, tokenized_test = load_data()
    train_dataloader, eval_dataloader = data_formatting(tokenized_train, tokenized_test)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    training_loop()

    torch.save(model.state_dict(), 'models/bert_fine_tuned.pth')


    plt.plot(bert_loss, label="bert")
    plt.legend()
    plt.title("Training loss")
    plt.savefig("loss_fine_b.png")
    plt.show()


    plt.plot(accuracy_t, label="accuracy")
    plt.plot(precision_t, label="precision")
    plt.plot(recall_t, label="recall")
    plt.legend()
    plt.title("Evaluation metrics during training")
    plt.savefig("metrics_fine_b.png")
    plt.show()

    log_training()
