import torch
from transformers import AutoModel,AutoTokenizer, DataCollatorWithPadding, AutoConfig
from transformers import get_scheduler
import json
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm.auto import tqdm
import re
from transformers import AdamW
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=512, hidden_size=512):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, value):
        return self.generator(value)

class Discriminator(nn.Module):
    def __init__(self, input_size=512, output_size=3, hidden_size=512):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(0.1)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_layer = nn.Dropout(0.1)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, representation):
        representation = self.input_dropout(representation)
        hidden_layer = self.hidden_layer(representation)
        activation_layer = self.activation(hidden_layer)
        last_rep = self.dropout_layer(activation_layer)
        logits = self.output_layer(last_rep)
        probs = self.output_activation(logits)
        return last_rep, logits, probs

def tokenize_function(example):
    return tokenizer(example["messages"], max_length=64, padding="max_length", truncation=True)

def remove_email(text):
    return re.sub("(([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\." \
                  "([a-z]{2,6}(?:\.[a-z]{2})?))(?![^<]*>)", "<email>", text)
def remove_url(text):
    return re.sub("(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])", "<url>", text)

def lemmatize(text):
    words = word_tokenize(text)
    new_w = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(new_w)

def stemming(text):
    words = word_tokenize(text)
    new_w = [ps.stem(w) for w in words]
    return " ".join(new_w)

def comment_preprocessing(text):
    text = remove_email(text)
    text = remove_url(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text


def load_balanced_data():
    with open('data/fb_comments_labelled.json') as f:
        data_labelled = json.load(f)

    data_dict_label = {'messages': [], 'labels': []}
    for d in tqdm(data_labelled):
        if d['sentiment_label'] != '0':
            label = 0 if float(d['sentiment_label'].replace("−", "-")) < 0 else 1
            data_dict_label['messages'].append("".join(comment_preprocessing(d['message'])))
            data_dict_label['labels'].append(label)


    X, y = np.array(data_dict_label['messages']), np.array(data_dict_label['labels'])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = list(X[train_index]), list(X[test_index])
        y_train, y_test = list(y[train_index]), list(y[test_index])

    if use_unlabelled:
        with open('data/fb_comments_unlabelled.json') as f:
            data_unlabelled = json.load(f)
        data_dict_unlabel = {'messages': [], 'labels': []}
        for d in tqdm(data_unlabelled[:un_amount]):
            label = unlabel_label
            data_dict_unlabel['messages'].append("".join(comment_preprocessing(d['message'])))
            data_dict_unlabel['labels'].append(label)
        X_train += data_dict_unlabel['messages']
        y_train += data_dict_unlabel['labels']

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
    model_params = [p for p in model.parameters()]
    d_params = model_params + [p for p in discriminator_net.parameters()]
    optimizer_g = AdamW(generator_net.parameters(), lr=5e-5)
    optimizer_d = AdamW(d_params, lr=5e-5)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler_g = get_scheduler(
        "linear",
        optimizer=optimizer_g,
        num_warmup_steps=warmup_proportion*num_training_steps,
        num_training_steps=num_training_steps
    )
    lr_scheduler_d = get_scheduler(
        "linear",
        optimizer=optimizer_d,
        num_warmup_steps=warmup_proportion*num_training_steps,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    generator_net.train()
    discriminator_net.train()
    for epoch in tqdm(range(num_epochs)):
        d_loss_epoch = 0
        g_loss_epoch = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            batch_labels = batch['labels']

            batch_sup = (batch_labels < unlabel_label).nonzero(as_tuple=True)
            batch_unsup = (batch_labels == unlabel_label).nonzero(as_tuple=True)
            masked_batch = torch.tensor([True if i<unlabel_label else False for i in batch_labels])

            del batch['labels']

            outputs = model(**batch)
            hidden_reps = outputs[-1]

            noise = torch.zeros(batch['input_ids'].shape[0], noise_size, device=device).normal_(0, 1)

            generated = generator_net(noise)

            discriminator_input = torch.cat([hidden_reps, generated], dim=0)

            d_rep, d_logits, d_prob = discriminator_net(discriminator_input)

            d_rep_list = torch.split(d_rep, len(batch_labels))
            d_rep_real = d_rep_list[0]
            d_rep_fake = d_rep_list[1]

            d_logits_list = torch.split(d_logits, len(batch_labels))
            d_logits_real = d_logits_list[0]
            d_logits_fake = d_logits_list[1]

            d_prob_list = torch.split(d_prob, len(batch_labels))
            d_prob_real = d_prob_list[0]
            d_prob_fake = d_prob_list[1]

            if len(batch_sup) == 0:
                d_loss_sup = 0
            else:
                log_probs = F.log_softmax(d_logits_real[:, 0:-1], dim=-1)
                label2one_hot = torch.nn.functional.one_hot(batch_labels, output_size)
                per_example_loss = -torch.sum(label2one_hot[:, 0:-1] * log_probs, dim=-1)
                per_example_loss = torch.masked_select(per_example_loss, masked_batch.to(device))
                d_loss_sup = torch.div(torch.sum(per_example_loss.to(device)), len(batch_sup))

            d_loss_unsup_u = -torch.mean(torch.log(1 - d_prob_real[:, -1] + eps))
            d_loss_unsup_f = -torch.mean(torch.log(d_prob_fake[:, -1] + eps))
            d_loss = d_loss_sup + d_loss_unsup_u + d_loss_unsup_f


            g_loss_fm = torch.sum(torch.pow(torch.mean(d_rep_real, dim=0) - torch.mean(d_rep_fake, dim=0), 2))
            g_loss_uns = -torch.mean(torch.log(1-d_prob_fake[:, -1] + eps))
            g_loss = g_loss_fm + g_loss_uns


            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            g_loss.backward(retain_graph=True)
            d_loss.backward()

            optimizer_g.step()
            optimizer_d.step()

            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()
            progress_bar.update(1)

            if lr_scheduling:
                lr_scheduler_g.step()
                lr_scheduler_d.step()

        discrimiantor_loss.append(d_loss_epoch/len(train_dataloader))
        generator_loss.append(g_loss_epoch/len(train_dataloader))
        print("\nEvaluation for epoch %d"%epoch)
        eval_loop()

def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)

    print("Acc = %.2f, F1 = %s, recall = %s, precision = %s, d_loss = %.2f, g_loss = %.2f"
    % (accuracy, str(f1), str(recall), str(precision), discrimiantor_loss[-1], generator_loss[-1]))

    print("%.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f"%(accuracy, f1[2], f1[1], f1[0], recall[2], recall[1], recall[0], precision[2], precision[1], precision[0]))
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
            batch_labels = batch['labels']
            del batch['labels']
            outputs = model(**batch)
            hidden_reps = outputs[-1]
            d_rep, d_logits, d_prob = discriminator_net(hidden_reps)
        _, preds = torch.max(d_prob[:, :output_size-1], 1) # Avoid max with fake label
        y_pred += list(preds.cpu().detach().numpy())
        y_true += list(batch_labels.cpu().detach().numpy())
    print_metrics(y_true, y_pred)


def log_training():
    with open('eval_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics)


if __name__ == "__main__":
    eps = 1e-8
    batch_size = 8
    num_epochs = 10
    warmup_proportion = 0.1
    output_size = 3
    use_unlabelled = True
    unlabel_label = 2
    un_amount = 2000
    lr_scheduling = True
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    discrimiantor_loss = []
    generator_loss = []
    accuracy_t = []
    precision_t = []
    recall_t = []
    metrics = [["accuracy", "f1", "recall", "precision"]]

    checkpoint = "KB/bert-base-swedish-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_train, tokenized_test = load_balanced_data()
    train_dataloader, eval_dataloader = data_formatting(tokenized_train, tokenized_test)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    noise_size = 100
    config = AutoConfig.from_pretrained(checkpoint)
    hidden_size = int(config.hidden_size)

    generator_net = Generator(input_size=noise_size, output_size=hidden_size, hidden_size=hidden_size)
    discriminator_net = Discriminator(input_size=hidden_size, output_size=output_size, hidden_size=hidden_size)
    generator_net.to(device)
    discriminator_net.to(device)
    training_loop()

    torch.save(model.state_dict(), 'models/bert.pth')
    torch.save(discriminator_net.state_dict(), 'models/discriminator.pth')
    torch.save(generator_net.state_dict(), 'models/generator_net.pth')


    plt.plot(discrimiantor_loss, label="discriminator")
    plt.plot(generator_loss, label="generator")
    plt.legend()
    plt.title("Discriminator and generator loss")
    plt.savefig("loss.png")
    plt.show()


    plt.plot(accuracy_t, label="accuracy")
    plt.plot(precision_t, label="precision")
    plt.plot(recall_t, label="recall")
    plt.legend()
    plt.title("Evaluation metrics during training")
    plt.savefig("metrics.png")
    plt.show()


    log_training()
