import torch
import json
import csv
from tqdm import tqdm
import numpy as np
import re
import random
import gensim.downloader as api

word_vectors = api.load("word2vec-google-news-300")
word_vectors.save_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    if words:
        rand_idx = random.randint(0, len(words) - 1)
        words[rand_idx] = "unk"
    text = ' '.join(words)
    return text

file_path = "data/train.csv"
file1 = open(file_path, "r")
data = csv.reader(file1)

max_sentences = 20000

with open(file_path, "r") as file1:
    data = csv.reader(file1)
    corp_list = []
    corp_str = "sos "
    for i, row in enumerate(data):
        if i >= max_sentences:
            break
        string = preprocess(row[1])
        corp_list.append(string)
        corp_str += string + " sos eos "

final_list = []
corp_words = corp_str.split()
# make sentences of 32 words
for i in range(0, len(corp_words), 32):
    final_list.append(corp_words[i:i+32])
print(len(final_list))


word2idx = {}
idx2word = {}
word_set = set()
for i, word in enumerate(corp_words):
    word_set.add(word)
i = 0
for word in word_set:
    if word not in word2idx.keys():
        word2idx[word] = i
        idx2word[i] = word
        i+=1

from torch import nn, optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Making Dataset...")
class dual_dataset(torch.utils.data.Dataset):
    def __init__(self, sent_list, word2idx, idx2word):
        self.sent_list = sent_list
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.forward_labels, self.forward_targets = self.forward()
        self.backward_labels, self.backward_targets = self.backward()
    def forward(self):
        labels = []
        targets = []
        for sent in self.sent_list:
            loc_labels = []
            loc_targets = []
            for i in range(1, len(sent)-1):
                loc_labels.append(self.word2idx[sent[i]])
                loc_targets.append(self.word2idx[sent[i+1]])
            labels.append(loc_labels)
            targets.append(loc_targets)
        return labels, targets
    def backward(self):
        labels = []
        targets = []
        for_labels = self.forward_labels
        for_targets = self.forward_targets
        for label in for_labels:
            rev_label = label[::-1]
            labels.append(rev_label)
        for target in for_targets:
            rev_target = target[::-1]
            targets.append(rev_target)
        return labels, targets
    def __len__(self):
        return len(self.sent_list)

    def __getitem__(self, idx):
        if len(self.forward_labels[idx]) != len(self.forward_labels[0]):
            idx = idx -1
        forward_labels = torch.LongTensor(self.forward_labels[idx])
        backward_labels = torch.LongTensor(self.backward_labels[idx])
        forward_targets = torch.LongTensor(self.forward_targets[idx])
        backward_targets = torch.LongTensor(self.backward_targets[idx])
        return forward_labels, backward_labels, forward_targets, backward_targets


my_dataset = dual_dataset(final_list, word2idx, idx2word)
class lstm(nn.Module):
    def __init__(self, vocab_size):
        super(lstm, self).__init__()
        self.vocab_size = vocab_size
        # lstm with 2 stacks
        num_stacks = 2
        self.embeddings = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, 300, 1, batch_first=True)
        self.lstm1 = nn.LSTM(300, 300, 1, batch_first=True)
        self.linear = nn.Linear(300, vocab_size)
    def forward(self, x):
        embed = self.embeddings(x)
        x1, _ = self.lstm(embed)
        x2, _ = self.lstm1(x1)
        x = self.linear(x2)
        return x, (embed, x1, x2)


forward_model = lstm(len(word2idx)).to(device)
backward_model = lstm(len(word2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(forward_model.parameters(), lr=0.001)
optimizer2 = optim.Adam(backward_model.parameters(), lr=0.001)
dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=50, shuffle=True)

torch.save(word2idx, "word2idx.pt")
torch.save(idx2word, "idx2word.pt")
print("Done making the dictionaries.")

n_epochs = 15

print("Starting Training...")
for epoch in range(n_epochs):
    forward_model.train()
    backward_model.train()
    total_forward_loss = 0.0
    total_backward_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Epoch {}".format(epoch))
    for batch, (forward_labels, backward_labels, forward_targets, backward_targets) in progress_bar:
        forward_targets = forward_targets.to(device)
        backward_targets = backward_targets.to(device)
        forward_labels = forward_labels.to(device)
        backward_labels = backward_labels.to(device)
        # change the shape of forward_vcts and backward_vcts to batch-size, seq_len, embedding_dim
        forward_targets = forward_targets.view(forward_targets.shape[1], forward_targets.shape[0])
        backward_targets = backward_targets.view(backward_targets.shape[1], backward_targets.shape[0])
        forward_labels = forward_labels.view(forward_labels.shape[1], forward_labels.shape[0])
        backward_labels = backward_labels.view(backward_labels.shape[1], backward_labels.shape[0])
        # forward pass
        optimizer1.zero_grad()
        forward_out,_  = forward_model(forward_labels)
        forward_out = forward_out.view(forward_out.shape[0]*forward_out.shape[1], forward_out.shape[2])
        forward_targets = forward_targets.view(forward_targets.shape[0]*forward_targets.shape[1])
        forward_loss = loss_fn(forward_out, forward_targets)
        forward_loss.backward()
        optimizer1.step()
        total_forward_loss += forward_loss.item()
        # backward pass
        optimizer2.zero_grad()
        backward_out,_ = backward_model(backward_labels)
        backward_out = backward_out.view(backward_out.shape[0]*backward_out.shape[1], backward_out.shape[2])
        backward_targets = backward_targets.view(backward_targets.shape[0]*backward_targets.shape[1])
        backward_loss = loss_fn(backward_out, backward_targets)
        backward_loss.backward()
        optimizer2.step()
        total_backward_loss += backward_loss.item()
        forward_peplexity = torch.exp(forward_loss)
        backward_peplexity = torch.exp(backward_loss)
        progress_bar.set_postfix(forward_loss=forward_loss.item(), backward_loss=backward_loss.item(), avg_forward_loss=total_forward_loss/(batch+1), avg_backward_loss=total_backward_loss/(batch+1))

    average_forward_loss = total_forward_loss / len(dataloader)
    average_backward_loss = total_backward_loss / len(dataloader)

    #Print epoch-level loss
    print("Epoch: {}, Average Forward Loss: {:.4f}, Average Backward Loss: {:.4f}".format(epoch, average_forward_loss, average_backward_loss))

torch.save(forward_model, f"forward_model.pt")
torch.save(backward_model, f"backward_model.pt")