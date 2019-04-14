import pandas as pd
import numpy as np
import gensim
import torch
import pickle
import random
from torch import nn
from torch.autograd import Variable
import collections
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import matplotlib
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

with open('../data/subjects_small', 'r') as f:
    subs = f.read()

with open('../data/relations_small', 'r') as f:
    rels = f.read()

with open('../data/objects_small', 'r') as f:
    obs = f.read()

from string import punctuation


all_text = ''.join([c for c in subs if c not in punctuation])
subs = all_text.split('\n')

all_text = ''.join([c for c in rels if c not in punctuation])
rels = all_text.split('\n')

all_text = ''.join([c for c in obs if c not in punctuation])
obs = all_text.split('\n')

all_text = ' '.join(subs)
all_text += ' '.join(rels)
all_text += ' '.join(obs)
words = all_text.lower().split()

with open('dic.pkl','rb') as f:
 w2v_embed = pickle.load(f)

# changing here
words = list(set(words))
vocab_to_int = dict()

for i in range(len(words)):
    vocab_to_int.update({words[i]: i})
# from collections import Counter
# counts = Counter(words)
# vocab = sorted(counts, key=counts.get, reverse=True)
# vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

sub_ints = []
for each in subs:
    each = each.lower()
    sub_ints.append([vocab_to_int[word] for word in each.split()])


rel_ints = []
for each in rels:
    each = each.lower()
    rel_ints.append([vocab_to_int[word] for word in each.split()])

ob_ints = []
for each in obs:
    each = each.lower()
    ob_ints.append([vocab_to_int[word] for word in each.split()])

from collections import Counter

sub_lens = Counter([len(x) for x in sub_ints])
rel_lens = Counter([len(x) for x in rel_ints])
ob_lens = Counter([len(x) for x in ob_ints])
print("Zero-length sub {}".format(sub_lens[0]))
print("Maximum sub length: {}".format(max(sub_lens)))

print("Zero-length rel: {}".format(rel_lens[0]))
print("Maximum rel length: {}".format(max(rel_lens)))

print("Zero-length obj: {}".format(ob_lens[0]))
print("Maximum obj length: {}".format(max(ob_lens)))


sub_lens1 = [len(x) for x in sub_ints]
rel_lens1 = [len(x) for x in rel_ints]
ob_lens1 = [len(x) for x in ob_ints]

plt.plot(sub_lens1)
plt.title('Subject Length')
plt.show()
plt.savefig('sub_len.png')

plt.plot(rel_lens1)
plt.title('Relation Length')
plt.show()
plt.savefig('rel_len.png')

plt.plot(ob_lens1)
plt.title('Object Length')
plt.show()
plt.savefig('obj_len.png')

ts = []
tr = []
to = []

for i in range(len(subs)):
    if len(sub_ints[i])*len(rel_ints[i])*len(ob_ints[i]) > 0:
      ts.append(sub_ints[i])
      tr.append(rel_ints[i])
      to.append(ob_ints[i])

sub_ints = np.array(ts)
rel_ints = np.array(tr)
ob_ints = np.array(to)


from collections import Counter


sub_lens = Counter([len(x) for x in sub_ints])
print("Zero-length subs: {}".format(sub_lens[0]))
print("Maximum sub length: {}".format(max(sub_lens)))

rel_lens = Counter([len(x) for x in rel_ints])
print("Zero-length rels: {}".format(rel_lens[0]))
print("Maximum rel length: {}".format(max(rel_lens)))

ob_lens = Counter([len(x) for x in ob_ints])
print("Zero-length obs: {}".format(ob_lens[0]))
print("Maximum ob length: {}".format(max(ob_lens)))

mx_sub = max(sub_lens)
mx_rel = max(rel_lens)
mx_ob = max(ob_lens)

sub_seq_len = mx_sub
rel_seq_len = mx_rel
ob_seq_len = mx_ob
sub_features = np.zeros((len(sub_ints), sub_seq_len), dtype=int)
rel_features = np.zeros((len(rel_ints), rel_seq_len), dtype=int)
ob_features = np.zeros((len(ob_ints), ob_seq_len), dtype=int)

for i, row in enumerate(sub_ints):
    sub_features[i, -len(row):] = np.array(row)[:sub_seq_len]

for i, row in enumerate(rel_ints):
    rel_features[i, -len(row):] = np.array(row)[:rel_seq_len]

for i, row in enumerate(ob_ints):
    ob_features[i, -len(row):] = np.array(row)[:ob_seq_len]


split_frac = 0.7

split_index = int(split_frac * len(sub_features))

train_sub, val_sub = sub_features[:split_index], sub_features[split_index:]
train_rel, val_rel = rel_features[:split_index], rel_features[split_index:]
train_ob, val_ob = ob_features[:split_index], ob_features[split_index:]

sub_dim = train_sub.shape[1]
rel_dim = train_rel.shape[1]
obj_dim = train_ob.shape[1]

sub_len = val_sub.shape[0]
rel_len = val_rel.shape[0]
obj_len = val_ob.shape[0]

print(train_sub.shape)
print(train_rel.shape)
print(train_ob.shape)

spo_train = np.concatenate([train_sub, train_rel, train_ob], axis=-1)
spo_valid = np.concatenate([val_sub, val_rel, val_ob], axis=-1)

print(spo_train.shape)

neg_sub1 = np.zeros(train_sub.shape)
neg_rel1 = np.zeros(train_rel.shape)
neg_ob1 = np.zeros(train_ob.shape)

train_len = len(train_sub)

for i in range(train_len):
    if i < train_len//2:
        neg_sub1[i][:] = train_sub[np.random.randint(0, train_len)][:]
        neg_rel1[i][:] = train_rel[i][:]
        neg_ob1[i][:] = train_ob[i][:]
    else:
        neg_sub1[i][:] = train_sub[i][:]
        neg_rel1[i][:] = train_rel[i][:]
        neg_ob1[i][:] = train_ob[np.random.randint(0, train_len)][:]


spo_neg1 = np.concatenate([neg_sub1, neg_rel1, neg_ob1], axis=-1)

class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, weight_matrix, hidden_dim, num_layers, p_dropout, batch_size, sub_dim, rel_dim, obj_dim):

        super(Model, self).__init__()

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.hidden_size = hidden_dim


        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.initialize_embeddings(weight_matrix)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=p_dropout)

    def initialize_embeddings(self, weight_matrix):

        self.embedding.weight = nn.Parameter(weight_matrix)

    def forward(self, sub, rel, obj, batch_size):

        h0_sub = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0_sub = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        h0_rel = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0_rel = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        h0_obj = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0_obj = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        sub = self.embedding(sub)
        rel = self.embedding(rel)
        obj = self.embedding(obj)

        lstm_sub, _ = self.lstm(sub, (h0_sub, c0_sub))
        lstm_rel, _ = self.lstm(rel, (h0_rel, c0_rel))
        lstm_obj, _ = self.lstm(obj, (h0_obj, c0_obj))


        lstm_sub = self.dropout(lstm_sub)
        lstm_rel = self.dropout(lstm_rel)
        lstm_obj = self.dropout(lstm_obj)

        lstm_sub = lstm_sub[:, sub_dim-1:, :]
        lstm_rel = lstm_rel[:, rel_dim-1:, :]
        lstm_obj = lstm_obj[:, obj_dim-1:, :]

        # print(lstm_sub.size())
        # print(lstm_rel.size())
        # print(lstm_obj.size())
        #
        # # lstm_sub = lstm_sub.view(-1, lstm_sub.size(2))
        # # lstm_rel = lstm_rel.view(-1, lstm_rel.size(2))
        # # lstm_obj = lstm_obj.view(-1, lstm_obj.size(2))
        # #
        # # sub_out = self.linear(lstm_sub)
        # # rel_out = self.linear(lstm_rel)
        # # obj_out = self.linear(lstm_obj)
        #
        #
        # # print(sub_out.size())
        # # print(rel_out.size())
        # # print(obj_out.size())
        score = torch.sum(lstm_sub + lstm_rel - lstm_obj, -1)
        return score.view(-1,1)

batch_size = 50

spo_train = torch.LongTensor(spo_train).to(device)
spo_valid = torch.LongTensor(spo_valid).to(device)
spo_neg1 = torch.LongTensor(spo_neg1).to(device)

# spo_train = torch.cat((spo_train, spo_neg1), dim=0)
# print(spo_train.size())
train_dataset1 = TensorDataset(spo_train, torch.zeros(spo_train.size(0)))
valid_dataset = DataLoader(TensorDataset(spo_valid, torch.zeros(spo_valid.size(0))), batch_size=batch_size, shuffle=True, drop_last=True)
neg_dataset1 = TensorDataset(spo_neg1, torch.zeros(spo_neg1.size(0)))

train_dataset = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
neg_dataset = DataLoader(neg_dataset1, batch_size=batch_size, shuffle=True, drop_last=True)

p_dropout = 0.5
n_words = len(vocab_to_int) + 1
embed_size = 300
weights_matrix = np.zeros((n_words,embed_size), dtype=np.float)
hidden_size = 512
num_layers = 1


for i, word in enumerate(words):
    try:
        weights_matrix[int(i)] = w2v_model[word]
    except KeyError:
        weights_matrix[int(i)] = np.random.normal(scale=0.6, size=(embed_size, ))

weights_matrix = torch.from_numpy(weights_matrix).float()
model = Model(n_words, embed_size, weights_matrix, hidden_size, num_layers, p_dropout, batch_size, sub_dim, rel_dim, obj_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=5,factor=0.5)

def train(spo, sno):

    zero = torch.FloatTensor([0.0])
    sub, pred, obj = torch.split(spo, [sub_dim, rel_dim, obj_dim], dim=1)
    neg_sub, neg_pred, neg_obj = torch.split(sno, [sub_dim, rel_dim, obj_dim], dim=1)
    criterion = lambda pos, neg : torch.sum(torch.max(Variable(zero).to(device), 1 - pos + neg))
    optimizer.zero_grad()

    pos_score = model(Variable(sub).to(device), Variable(pred).to(device), Variable(obj).to(device), batch_size)
    neg_score = model(Variable(neg_sub).to(device), Variable(neg_pred).to(device), Variable(neg_obj).to(device), batch_size)
    loss = criterion(pos_score, neg_score)
    loss.backward()
    optimizer.step()

    return loss.item()


def hitsatk_transe(spo, k):

    total = 0.0

    s, p, o = torch.split(spo, [sub_dim, rel_dim, obj_dim], dim=1)
    model.eval()
    #Looping through all the subjects
    for i1 in range(len(s)):

        s1 = s[i1][:]
        p1 = p[i1][:]
        print(o.size())
        arr = np.arange(0,o.size(0)).tolist()

        #Taking 100 random samples including the actual object
        arr_list = random.sample(arr, 99)
        o1 = torch.zeros((100, o.size(1)), dtype = torch.long)
        for l in range(len(arr_list)):
            o1[l][:] = o[arr_list[l]][:]
        o1[99][:] = o[i1][:]
        o1 = o1.to(device)

        print(o1.size())
        score = dict()
        #Looping through the 1000 samples to get the score of each sample
        obj_val = o1.view(100,-1)
        sub_val = s1.repeat(100).view(100,-1)
        pred_val = p1.repeat(100).view(100,-1)
        output = model(Variable(sub_val), Variable(pred_val), Variable(obj_val), 100)

        for m in range(len(output)):
            score[m] = output[m]

            #Sorting according to scores and getting the indexes of those scores
        sorted_x = sorted(score.items(), key=lambda kv: kv[1])
        sorted_dict = collections.OrderedDict(sorted_x)
        sorted_key = list(sorted_dict.keys())

            #Checking for hits in top k indexes
        for a in range(k):
            out = sorted_key[a]
            val1 = o1[out][:].cpu().numpy()
            val2 = o[i1][:].cpu().numpy()
            if (val1 == val2).all():
                total += 1.0
        print("Total ", i1, " : ",total)
    return total/len(o)

def run_transe_validation():

    hits1 = hitsatk_transe(spo_valid, 1)
    hits10 = hitsatk_transe(spo_valid, 10)
  #  hits100 = hitsatk_transe(spo_valid, 100)

    print("Validation hits@1: %f", hits1)
    print("Validation hits@10: %f", hits10)
  #  print("Validation hits@100: %f", hits100)

def run_transe(num_epochs):

    for i in range(num_epochs):

        total_batches = len(train_dataset)
        epoch_loss = 0.0
        train_iter = iter(train_dataset)
        neg_iter = iter(neg_dataset)

        for j in range(len(train_dataset)):
            p,z_p = next(train_iter)
            n,z_n = next(neg_iter)
            epoch_loss += train(p, n)

        scheduler.step(epoch_loss/total_batches)
        print ("Epoch %d Total loss: %f" % (i+1, epoch_loss/total_batches))
        # if (i+1) % 10 == 0:
        #     run_transe_validation()


if __name__ == "__main__":

    run_transe(200)
    torch.save(model,"model_transE_lstm1.pt")
    model = torch.load("model_transE_lstm.pt", map_location=lambda storage, loc: storage)
    run_transe_validation()
