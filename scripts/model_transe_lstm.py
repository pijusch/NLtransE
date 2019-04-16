import numpy as np
import gensim
import torch
import random
from torch import nn
from torch.autograd import Variable
import collections
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Loading pretrained Google Word2Vec Model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

#Small Dataset read
with open('../data/subjects_small', 'r') as f:
    subs = f.read()

with open('../data/relations_small', 'r') as f:
    rels = f.read()

with open('../data/objects_small', 'r') as f:
    obs = f.read()

from string import punctuation


#Data Preprocessing

#Removing punctuations and lowercasing
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

# Encoding words into numbers
words = list(set(words))
vocab_to_int = dict()

for i in range(len(words)):
    vocab_to_int.update({words[i]: i})

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

#Calculating max length of subject, object, relation
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

# Zero padding
sub_features = np.zeros((len(sub_ints), sub_seq_len), dtype=int)
rel_features = np.zeros((len(rel_ints), rel_seq_len), dtype=int)
ob_features = np.zeros((len(ob_ints), ob_seq_len), dtype=int)

for i, row in enumerate(sub_ints):
    sub_features[i, -len(row):] = np.array(row)[:sub_seq_len]

for i, row in enumerate(rel_ints):
    rel_features[i, -len(row):] = np.array(row)[:rel_seq_len]

for i, row in enumerate(ob_ints):
    ob_features[i, -len(row):] = np.array(row)[:ob_seq_len]


# Splitting into training and validation sets
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

spo_train = np.concatenate([train_sub, train_rel, train_ob], axis=-1)
spo_valid = np.concatenate([val_sub, val_rel, val_ob], axis=-1)

# Negative Samples
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


#Model Class
class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, weight_matrix, hidden_dim, num_layers, p_dropout, batch_size, sub_dim, rel_dim, obj_dim):

        super(Model, self).__init__()

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.hidden_size = hidden_dim
        self.sub_dim = sub_dim
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim

        self.embedding_s = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_p = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_o = nn.Embedding(vocab_size, embedding_dim)
        self.initialize_embeddings(weight_matrix)
        self.embedding_s.weight.requires_grad = False
        self.embedding_p.weight.requires_grad = False
        self.embedding_o.weight.requires_grad = False

        self.lstm_s = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.lstm_p = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.lstm_o = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=p_dropout)

    def initialize_embeddings(self, weight_matrix):

        self.embedding_s.weight = nn.Parameter(weight_matrix)
        self.embedding_p.weight = nn.Parameter(weight_matrix)
        self.embedding_o.weight = nn.Parameter(weight_matrix)

    def forward(self, sub, rel, obj, batch_size):

        h0_sub = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0_sub = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        h0_rel = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0_rel = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        h0_obj = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0_obj = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        sub = self.embedding_s(sub)
        rel = self.embedding_p(rel)
        obj = self.embedding_o(obj)

        lstm_sub, _ = self.lstm_s(sub, (h0_sub, c0_sub))
        lstm_rel, _ = self.lstm_p(rel, (h0_rel, c0_rel))
        lstm_obj, _ = self.lstm_o(obj, (h0_obj, c0_obj))


        lstm_sub = self.dropout(lstm_sub)
        lstm_rel = self.dropout(lstm_rel)
        lstm_obj = self.dropout(lstm_obj)

        lstm_sub = lstm_sub[:, self.sub_dim-1:, :]
        lstm_rel = lstm_rel[:, self.rel_dim-1:, :]
        lstm_obj = lstm_obj[:, self.obj_dim-1:, :]

        score = torch.sum((lstm_sub + lstm_rel - lstm_obj)**2, -1)
        return score.view(-1,1)


batch_size = 200

spo_train = torch.LongTensor(spo_train).to(device)
spo_valid = torch.LongTensor(spo_valid).to(device)
spo_neg1 = torch.LongTensor(spo_neg1).to(device)


train_dataset1 = TensorDataset(spo_train, torch.zeros(spo_train.size(0)))
valid_dataset = DataLoader(TensorDataset(spo_valid, torch.zeros(spo_valid.size(0))), batch_size=batch_size, shuffle=True, drop_last=True)
neg_dataset1 = TensorDataset(spo_neg1, torch.zeros(spo_neg1.size(0)))

train_dataset = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
neg_dataset = DataLoader(neg_dataset1, batch_size=batch_size, shuffle=True, drop_last=True)

p_dropout = 0.5
n_words = len(vocab_to_int) + 1
embed_size = 300
weights_matrix = np.zeros((n_words,embed_size), dtype=np.float)
hidden_size = 300
num_layers = 1


for i, word in enumerate(words):
    try:
        weights_matrix[int(i)] = w2v_model[word]
    except KeyError:
        weights_matrix[int(i)] = np.array([0] * embed_size)

weights_matrix = torch.from_numpy(weights_matrix).float()
model = Model(n_words, embed_size, weights_matrix, hidden_size, num_layers, p_dropout, batch_size, sub_dim, rel_dim, obj_dim).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=5,factor=0.1)


# Training the model
def train(spo, sno):

    zero = torch.FloatTensor([0.0])
    sub, pred, obj = torch.split(spo, [sub_dim, rel_dim, obj_dim], dim=1)
    neg_sub, neg_pred, neg_obj = torch.split(sno, [sub_dim, rel_dim, obj_dim], dim=1)
    criterion = lambda pos, neg : torch.sum(torch.max(Variable(zero).to(device), 1 + pos - neg))
    optimizer.zero_grad()

    pos_score = model(Variable(sub).to(device), Variable(pred).to(device), Variable(obj).to(device), batch_size)
    neg_score = model(Variable(neg_sub).to(device), Variable(neg_pred).to(device), Variable(neg_obj).to(device), batch_size)
    loss = criterion(pos_score, neg_score)
    loss.backward()
    optimizer.step()

    return loss.item()


#Validation hits
def hitsatk_transe(spo, k):

    total = 0.0

    s, p, o = torch.split(spo, [sub_dim, rel_dim, obj_dim], dim=1)

    o2 = torch.unique(o, dim=0)
    model.eval()
    #Looping through all the subjects
    for i1 in range(len(s)):

        s1 = s[i1][:]
        p1 = p[i1][:]
        arr = np.arange(0,o2.size(0)).tolist()

        #Taking 100 random samples including the actual object
        arr_list = random.sample(arr, 999)
        o1 = torch.zeros((1000, o2.size(1)), dtype = torch.long)
        for l in range(len(arr_list)):
            o1[l][:] = o2[arr_list[l]][:]
        o1[999][:] = o[i1][:]
        o1 = o1.to(device)

        score = dict()
        #Looping through the 100 samples to get the score of each sample
        obj_val = o1.view(1000,-1)
        sub_val = s1.repeat(1000).view(1000,-1)
        pred_val = p1.repeat(1000).view(1000,-1)
        output = model(Variable(sub_val), Variable(pred_val), Variable(obj_val), 1000)

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
        if i1%100 == 0:
            print("Total ", i1, " : ",total)
    print("\n")
    return total/len(o)


def run_transe_validation():

    print("\nHits@1:")
    hits1 = hitsatk_transe(spo_valid, 1)
    print("\nHits@10:")
    hits10 = hitsatk_transe(spo_valid, 10)

    print("Validation hits@1: ", hits1)
    print("Validation hits@10: ", hits10)
    print("\n")

    return hits1, hits10


def run_transe(num_epochs):

    hits1 = 0.0
    for i in range(num_epochs):
        model.train()
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
        if (i+1) % 10 == 0:
            temp_hits1, temp_hits10 = run_transe_validation()
            if temp_hits1 >= hits1:
                hits1 = temp_hits1
                torch.save(model, "model_transe_lstm_300_adam.pt")


if __name__ == "__main__":

    run_transe(200)
    model = torch.load("model_transe_lstm_300_adam.pt")
    run_transe_validation()
