# -*- coding: utf-8 -*-
from __future__ import print_function
import collections
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Lambda
from keras.layers import LSTM, Multiply, Merge, add, subtract, Add, Subtract, Concatenate
from keras.optimizers import Adam, Adagrad
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling1D
import pandas as  pd
import numpy as np
import argparse
import gensim
import json
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pickle

#Validation hits
def hitsatk_transe(val_sub,val_rel,val_ob, k):

    total = 0.0


    o2 = np.unique(val_ob, axis=0)

    #Looping through all the subjects
    for i1 in range(len(val_sub)):

        s1 = val_sub[i1][:][:]
        p1 = val_rel[i1][:][:]
        arr = np.arange(0,o2.shape[0]).tolist()

        #Taking 100 random samples including the actual object
        arr_list = random.sample(arr, 99)
        o1 = np.zeros((100, o2.shape[1],o2.shape[2]), dtype = np.long)
        for l in range(len(arr_list)):
            o1[l][:][:] = o2[arr_list[l]][:][:]
        o1[99][:][:] = val_ob[i1][:][:]
        #
        # print(s1.shape)
        # print(o1.shape)
        # print(p1.shape)
        score = dict()
        #Looping through the 100 samples to get the score of each sample
    #    sdim1 = s1.shape[1]
    #    sdim2 = s1.shape[2]
        obj_val = o1.reshape(100,o1.shape[1],o1.shape[2])
        sub_val = s1.repeat(100).reshape(100,s1.shape[0],s1.shape[1])
        pred_val = p1.repeat(100).reshape(100,p1.shape[0],p1.shape[1])
        output = parallel_model.predict(x=[sub_val,pred_val,obj_val])
        # print(output)
        # print(output.shape)
        for m in range(len(output)):
            score[m] = output[m]

        #Sorting according to scores and getting the indexes of those scores
        sorted_x = sorted(score.items(), key=lambda kv: kv[1])
        sorted_dict = collections.OrderedDict(sorted_x)
        sorted_key = list(sorted_dict.keys())

            #Checking for hits in top k indexes
        for a in range(k):
            out = sorted_key[a]
            val1 = o1[out][:][:]
            val2 = val_ob[i1][:][:]
            if (val1 == val2).all():
                total += 1.0
        if i1%100 == 0:
            print("Total ", i1, " : ",total)
    print("\n")
    return total/len(val_ob)


def run_transe_validation():

    print("\nHits@1:")
    hits1 = hitsatk_transe(val_sub_e,val_rel_e,val_ob_e, 1)
    print("\nHits@10:")
    hits10 = hitsatk_transe(val_sub_e,val_rel_e,val_ob_e, 10)

    print("Validation hits@1: ", hits1)
    print("Validation hits@10: ", hits10)
    print("\n")


def customloss(y_pred,y_true):
 return y_pred-y_true
 #return K.l2_normalize(y_pred,axis=1)



class Metrics(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []

 def on_epoch_end(self, epoch, logs={}):
  val_predict = (np.asarray(parallel_model.predict([val_sent_e, val_claim_e]))).round()
  val_targ = val_y
  _val_f1 = f1_score(val_targ, val_predict)
  _val_recall = recall_score(val_targ, val_predict)
  _val_precision = precision_score(val_targ, val_predict)
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  print (' — val_f1: %f — val_precision: %f — val_recall %f' %( _val_f1, _val_precision, _val_recall))
  return

metrics = Metrics()



model = gensim.models.KeyedVectors.load_word2vec_format('/users/PAS1197/osu10552/NLtransE/scripts/GoogleNews-vectors-negative300.bin', binary=True)

#model = dict()

with open('../../data/subjects_total', 'r') as f:
    subs = f.read()

with open('../../data/relations_total', 'r') as f:
    rels = f.read()

with open('../../data/objects_total', 'r') as f:
    obs = f.read()

labels =np.array( pd.read_csv('../../data/labels',header=None))
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



#labels = np.array([1 if l == "Positive" else 0 for l in labels_org.split()])

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


# Filter out that review with 0 length
#claim_ints = [r for r in claim_ints if len(r) > 0]
#sent_ints = [r[0:500] for r in sent_ints if len(r) > 0]


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


split_frac = .9

split_index = int(split_frac * len(sub_features))

train_sub, val_sub = sub_features[:split_index], sub_features[split_index:]
train_rel, val_rel = rel_features[:split_index], rel_features[split_index:]
train_ob, val_ob = ob_features[:split_index], ob_features[split_index:]
train_lab,val_lab = labels[:split_index], labels[split_index:]


#split_frac = 1
#split_index = int(split_frac * len(val_claim))

#val_claim, test_claim = val_claim[:split_index], val_claim[split_index:]
#val_sent, test_sent = val_sent[:split_index], val_sent[split_index:]
#val_y, test_y = val_y[:split_index], val_y[split_index:]


n_words = len(vocab_to_int) + 1  # Add 1 for 0 added to vocab

embed_size = 300

w2v_embed = np.ndarray([n_words, embed_size])

for i in range(n_words - 1):
    if words[i] not in model:
        w2v_embed[vocab_to_int[words[i]]] = np.array([0] * embed_size)
    else:
        w2v_embed[vocab_to_int[words[i]]] = model[words[i]]

#with open('dic.pkl','wb') as f:
# pickle.dump(w2v_embed,f)

#with open('../dic.pkl','rb') as f:
# w2v_embed = pickle.load(f)

import random

'''
idx = random.sample(range(len(train_claim)), len(train_claim))

train_claim_s = []
train_sent_s = []
train_y_s = []

for i in idx:
    train_claim_s.append(train_claim[i])
    train_sent_s.append(train_sent[i])
    train_y_s.append(train_y[i])

train_claim = np.array(train_claim_s)
train_sent = np.array(train_sent_s)
train_y = np.array(train_y_s)
#test_claim = np.array(test_claim)
#test_sent = np.array(test_sent)
#test_y = np.array(test_y)
'''

train_sub_e = np.ndarray((len(train_sub), mx_sub, embed_size))
train_rel_e = np.ndarray((len(train_rel), mx_rel, embed_size))
train_ob_e = np.ndarray((len(train_ob), mx_ob, embed_size))

for i in range(len(train_sub)):
    for j in range(mx_sub):
        train_sub_e[i][j][:] = w2v_embed[train_sub[i][j]]

for i in range(len(train_rel)):
    for j in range(mx_rel):
        train_rel_e[i][j][:] = w2v_embed[train_rel[i][j]]

for i in range(len(train_ob)):
    for j in range(mx_ob):
        train_ob_e[i][j][:] = w2v_embed[train_ob[i][j]]

val_sub_e = np.ndarray((len(val_sub), mx_sub, embed_size))
val_rel_e = np.ndarray((len(val_rel), mx_rel, embed_size))
val_ob_e = np.ndarray((len(val_ob), mx_ob, embed_size))

for i in range(len(val_sub)):
    for j in range(mx_sub):
        val_sub_e[i][j][:] = w2v_embed[val_sub[i][j]]

for i in range(len(val_rel)):
    for j in range(mx_rel):
        val_rel_e[i][j][:] = w2v_embed[val_rel[i][j]]

for i in range(len(val_ob)):
    for j in range(mx_ob):
        val_ob_e[i][j][:] = w2v_embed[val_ob[i][j]]



hidden_size = 256
use_dropout = True
vocabulary = n_words

embedding_layer = Embedding(input_dim=vocabulary, output_dim=300)

lstm_out = 150

lstm1 = Sequential()
#model1.add(embedding_layer)
#model1.add(Embedding(vocabulary, embed_size, input_length=mx_sent))
lstm1.add(LSTM(lstm_out, return_sequences=False, input_shape=(mx_sub, embed_size)))
#model1.add(LSTM(embed_size, return_sequences=True))
#model1.add(GlobalAveragePooling1D())
#model1.add(TimeDistributed(Dense(1)))
#model1.add(LSTM(embed_size, return_sequences=False))
if use_dropout:
    lstm1.add(Dropout(0.5))
lstm1.add(Dense(lstm_out, activation='sigmoid', name='out1'))

lstm2 = Sequential()
#model2.add(embedding_layer)
#model2.add(Embedding(vocabulary, embed_size, input_length=mx_claim))
#model2.add(LSTM(embed_size, return_sequences=True))
lstm2.add(LSTM(lstm_out, return_sequences=False, input_shape=(mx_rel, embed_size)))
#model2.add(LSTM(embed_size, return_sequences=True))
#model2.add(LSTM(embed_size, return_sequences = False))
#model2.add(GlobalAveragePooling1D())
#model2.add(TimeDistributed(Dense(1)))
if use_dropout:
    lstm2.add(Dropout(0.5))
lstm2.add(Dense(lstm_out, activation='sigmoid', name='out2'))

lstm3 = Sequential()
#model2.add(embedding_layer)
#model2.add(Embedding(vocabulary, embed_size, input_length=mx_claim))
#model2.add(LSTM(embed_size, return_sequences=True))
lstm3.add(LSTM(lstm_out, return_sequences=False, input_shape=(mx_ob, embed_size)))
#model2.add(LSTM(embed_size, return_sequences=True))
#model2.add(LSTM(embed_size, return_sequences = False))
#model2.add(GlobalAveragePooling1D())
#model2.add(TimeDistributed(Dense(1)))
if use_dropout:
    lstm3.add(Dropout(0.5))
lstm3.add(Dense(lstm_out, activation='sigmoid', name='out3'))
lstm3.add(Lambda(lambda x: x * -1))

model = Sequential()
#model = Add()([lstm1,lstm2])
#model = Subtract()([model,lstm3])
model.add(Merge([Merge([lstm1, lstm2], mode='mul'),lstm3], mode='mul'))
#model.add(Lambda(lambda x: K.sum(x,axis=1),output_shape =[1]))
model.add(Dense(1, activation = 'sigmoid'))
#model.add(Merge([lstm1, lstm2], mode='sum'))
# model = Multiply()([model1.get_layer('out1').output,model2.get_layer('out2').output])

# model.add(TimeDistributed(Dense(vocabulary)))
# model.add(Activation('softmax'))

#optimizer = Adam()
# model1.compile(loss='mean_squared_error', optimizer='adam')
# parallel_model = multi_gpu_model(model, gpus=2)
parallel_model = model
parallel_model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001) , metrics=['acc'])
#parallel_model.compile(loss='mean_squared_error', optimizer='adam' )

print(model.summary())
print(lstm1.summary())
print(lstm2.summary())
print(lstm3.summary())
# checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 1
plot_model(parallel_model, to_file='model.png')
parallel_model.fit(x=[train_sub_e, train_rel_e,train_ob_e],y =train_lab, batch_size=64, epochs=num_epochs,validation_split=0.1)
#               validation_data=([val_sub_e,val_rel_e,val_ob_e],[0]*len(val_sub_e)))

#parallel_model.save("final_model.hdf5")
#parallel_model.load_weights("final_model.hdf5")
run_transe_validation()

#print(parallel_model.predict([val_sub_e, val_rel_e, val_ob_e]))
print(parallel_model.evaluate([train_sub_e,train_rel_e,train_ob_e],train_lab))
print(parallel_model.evaluate([val_sub_e,val_rel_e,val_ob_e],val_lab))
#parallel_model.save("final_model.hdf5")
val_pred_ur = np.asarray(parallel_model.predict(x=[val_sub_e,val_rel_e,val_ob_e]))
val_pred = np.asarray(parallel_model.predict(x=[val_sub_e,val_rel_e,val_ob_e])).round()


np.savetxt('val_pred_ur.out',val_pred_ur)
np.savetxt('val_lab.out',val_lab)

val_pred = val_pred.astype(int)
val_lab = val_lab.astype(int)
np.savetxt('val_pred.out',val_pred)
nb_confusion_matrix = confusion_matrix(val_lab, val_pred)
print("Confusion Matrix:")

nb_tp = nb_confusion_matrix[0][0]
nb_fn = nb_confusion_matrix[0][1]
nb_fp = nb_confusion_matrix[1][0]
nb_tn = nb_confusion_matrix[1][1]
print(nb_tp,nb_tn,nb_fp,nb_fn)
print('\t','Pred +\t','Pred -\t')
print('Actual +',str(nb_tp)+'\t',str(nb_fn)+'\t')
print('Actual -',str(nb_fp)+'\t',str(nb_tn)+'\t')

acc = (nb_tp+nb_tn)/float(nb_tp+nb_tn+nb_fn+nb_fp)
tpr = (nb_tp)/float(nb_tp+nb_fn)
fpr = (nb_fp)/float(nb_fp+nb_tn)
prec = (nb_tp)/float(nb_tp + nb_fp)
rec = (nb_tp)/float(nb_tp + nb_fn)
f1 = (2*nb_tp)/float(2*nb_tp + nb_fn + nb_fp)

print("Accuracy: " + str(acc))
print("True Positive Rate:"+str(tpr))
print("False Positive Rate:"+str(fpr))
print("Precision:"+str(prec))
print("Recall: "+str(rec))
print("F-Measure:"+str(f1))
