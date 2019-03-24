import pandas as pd

import numpy as np

import scipy.sparse as sp

import torch

import math

from torch import nn

from torch import sparse

from torch.autograd import Variable

from torch.nn import BCELoss

from torch import optim

from torch.utils.data import DataLoader, Dataset, TensorDataset





gpu = 1

cudafy = lambda x: x.cuda(gpu) if gpu is not None else x



facts = pd.read_csv("FB15k/facts.txt", sep="\t", header=None)

df = pd.read_csv("FB15k/train.txt", sep="\t", header=None)

df2 = pd.read_csv("FB15k/valid.txt", sep="\t", header=None)

s_facts, p_facts, o_facts = facts[0], facts[1], facts[2]

s_train, p_train, o_train = df[0], df[1], df[2]

s_valid, p_valid, o_valid = df2[0], df2[1], df2[2]



cat_entities = [s_facts, o_facts, s_train, o_train, s_valid, o_valid]

cat_relations = [p_facts, p_train, p_valid]

mapped_entities, id2entity = pd.factorize(pd.concat(cat_entities))

mapped_relations, id2relation = pd.factorize(pd.concat(cat_relations))



def getSplits(cat):

    total = 0

    splits = []

    for array in cat[:-1]:

        splits += [total + len(array)]

        total += len(array)

    return splits



s_facts, o_facts, s_train, o_train, s_valid, o_valid = [np.reshape(i, (-1, 1)) for i in np.split(mapped_entities, getSplits(cat_entities))]

p_facts, p_train, p_valid = [np.reshape(i, (-1, 1)) for i in np.split(mapped_relations, getSplits(cat_relations))]



spo_facts = np.concatenate([s_facts, p_facts, o_facts], axis=-1)

spo_train = np.concatenate([s_train, p_train, o_train], axis=-1)

spo_valid = np.concatenate([s_valid, p_valid, o_valid], axis=-1)




class TransE(nn.Module):

    def __init__(self, num_entities, num_relations, entity_dimensions, relation_dimensions, gpu=None):

        super(TransE, self).__init__()

        assert entity_dimensions == relation_dimensions

        self.num_entities = num_entities

        self.num_relations = num_relations

        self.entity_dimensions = entity_dimensions

        self.relation_dimensions = relation_dimensions

        #self.batch_size = batch_size



        # Model parameters

        self.embed_entities = nn.Embedding(self.num_entities, self.entity_dimensions)

        self.embed_relations = nn.Embedding(self.num_relations, self.relation_dimensions)

        self.embeddings = [self.embed_entities, self.embed_relations]

        self.initialize_embeddings()



        if gpu is not None:

            self.cuda(gpu)



    def normalize_embeddings(self):

        for e in self.embeddings:

            e.weight.data.renorm_(p=2, dim=0, maxnorm=1)



    def initialize_embeddings(self):

        r = 6/np.sqrt(self.entity_dimensions)



        for e in self.embeddings:

            e.weight.data.uniform_(-r, r)



        self.normalize_embeddings()



    def forward(self, subject, relation, object):

        # print "before", subject.size(), relation.size(), object.size()

        subject = self.embed_entities(subject)

        relation = self.embed_relations(relation)

        object = self.embed_entities(object)

        # print subject.size(), relation.size(), object.size()

        score = torch.sum(subject+relation-object, -1)

        # print score.size()

        # score = torch.bmm(a.view(-1, 1, self.entity_dimensions), a.view(-1, self.entity_dimensions, 1))




        return score.view(-1,1)



batch_size_valid = 200
batch_size_train = 10000



spo_train = cudafy(torch.LongTensor(spo_train))

spo_valid = cudafy(torch.LongTensor(spo_valid))

train_dataset = DataLoader(TensorDataset(spo_train, torch.zeros(spo_train.size(0))), batch_size=batch_size_train, shuffle=True, drop_last=True)

valid_dataset = DataLoader(TensorDataset(spo_valid, torch.zeros(spo_valid.size(0))), batch_size=batch_size_valid, shuffle=True, drop_last=True)



entities = cudafy(torch.LongTensor(np.arange(id2entity.size)))

model = TransE(id2entity.size, id2relation.size, 128, 128,  gpu=gpu)

#model = nn.DataParallel(TransE(id2entity.size, id2relation.size, 128, 128, batch_size, gpu=gpu)) #changed

optimizer = optim.Adam(model.parameters())

zero = cudafy(torch.FloatTensor([0.0]))



def train_transE(batch, total_batches, i, spo):

    s, p, o = torch.chunk(spo, 3, dim=1)

    no = cudafy(torch.LongTensor(np.random.randint(0, model.num_entities, o.size(0))).view(-1,1))

    ns = cudafy(torch.LongTensor(np.random.randint(0, model.num_entities, s.size(0))).view(-1,1))

    criterion = lambda pos, neg: torch.sum(torch.max(Variable(zero), 1.0 - pos + neg))


    optimizer.zero_grad()

    pos_score = model(Variable(s), Variable(p), Variable(o))

    neg_score = model(Variable(s), Variable(p), Variable(no))

    loss = criterion(pos_score, neg_score)

    loss.backward()

    # if batch % 10 == 0:

    #     print ("Epoch %d: Batch: %d/%d Loss: %f" % (i+1, batch+1, total_batches, loss.data[0]))

    optimizer.step()



    optimizer.zero_grad()

    pos_score = model(Variable(s), Variable(p), Variable(o))

    neg_score = model(Variable(ns), Variable(p), Variable(o))

    loss = criterion(pos_score, neg_score)

    loss.backward()

    if batch % 10 == 0:

        print ("Epoch %d: Batch: %d/%d Loss: %f" % (i+1, batch+1, total_batches, loss.data[0]))

    optimizer.step()




    return loss.data[0]



def hitsatk_transe(spo, k):

    total = 0.0

    # for i in range(0, tmodel.batch_size, 1):

    s, p, o = torch.chunk(spo, 3, dim=1)

    s = s.repeat(1, model.num_entities).view(-1, 1)

    p = p.repeat(1, model.num_entities).view(-1, 1)

    e = entities.repeat(batch_size_valid).view(-1,1)

    # print (s.size(), p.size(), entities.size())

    output = model(Variable(s), Variable(p), Variable(e))

    output = output.view(-1, entities.size(0))

    # print output.size()

    # torch.cat(scores)

    hits = torch.nonzero((o == torch.topk(output, k, dim=-1)[1].data).view(-1))

    if len(hits.size()) > 0:

        total += float(hits.size(0)) / o.size(0)



    return total





def run_transe_validation():

    hits1 = []
    hits10 = []
    hits100 = []

    for batch_id, (spo, _) in enumerate(valid_dataset):

        print ("Validation batch ", batch_id)

        hits1 += [hitsatk_transe(spo, 1)]
        hits10 += [hitsatk_transe(spo, 10)]
        hits100 += [hitsatk_transe(spo, 100)]

    print( "Validation hits@1: %f" % (float(sum(hits1)) / len(hits1)))
    print( "Validation hits@10: %f" % (float(sum(hits10)) / len(hits10)))
    print( "Validation hits@100: %f" % (float(sum(hits100)) / len(hits100)))


def run_transe():

    #run_transe_validation()

    for i in range(0,40):

        total_batches = len(train_dataset)

        epoch_loss = 0.0

        for batch_id, (spo, _) in enumerate(train_dataset):

            epoch_loss += train_transE(batch_id, total_batches, i, spo)

            # break



        print ("Epoch %d Total loss: %f" % (i+1, epoch_loss/total_batches))



       # if (i+1)%10 == 0:
       #     run_transe_validation()





if __name__ == "__main__":

    run_transe()
    torch.save(model, './models/FB15k/model')
    run_transe_validation()
