'''
A Fast and Accurate Dependency Parser using Neural Networks
'''

import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
import sys
import os
import nltk
import copy

sys.path.append('../tree')
from tree import *

def simpleedus(edus):

    pas = []
    for pa in edus:
        tokseq = []
        labels = []
        patags = []
        for line in pa:
            # print line.strip()
            toks = nltk.word_tokenize(line.strip().lower())
            # print toks

            # build tags
            label = (len(toks)-1)*[0] + [1]
            # print label
            tokseq.extend(toks)
            labels.extend(label)
        pas.append((tokseq,labels))


    return pas

def build_data(dir_path):
    
    files = os.listdir(dir_path);
    edus_path = [];
    for filename in files:
        if '.edus' in filename:
            # print filename;
            edus_path.append(filename);
    edus = [];
    sens = [];
    for edu_path in edus_path:
        edus.append(open(dir_path+'/'+edu_path).readlines());
        sens.extend(open(dir_path+'/'+edu_path[:-5]).readlines());

    train_x = [];
    train_y = [];

    print len(edus)
    pas = simpleedus(edus)

    for pa in pas:
        train_x.append(pa[0])
        train_y.append(pa[1])
    return [train_x,train_y]



def extract_edu(dir_path):
    files = os.listdir(dir_path);
    edus_path = [];
    for filename in files:
        if '.edus' in filename:
            # print filename;
            edus_path.append(filename);
    edus = [];
    sens = [];
    for edu_path in edus_path:
        edus.extend(open(dir_path+'/'+edu_path).readlines());
        sens.extend(open(dir_path+'/'+edu_path[:-5]).readlines());
    return edus;

def process_edu(edu_list):
    ap_edus = [];
    for edu in edu_list:
        tokens = word_tokenize(edu);

def load_wordembedding():
    pass;

def build_dataset(trnedus):
    """
    return tokens list of edu
    list of list edu tokens
    """
    edus_toks = [];
    for edu in trnedus:
        toks = nltk.word_tokenize(edu.strip().lower())
        edus_toks.append(toks)
    return edus_toks

def build_datasetlabel(trnedus):
    """

    """
    edus_toks_label = [];
    for i,edu in enumerate(trnedus):
        toks = nltk.word_tokenize(edu.strip().lower())

        # Begin 0
        # End 1
        # Continuance 2
        labels = [];
        for j,tok in enumerate(toks):
            if j == 0 :
                labels.append(0)
                # 
            elif j == len(toks)-1:
                labels.append(1)
            else :
                labels.append(2)

        edus_toks_label.append(labels)

    return edus_toks_label


class bidirectional_GRU:
    """
    the bidirectional gated recurrent unit neural network 

    maintain 2 GRU layer 

    combine this 2 layer output to generate output

    when you are using the GRU to perform the sequence labeling you need hyper-parameter below

    dimension of the hidden layer 
    number of classes
    number of word embeddings in the vocabulary (vocabulary size)
    dimesion of the word embeddings
    word window conetxt size is optional

    word embedding layer 


    """
    def  __init__(self,word_dim,label_dim,vocab_size,hidden_dim=128,word_embedding=None,bptt_truncate=-1):

        # assign instance variables

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # initialize the network parameters

        if word_embedding is None:
            # using random word vector
            # using glove 
            # using word2vec

            E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,vocab_size))
        else:
            # using pre-trained word vector
            E = word_embedding

        U = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,hidden_dim))
        # combine hidden states from 2 layer 
        V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(label_dim,hidden_dim*2))
        b = np.zeros((6,hidden_dim))
        c = np.zeros(label_dim)

        # Created shared variable
        self.E = theano.shared(name='E',value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V',value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c',value=c.astype(theano.config.floatX))

        # SGD / rmsprop : initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b , self.c

        x = T.ivector('x')
        y = T.ivector('y')

        def forward_direction_prop_step(x_t,s_t_prev):
            #
            #
            # Word embedding layer
            x_e = E[:,x_t]
            # GRU layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e)+W[0].dot(s_t_prev)) + b[0]
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e)+W[1].dot(s_t_prev)) + b[1]
            c_t = T.tanh(U[2].dot(x_e)+W[2].dot(s_t_prev*r_t)+b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            # directly return the hidden state as intermidate output 
            return [s_t]

        def backward_direction_prop_step(x_t,s_t_prev):
            #
            #
            #
            x_e = E[:,x_t]

            # GRU layer 2
            z_t = T.nnet.hard_sigmoid(U[3].dot(x_e)+W[3].dot(s_t_prev)) + b[3]
            r_t = T.nnet.hard_sigmoid(U[4].dot(x_e)+W[4].dot(s_t_prev)) + b[4]
            c_t = T.tanh(U[5].dot(x_e)+W[5].dot(s_t_prev*r_t)+b[5])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            return [s_t]

        def o_step(combined_s_t):
            o_t = T.nnet.softmax(V.dot(combined_s_t)+c)[0]
            return o_t


        # forward direction states
        f_s , updates = theano.scan(
                forward_direction_prop_step,
                sequences=x,
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))
            
        # backward direction states
        b_s , updates = theano.scan(
                backward_direction_prop_step,
                sequences=x[::-1], # the reverse direction input
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))

        self.f_s = f_s
        self.b_s = b_s

        f_b_s = b_s[::-1]


        # combine the forward GRU state and backward GRU state together 
        combined_s = T.concatenate([f_s,b_s[::-1]],axis=1)
        # concatenate the hidden state from 2 GRU layer to do the output
        o , updates = theano.scan(
                o_step,
                sequences=combined_s,
                truncate_gradient=self.bptt_truncate,
                outputs_info=None)



        prediction = T.argmax(o,axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o,y))

        cost = o_error

        # Gradients
        dE = T.grad(cost,E)
        dU = T.grad(cost,U)
        dW = T.grad(cost,W)
        db = T.grad(cost,b)
        dV = T.grad(cost,V)
        dc = T.grad(cost,c)

        # Assign functions
        self.predict = theano.function([x],o)
        self.predict_class = theano.function([x],prediction)
        self.ce_error = theano.function([x,y],cost)
        # self.bptt = theano.function([x,y],[dE,dU,dW,db,dV,dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1-decay) * dE ** 2
        mU = decay * self.mU + (1-decay) * dU ** 2
        mW = decay * self.mW + (1-decay) * dW ** 2
        mV = decay * self.mV + (1-decay) * dV ** 2
        mb = decay * self.mb + (1-decay) * db ** 2
        mc = decay * self.mc + (1-decay) * dc ** 2

       
        updates = [(E,E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                   (U,U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                   (W,W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                   (V,V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                   (b,b - learning_rate * db / T.sqrt(mb + 1e-6)),
                   (c,c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                   (self.mE,mE),
                   (self.mU,mU),
                   (self.mW,mW),
                   (self.mV,mV),
                   (self.mb,mb),
                   (self.mc,mc)]
        

        self.sgd_step = theano.function(
                [x,y, learning_rate, theano.Param(decay,default=0.9)],
                [],
                updates=updates)

    def calculate_total_loss(self,X,Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self,X,Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)


def sgd_callback(model,num_examples_seen,X_train,y_train,index_to_word,word_to_index):
    dt = datetime.now().isoformat()
    loss = model.calculate_loss(X_train[:10000],y_train[:10000])

    print("\n%s (%d)" % (dt, num_examples_seen)) 
    print("-------------------------------------------------")
    print("Loss : %f" % loss)

    generate_sentences(model,10,index_to_word,word_to_index)

    print("\n")
    sys.stdout.flush()


def train_with_sgd(model,X_train,y_train,learning_rate=0.001,nepoch=20,decay=0.9,index_to_word=[]):
    num_examples_seen = 0

    print 'now learning_rate : ' , learning_rate;
    for epoch in range(nepoch):
        # For each training example ...

        #
        #
        #
        tocount = 0
        tccount = 0
        tycount = 0

        for i in np.random.permutation(len(y_train)):
            # One SGT step
            model.sgd_step(X_train[i],y_train[i],learning_rate,decay)
            num_examples_seen += 1
            # Optionally do callback
            print '>>>>>'

            wrds = []
            for j in X_train[i]:
                wrds.append(index_to_word[j])

            print 'i-th :' , i;
            print 'X_train[i] : ' , X_train[i]
            print 'the edus : ' ," ".join(wrds)
            print 'ce_error : ' , model.ce_error(X_train[i],y_train[i])
            # print 'predict : ' , model.predict(X_train[i])
            output = model.predict_class(X_train[i])
            print 'predict_class : ' , output

            # boundary accuracy 
            # ocount is total boundary output number
            # ccount is correct boundary number
            # ycount is the true boundary number
            ocount = 0
            ccount = 0
            ycount = 0

            for o,y in zip(output,y_train[i]):
                if o == 1:
                    ocount += 1
                if y == 1:
                    ycount += 1
                if y == o and y == 1:
                    ccount += 1
            #
            tocount += ocount
            tccount += ccount
            tycount += ycount
            if ccount != 0 and ocount != 0:
                precision = float(ccount) / float(ocount)
                recall = float(ccount) / float(ycount)
                if (precision+recall) != 0:
                    Fmeasure = 2 * (precision*recall) / (precision+recall)
                else:
                    Fmeasure = 0
            else :
                precision = 0
                recall = 0
                Fmeasure = 0
            print 'Accuracy : ' , precision
            print 'Recall : ' , recall
            print 'Fmeasure : ' , Fmeasure

        # a epoch for training end here
        if tocount != 0 and tccount != 0:
            precision = float(tccount) / float(tocount)
            recall = float(tccount) / float(tycount)
            if (precision+recall) != 0:
                Fmeasure = 2 * (precision*recall) / (precision+recall)
            else:
                Fmeasure = 0
        else:
            precision = 0
            recall = 0
            Fmeasure = 0

        print 'Accuracy of training set: ' , precision
        print 'Recall of training set: ' , recall
        print 'Fmeasure of training set: ' , Fmeasure


    return model


def test_score(model,X_test,y_test):
    print 'now score the test dataset'
    scores = [];
    tocount = 0
    tccount = 0
    tycount = 0

    for i in range(len(y_test)):
        output = model.predict_class(X_test[i])
        ocount = 0
        ccount = 0
        ycount = 0

        for o,y in zip(output,y_test[i]):
            if o == 1:
                ocount += 1
            if y == 1:
                ycount += 1
            if y == o and y == 1:
                ccount += 1

        tocount += ocount
        tccount += ccount
        tycount += ycount

    if tocount != 0 and tccount != 0 :
        precision = float(tccount) / float(tocount)
        recall = float(tccount) / float(tycount)
        if (precision+recall) != 0:
            Fmeasure = 2 * (precision*recall) / (precision+recall)
        else:
            Fmeasure = 0
    else:
        precision = 0
        recall = 0
        Fmeasure = 0

    print 'Accuracy of test set: ' , precision
    print 'Recall of test set: ' , recall
    print 'Fmeasure of test set: ' , Fmeasure

def load_word_embedding(path):

    #
    #
    if False:
        pass;
    else:

        wdvec = open(path,'r').readlines()
        wvdic = dict()
        for widx, vec in enumerate(wdvec):
            items = vec.strip().split();
            pv = np.array([float(x) for x in items[1:]]);
            wvdic[items[0]] = pv;
            wrddim = wvdic.itervalues().next().shape[0];
    return wvdic

def build_we_matrix(wvdic,index_to_word,word_to_index,word_dim):

    # index_to_word is the word list
    vocab_size = len(index_to_word)
    E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,vocab_size)) 
    
    # for those word can be found in glove or word2vec pre-trained word vector
    for w in index_to_word:
        if w in wvdic:
            E[:,word_to_index[w]] = wvdic[w]

    return E



def parser():
    

    edux , eduy = build_data('data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING');
    testx, testy = build_data('data/RSTmain/RSTtrees-WSJ-main-1.0/TEST')

    


if __name__ == '__main__':
    relation();
