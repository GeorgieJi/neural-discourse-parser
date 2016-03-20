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


def traversal(tree):
    parentpair = []
    # usually the len of tree is 4
    # print '>>>>>>>>>>>>>>>>>>>>.*'
    left_tree = tree[2]
    right_tree = tree[3]
    span = ''
    leftspan = ''
    rightspan = ''
    leftpair = []
    rightpair = []

    if len(left_tree) == 4:
        leftpair , leftspan = traversal(left_tree)

    if len(right_tree) == 4:
        rightpair , rightspan = traversal(right_tree)

    if len(left_tree) == 3:
        leftspan = left_tree[2]

    if len(right_tree) == 3:
        rightspan = right_tree[2]

    parentpair = [ [left_tree[0]+'-'+right_tree[0] , leftspan, rightspan] ]
    parentpair.extend(leftpair)
    parentpair.extend(rightpair)
    # print '->' , parentpair
    
    return parentpair, leftspan + ' ' + rightspan


def extract_nucleus(tree_str):

    tree = parse_tree(tree_str)
    pairs = traversal(tree)[0]
    return pairs

def build_data(dir_path):
    
    files = os.listdir(dir_path);
    edus_path = [];
    for filename in files:
        if '.dis' in filename:
            # print filename;
            edus_path.append(filename);

    trees = [];
    for edu_path in edus_path:
        trees.append(open(dir_path+'/'+edu_path).readlines());


    pairs = []
    for tree in trees:
        pairs.extend(extract_nucleus(tree))

    senas = []
    senbs = []
    nucs = []

    for pair in pairs:
        errorflag = True
        if pair[0] == 'Nucleus-Satellite':
            nucs.append([0])
        elif pair[0] == 'Nucleus-Nucleus':
            nucs.append([1])
        elif pair[0] == 'Satellite-Nucleus':
            nucs.append([2])
        else:
            errorflag = False # filter the wrong tree
            

        if errorflag:
            senas.append(nltk.word_tokenize(pair[1].strip().lower()))
            senbs.append(nltk.word_tokenize(pair[2].strip().lower()))
        else:
            continue
    
    return senas , senbs , nucs



class Siamese_GRU:
    """
    A implemention of Siamese Recurrent Architectures for Learning Sentence Similarity
    http://www.aaai.org/Conferences/AAAI/2016/Papers/15Mueller12195.pdf

    """
    def  __init__(self,word_dim,label_dim,vocab_size,hidden_dim=128,word_embedding=None,bptt_truncate=-1):

        """
        Train 2 spearate GRU network to represent each sentence in pair as a fixed-length vector
        then calculate the 2 sentence vector Manhanttan distance 
        """
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

        x_a = T.ivector('x_a')
        x_b = T.ivector('x_b')
        y = T.lvector('y')

        def sena_forward_step(x_t,s_t_prev):
            # Word embedding layer
            x_e = E[:,x_t]
            # GRU layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e)+W[0].dot(s_t_prev)) + b[0]
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e)+W[1].dot(s_t_prev)) + b[1]
            c_t = T.tanh(U[2].dot(x_e)+W[2].dot(s_t_prev*r_t)+b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            # directly return the hidden state as intermidate output 
            return [s_t]

        def senb_forward_step(x_t,s_t_prev):
            x_e = E[:,x_t]

            # GRU layer 2
            z_t = T.nnet.hard_sigmoid(U[3].dot(x_e)+W[3].dot(s_t_prev)) + b[3]
            r_t = T.nnet.hard_sigmoid(U[4].dot(x_e)+W[4].dot(s_t_prev)) + b[4]
            c_t = T.tanh(U[5].dot(x_e)+W[5].dot(s_t_prev*r_t)+b[5])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            return [s_t]

        def manhattan_distance(h_a,h_b):
            return T.exp(-1*abs(h_a-h_b))

        def mse_error(p,y):
            return (p-y)**2

        # sentence a vector (states)
        a_s , updates = theano.scan(
                sena_forward_step,
                sequences=x_a,
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))
            
        # sentence b vector (states)
        b_s , updates = theano.scan(
                senb_forward_step,
                sequences=x_b,
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))

        # semantic similarity 
        # s_sim = manhattan_distance(a_s[-1],b_s[-1])

        # for classification using simple strategy 
        sena = a_s[-1]
        senb = b_s[-1]

        combined_s = T.concatenate([sena,senb],axis=0)

        # softmax class
        o = T.nnet.softmax(V.dot(combined_s)+c)[0]
        om = o.reshape((1,o.shape[0]))
        prediction = T.argmax(om,axis=1)
        o_error = T.nnet.categorical_crossentropy(om,y)


        # cost 
        cost = T.sum(o_error)

        # Gradients
        dE = T.grad(cost,E)
        dU = T.grad(cost,U)
        dW = T.grad(cost,W)
        db = T.grad(cost,b)
        dV = T.grad(cost,V)
        dc = T.grad(cost,c)

        # Assign functions
        self.predict = theano.function([x_a,x_b],om)
        self.predict_class = theano.function([x_a,x_b],prediction)
        self.ce_error = theano.function([x_a,x_b,y],cost)
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
                [x_a,x_b,y, learning_rate, theano.Param(decay,default=0.9)],
                [],
                updates=updates)


def index_to_class(index):
    label = '';
    if index == 0:
        label = 'Nucleus-Satellite'
    elif index == 1:
        label = 'Nucleus-Nucleus'
    elif index == 2:
        label = 'Satellite-Nucleus'
    return label

def train_with_sgd(model,X_1_train,X_2_train,y_train,learning_rate=0.001,nepoch=20,decay=0.9,index_to_word=[]):
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
            model.sgd_step(X_1_train[i],X_2_train[i],y_train[i],learning_rate,decay)
            num_examples_seen += 1
            # Optionally do callback
            print '>>>>>'

            lwrds = []
            rwrds = []
            for lj,rj in zip(X_1_train[i],X_2_train[i]):
                lwrds.append(index_to_word[lj])
                rwrds.append(index_to_word[rj])

            print 'i-th :' , i;
            print 'the left edu : ' ," ".join(lwrds)
            print 'the right edu : ' , " ".join(rwrds)
            print 'ce_error : ' , model.ce_error(X_1_train[i],X_2_train[i],y_train[i])
            # print 'predict : ' , model.predict(X_train[i])
            output = model.predict_class(X_1_train[i],X_2_train[i])
            print 'predict_class : ' , output
            print index_to_class(output)
            print 'true label : ' , y_train[i]
            print index_to_class(y_train[i][0])
            # boundary accuracy 
            # ocount is total boundary output number
            # ccount is correct boundary number
            # ycount is the true boundary number
            ocount = 0
            ccount = 0
            ycount = 0

            for o,y in zip(output,y_train[i]):
                if o == y:
                    ccount += 1
                ycount += 1
                ocount += 1
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


def test_score(model,X_1_test,X_2_test,y_test,index_to_word):
    print 'now score the test dataset'
    scores = [];
    tocount = 0
    tccount = 0
    tycount = 0

    for i in range(len(y_test)):
        output = model.predict_class(X_1_test[i],X_2_test[i])
        ocount = 0
        ccount = 0
        ycount = 0

        lwrds = []
        rwrds = []

        for lj, rj in zip(X_1_test[i],X_2_test[i]):
            lwrds.append(index_to_word[lj])
            rwrds.append(index_to_word[rj])

        print 'i-th : ' , i;
        print 'the left edu : , ' , " ".join(lwrds)
        print 'the right edu : ' , " ".join(rwrds)
        print 'ce_error : ' , model.ce_error(X_1_test[i],X_2_test[i],y_test[i])

        # print 
        print 'predict_class : ' , output
        print index_to_class(output)
        print 'true label : ' , y_test[i]
        print index_to_class(y_test[i][0])


        for o,y in zip(output,y_test[i]):
            if y == o:
                ccount += 1
            ycount += 1
            ocount += 1

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



def nucleus():

    ledus, redus , nucs = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING');
    tst_ledus, tst_redus, tst_nucs = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TEST')


    print 'load in ' , len(nucs) , 'training sample'
    print 'load in ' , len(tst_nucs) , 'test sample'

    token_list = []
    for sena, senb in zip(ledus,redus):
        token_list.extend(sena)
        token_list.extend(senb)

    word_freq = nltk.FreqDist(token_list)
    print 'Found %d unique words tokens . ' % len(word_freq.items())

    vocabulary_size = 5*1000
    unknown_token = 'UNK'

    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    print 'vocab : '
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print 'Using vocabulary size %d. ' % vocabulary_size
    print "the least frequent word in our vocabulary is '%s' and appeared %d times " % (vocab[-1][0],vocab[-1][1])

    # training dataset
    for i,(edua,edub) in enumerate(zip(ledus,redus)):
        ledus[i] = [w if w in word_to_index else unknown_token for w in edua]
        redus[i] = [w if w in word_to_index else unknown_token for w in edub]

    # test dataset
    for i,(edua,edub) in enumerate(zip(tst_ledus,tst_redus)):
        tst_ledus[i] = [w if w in word_to_index else unknown_token for w in edua]
        tst_redus[i] = [w if w in word_to_index else unknown_token for w in edub]

    # X_1_train , X_2_train , y_train
    X_1_train = np.asarray([[word_to_index[w] for w in sent ] for sent in ledus])
    X_2_train = np.asarray([[word_to_index[w] for w in sent ] for sent in redus])
    y_train = (nucs)

    # X_1_test, X_2_test , y_train
    X_1_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_ledus])
    X_2_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_redus])
    y_test = (tst_nucs)

    print "\n Example sentence '%s' " % " ".join(ledus[0])
    print "\n Example sentence '%s' " % " ".join(redus[0])
    print "\n Example sentence after Pre-processing : '%s' " % X_1_train[0]
    print "\n Example sentence after Pre-processing : '%s' " % X_2_train[0]
    print "\n Example label : ", y_train[0]
    print ""

    # build Embedding matrix
    label_size = 3
    wvdic = load_word_embedding('../data/glove.6B.200d.txt')
    word_dim = wvdic.values()[0].shape[0]

    E = build_we_matrix(wvdic,index_to_word,word_to_index,word_dim)

    model = Siamese_GRU(word_dim,label_size,vocabulary_size,hidden_dim=128,word_embedding=E,bptt_truncate=-1)

    # Print SGD step time
    t1 = time.time()
    print model.predict(X_1_train[0],X_2_train[0])
    output = model.predict_class(X_1_train[0],X_2_train[0])
    print 'predict_class : ' , output
    print 'ce_error : ' , model.ce_error(X_1_train[0],X_2_train[0],y_train[0])
    learning_rate = 0.0000005

    model.sgd_step(X_1_train[0],X_2_train[0],y_train[0],learning_rate)
    t2 = time.time()

    print "SGD Step time : %f milliseconds " % ((t2-t1)*1000.)
    sys.stdout.flush()

    # 
    NEPOCH = 100

    for epoch in range(NEPOCH):

        print 'this is epoch : ' , epoch
        train_with_sgd(model,X_1_train,X_2_train,y_train,learning_rate=learning_rate,nepoch=1,decay=0.9,index_to_word=index_to_word)

        test_score(model,X_1_test,X_2_test,y_test,index_to_word=index_to_word)

    


if __name__ == '__main__':
    nucleus();
