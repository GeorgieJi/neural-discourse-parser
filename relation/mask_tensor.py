'''
A Fast and Accurate Discourse Parser using Neural Networks

* using bidirectional Gated Recurrent Unit with mask to learn sentence representation
* using tensor 
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
# from theano.compile.nanguardmode import NanGuardMode
from collections import OrderedDict

sys.path.append('../tree')
from tree import *
from utils import *

class ConvPoolLayer1D(object):

    def __init__(self,filter_shape,image_shape,poolsize=(2,2),W=None,b=None):

        assert iamge_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])/np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = rng.uniform(low=-W_bound,high=W_bound,size=filter_shape)
        b = np.zeros((filter_shape[0]),dtype=theano.config.floatX)
        
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))
        
        self.params = [self.W,self.b]
        
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):

        W, b = self.W, self.b
        x = T.tensor3('x')
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
                input = x,
                filters = W,
                filter_shape = filter_shape,
                image_shape = image_shape
                )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=False
                )

        output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        self.cnn_output = theano.function([x],output)



class HiddenLayer(object):
    def __init__(self, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        rng = np.random.RandomState(2345)
        # end-snippet-1
        if True :
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
        self.W = theano.shared(value=W_values, name='W', borrow=True)
        if True :
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        if W is not None:
            self.W = W;
        if b is not None:
            self.b = (b);
        self.n_in = n_in;
        self.n_out = n_out;
        self.activation=activation;
        # parameters of the model
        self.params = [self.W, self.b]
        
        # self.theano = {}
        # self.__theano_build__()
    def forward(self,x):
        W, b, activation  = self.W , self.b, self.activation
        # x = T.vector('x')
        lin_output = T.dot(x, self.W) + self.b
        output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        return output



class bid_GRU:

    def __init__(self,word_dim,hidden_dim,bptt_truncate=-1):

        # assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        

        # print E[:,0]
        # initialize the network parameters

        U = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,hidden_dim))
        b = np.zeros((6,hidden_dim))

        # initialize the soft attention parameters
        # basically the soft attention is the single hidden layer 
        # no idea how to set the attention layer hidden node dim just set it as hidden dim for now


        # Created shared variable
        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))

        # self.params = [self.U,self.W,self.b,self.W_att,self.v_att,self.b_att,self.sv_att]
        self.params = [self.U,self.W,self.b]
        self.L2 = (self.U**2).sum() + (self.W**2).sum()

        # store the theano graph 
        # self.theano = {}
        # self.__theano_build__()
 
    def recurrent(self,x_s,x_s_m,E):
        U, W, b = self.U, self.W, self.b
        # x_s = T.ivector('x_s')

        def forward_direction_step(x_t,x_s_m_t,s_t_prev):
            # Word embedding layer
            x_e = E[:,x_t]
            # GRU layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e)+W[0].dot(s_t_prev)) + b[0]
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e)+W[1].dot(s_t_prev)) + b[1]
            c_t = T.tanh(U[2].dot(x_e)+W[2].dot(s_t_prev*r_t)+b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev

            # leaky integrate and obtain next hidden state
            s_t = s_t * x_s_m_t + s_t_prev * (1. - x_s_m_t)

            # directly return the hidden state as intermidate output 
            return [s_t]
        
        def backward_direction_step(x_t,x_s_m_t,s_t_prev):
            # Word embedding layer
            x_e = E[:,x_t]
            # GRU layer 2
            z_t = T.nnet.hard_sigmoid(U[3].dot(x_e)+W[3].dot(s_t_prev)) + b[3]
            r_t = T.nnet.hard_sigmoid(U[4].dot(x_e)+W[4].dot(s_t_prev)) + b[4]
            c_t = T.tanh(U[5].dot(x_e)+W[5].dot(s_t_prev*r_t)+b[5])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev

            # leaky integrate and obtain next hidden state
            s_t = s_t * x_s_m_t + s_t_prev * (1. - x_s_m_t)

            # directly return the hidden state as intermidate output 
            return [s_t]

        # create sequence hidden states from input
        s_f , updates = theano.scan(
                forward_direction_step,
                sequences=[x_s,x_s_m],
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))

        s_b , updates = theano.scan(
                backward_direction_step,
                sequences=[x_s[::-1],x_s_m[::-1]],
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))

        h_s = T.concatenate([s_f,s_b[::-1]],axis=1)

        return h_s


# new update for soft attention layer
class soft_attention_layer:

    def __init__(self,hidden_dim):

        # assign instance variables
        self.hidden_dim = hidden_dim

        W_att = np.random.uniform(-np.sqrt(1./hidden_dim*2),np.sqrt(1./hidden_dim*2),(hidden_dim,hidden_dim*2))
        v_att = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim))
        b_att = np.zeros((hidden_dim))
        sv_att = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim*2))

        # 
        self.W_att = theano.shared(name='W_att',value=W_att.astype(theano.config.floatX))
        self.v_att = theano.shared(name='v_att',value=v_att.astype(theano.config.floatX))
        self.b_att = theano.shared(name='b_att',value=b_att.astype(theano.config.floatX))
        self.sv_att = theano.shared(name='sv_att',value=sv_att.astype(theano.config.floatX))

        # collect parameter

        self.params = [self.sv_att]
        self.L2 = (self.sv_att**2).sum()

    def soft_attention(self,x_s,x_s_m):
        W_att, v_att, b_att, sv_att = self.W_att, self.v_att, self.b_att, self.sv_att
        
        def score_attention(h_i,x_s_m_t):
            return x_s_m_t*sv_att.dot(h_i)

        def weight_attention(h_i,a_j):
            return h_i*a_j

        h_att, updates = theano.scan(
                score_attention,
                sequences=[x_s,x_s_m]
                )

        h_att = T.exp(h_att)
        h_att = h_att.flatten()
        h_att = h_att / h_att.sum()

        h_s_att, updates = theano.scan(
                weight_attention,
                sequences=[x_s,h_att]
                )

        a_s = h_s_att.sum(axis=0)
        
        return a_s


# batch preparation
def prepare_data(seqs_x, maxlen=None):

    # 
    lengths_x = [len(s) for s in seqs_x]
    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []

        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1

    x = np.zeros((n_samples, maxlen_x)).astype('int64')
    x_mask = np.zeros((n_samples, maxlen_x)).astype('float32')

    for idx , s_x in enumerate(seqs_x):
        x[idx,:lengths_x[idx]] = s_x
        x_mask[idx,:lengths_x[idx]] = 1.

    return x, x_mask


class tensor_layer:
    """
    the sentence vector is hidden_dim , with k slices , tensor acts like a score function
    build tensor layer for each discourse relation
    """
    def __init__(self,hidden_dim,k,bptt_truncate=-1):
        
        # assign instance variables
        self.hidden_dim = hidden_dim
        self.k = k
        self.bptt_truncate = bptt_truncate
        
        # 
        # build tensor layer here
        # g (e1,R,e2) = uTr f (e1T Wr(1:k) e2 + Vr [e1,e2] + bR)
        # loss = N_sum Kclass_sum max(0,1-g(Ti)+g(Tic)) + decay 

        # u = R'k
        # w = R'k*d*d
        # v = R'k*2d
        # b = R'k
        Wr = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(k,hidden_dim,hidden_dim))
        Vr = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(k,2*hidden_dim))
        br = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(k))
        ur = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(k))

        # Created shared variable
        self.Wr = theano.shared(name='Wr',value=Wr.astype(theano.config.floatX))
        self.Vr = theano.shared(name='Vr',value=Vr.astype(theano.config.floatX))
        self.br = theano.shared(name='br',value=br.astype(theano.config.floatX))
        self.ur = theano.shared(name='ur',value=ur.astype(theano.config.floatX))

        # parameter collections

        self.params = [self.Wr,self.Vr,self.br,self.ur]

        # L2 regularized 
        self.L2 = (self.Wr**2).sum() + (self.Vr**2).sum() + (self.ur**2).sum()

    # build theano graph here

    def tensor_score(self,s1,s2):
        Wr, Vr, br, ur = self.Wr, self.Vr, self.br, self.ur
        # for each slice of k
        def slice_tensor(w,v,b,s1,s2):
            c_s = T.concatenate([s1,s2],axis=0)
            g = T.tanh( ( (s1.T).dot(w) ).dot(s2) + v.dot(c_s.T) + b )
            return g
        kscore, updates = theano.scan(
                slice_tensor,
                sequences=[Wr,Vr,br],
                non_sequences=[s1,s2],
                outputs_info=None,
                truncate_gradient=self.bptt_truncate
                )
        # kscore = kscore.reshape((kscore.shape[0]))
        # perform ur in k vector
        tscore = ur.dot(kscore.T)
        
        return tscore


class label_score:

    def __init__(self,label_dim,hidden_dim,k):

        self.tensor_layers = []
        self.params = []
        self.L2 = 0

        for i in range(label_dim):
            tmp_tensor = tensor_layer(hidden_dim,k)
            self.tensor_layers.append(tmp_tensor)
            self.params += tmp_tensor.params
            self.L2 += tmp_tensor.L2

    def score(self,s1,s2):
        c_s = []
        for t in self.tensor_layers:
            c_s.append(t.tensor_score(s1,s2))

        # use Stack tensors in sequence on given axis
        c_s = T.stack(c_s).flatten()
        return c_s


class framework:
    """
    build all theano graph here , in here we can combine as many as nerual layer we need !
    """
    def  __init__(self,word_dim,label_dim,vocab_size,hidden_dim=128,word_embedding=None,bptt_truncate=-1):

        # the frameword only holds the global word embedding 
        if word_embedding is None:
            E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,vocab_size))
        else:
            E = word_embedding
        self.E = theano.shared(name='E',value=E.astype(theano.config.floatX))

        # build bi-GRU
        # def __init__(self,word_dim,hidden_dim,word_embedding,bptt_truncate=-1)
        gru_layer = bid_GRU(word_dim,hidden_dim,bptt_truncate=-1)

        x_a = T.lvector('x_a')
        x_a_m = T.vector('x_a_m')
        x_b = T.lvector('x_b')
        x_b_m = T.vector('x_b_m')

        y = T.lvector('y')

        # 2 symbolic vector (1*)
        # 2 symbolic vector list / code update 
        v_a = gru_layer.recurrent(x_a,x_a_m,self.E)
        v_b = gru_layer.recurrent(x_b,x_b_m,self.E)

        # soft attention layer , single layer for all
        sa_layer = soft_attention_layer(hidden_dim)

        a_x_a = sa_layer.soft_attention(v_a,x_a_m)
        a_x_b = sa_layer.soft_attention(v_b,x_b_m)

        # print type(a_x_a)
        # print type(a_x_b)
        # now the a_x_a and a_x_b is vector with shape of (hidden_dim*2)


        tcr_score = label_score(label_dim,hidden_dim*2,1)
        # 
        l_s = tcr_score.score(a_x_a,a_x_b)

        print 'number of parameters of tensor layer'
        print len(tcr_score.params)

        # cost 
        # L(i) = max(0, 1-g(i)+g(c))

        # index
        # tensor score 
        c_i = y[0]
        ks_r = T.arange(l_s.shape[0])

        def _compare(k,y,l_s):
            # T.eq
            t = T.switch( T.eq(k,y) , 0 , T.maximum(0,1-l_s[y]+l_s[k]) )
            return t

        o,updates = theano.scan(
                _compare,
                sequences=ks_r,
                non_sequences = [c_i,l_s]
                )


        # edu_pair_fea = T.concatenate([a_x_a,a_x_b],axis=0)
        # build hidden_layer for edu pair
        # mlp_layer_1 = HiddenLayer(hidden_dim*4,label_dim)
        # mlp_layer_2 = HiddenLayer(hidden_dim,label_dim)
        # ep_fea_2 = mlp_layer_1.forward(edu_pair_fea)
        # ep_fea_3 = mlp_layer_2.forward(ep_fea_2)
        # softmax 
        # o = T.nnet.softmax(ep_fea_2)[0]

        # trick for prevent nan
        # eps = np.asarray([1.0e-10]*label_dim,dtype=theano.config.floatX)
        # o = o + eps
        om = l_s.reshape((1,l_s.shape[0]))
        prediction = T.argmax(om,axis=1)
        # o_error = T.nnet.categorical_crossentropy(om,y)

        # cost
        # cost = T.sum(o_error)

        # collect the params from each model
        self.params = []
        self.params = self.params + [ self.E ]
        self.params = self.params + gru_layer.params 
        self.params = self.params + sa_layer.params
        self.params = self.params + tcr_score.params

        # collect the L2 reg from each model
        self.L2 = 0
        self.L2 += self.L2 + gru_layer.L2
        self.L2 += self.L2 + sa_layer.L2
        self.L2 += self.L2 + tcr_score.L2

        cost = T.sum(o) + 0.00001*self.L2


        # please verify the parameters of model
        print 'please verify the parameter of model'
        print self.params
        print len(self.params)

        # updates
        updates = sgd_updates_adadelta(norm=0,params=self.params,cost=cost)


        # framework assign function
        L2 = self.L2
        self.L2_reg = theano.function([],L2)
        self.predict = theano.function([x_a,x_a_m,x_b,x_b_m],prediction)
        self.predict_class = theano.function([x_a,x_a_m,x_b,x_b_m],prediction)
        self.ce_error = theano.function([x_a,x_a_m,x_b,x_b_m,y],cost)
        self.t_score = theano.function([x_a,x_a_m,x_b,x_b_m],l_s)
        # self.comsen = theano.function([],)

        self.sgd_step = theano.function(
                [x_a,x_a_m,x_b,x_b_m,y],
                [],
                updates = updates
                )

        pass


def relation():

    maxlen = 50
    ledus, redus , rels = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING',maxlen);
    tst_ledus, tst_redus, tst_rels = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TEST',maxlen)
    print 'load in ' , len(rels) , 'training sample'
    print 'load in ' , len(tst_rels) , 'test sample'

    token_list = []
    for sena, senb in zip(ledus,redus):
        token_list.extend(sena)
        token_list.extend(senb)

    # collect discourse relations
    relation_list = []
    relation_list.extend(rels)
    relation_list.extend(tst_rels)

    # 
    rel_freq = nltk.FreqDist(relation_list)
    print 'Found %d unique discourse relations . ' % len(rel_freq.items())

    disrel_num = len(rel_freq.items())
    rel_vocab = rel_freq.most_common(disrel_num)
    index_to_relation = [x[0] for x in rel_vocab]
    relation_to_index = dict([(r,i) for i,r in enumerate(index_to_relation)])
    print index_to_relation
    print relation_to_index


    # 
    # code-snippet-2 build general word vocabulary
    #
    word_freq = nltk.FreqDist(token_list)
    print 'Found %d unique words tokens . ' % len(word_freq.items())
    vocabulary_size = len(word_freq)  
    unknown_token = 'UNK'
    vocab = word_freq.most_common(vocabulary_size)
    index_to_word = []
    index_to_word.append('E_O_E')
    index_to_word.extend([x[0] for x in vocab])
    # load in frequent word in common sentences
    freq_word = load_freq_word('../freq_word/freq_word')
    index_to_word.extend(freq_word)
    index_to_word.append(unknown_token)
    # remove the reduplicate word
    index_to_word = list(OrderedDict.fromkeys(index_to_word))

    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    print 'Using vocabulary size %d. ' % len(index_to_word )
    print "the least frequent word in our vocabulary is '%s' and appeared %d times " % (vocab[-1][0],vocab[-1][1])


    # code-snippet-3 build general training dataset
    # training dataset
    # 
    for i,(edua,edub) in enumerate(zip(ledus,redus)):
        ledus[i] = [w if w in word_to_index else unknown_token for w in edua]
        redus[i] = [w if w in word_to_index else unknown_token for w in edub]
    for i,rel in enumerate(rels):
        rels[i] = [relation_to_index[rel]]

    # test dataset
    for i,(edua,edub) in enumerate(zip(tst_ledus,tst_redus)):
        tst_ledus[i] = [w if w in word_to_index else unknown_token for w in edua]
        tst_redus[i] = [w if w in word_to_index else unknown_token for w in edub]

    for i,rel in enumerate(tst_rels):
        tst_rels[i] = [relation_to_index[rel]]

    # X_1_train , X_2_train , y_train
    X_1_train = np.asarray([[word_to_index[w] for w in sent ] for sent in ledus])
    X_2_train = np.asarray([[word_to_index[w] for w in sent ] for sent in redus])
    X_1_train_nu, X_1_train_mask = prepare_data(X_1_train,maxlen)
    X_2_train_nu, X_2_train_mask = prepare_data(X_2_train,maxlen)
    y_train = (rels)

    # X_1_test, X_2_test , y_train
    X_1_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_ledus])
    X_2_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_redus])
    X_1_test_nu, X_1_test_mask = prepare_data(X_1_test,maxlen)
    X_2_test_nu, X_2_test_mask = prepare_data(X_2_test,maxlen)
    y_test = (tst_rels)


    print "Example sentence '%s' " % " ".join(ledus[0])
    print "Example sentence '%s' " % " ".join(redus[0])
    print "Example sentence after Pre-processing : '%s' " % X_1_train[0]
    print "Example sentence after Pre-processing : '%s' " % X_2_train[0]
    print "Example sentence mask after masked : '%s' " % X_1_train_mask[0]
    print "Example sentence mask after masked : '%s' " % X_2_train_mask[0]
    print "Example label : ", y_train[0]
    print ""


    # build Embedding matrix
    label_size = 18
    wvdic = load_word_embedding('../data/glove.6B.300d.txt')
    word_dim = wvdic.values()[0].shape[0]
    E = build_we_matrix(wvdic,index_to_word,word_to_index,word_dim)


    hidden_dim = 100
    print 'now build model ...'
    print 'hidden dim : ' , hidden_dim
    print 'word dim : ' , word_dim
    print 'vocabulary size : ' , len(index_to_word)

    model = framework(word_dim,label_size,vocabulary_size,hidden_dim=hidden_dim,word_embedding=E,bptt_truncate=-1)


    # Print SGD step time
    t1 = time.time()
    print X_1_train[0]
    print X_2_train[0]
    print 'attention weight : '
    # a_att, b_att = model.comsen(X_1_train[0],X_2_train[0])
    # print a_att
    # print b_att
    output = model.predict_class(X_1_train[0],X_1_train_mask[0],X_2_train[0],X_2_train_mask[0])
    t_s = model.t_score(X_1_train[0],X_1_train_mask[0],X_2_train[0],X_2_train_mask[0])
    print 'predict_class : ' , output
    print 'tensor_score : ' , t_s
    print 'ce_error : ' , model.ce_error(X_1_train[0],X_1_train_mask[0],X_2_train[0],X_2_train_mask[0],y_train[0])
    learning_rate = 0.000005

    model.sgd_step(X_1_train[0],X_1_train_mask[0],X_2_train[0],X_2_train_mask[0],y_train[0])
    t2 = time.time()

    print "SGD Step time : %f milliseconds " % ((t2-t1)*1000.)
    sys.stdout.flush()

    # 
    NEPOCH = 100

    for epoch in range(NEPOCH):

        print 'this is epoch : ' , epoch
        train_with_sgd(model,X_1_train,X_2_train,X_1_train_mask,X_2_train_mask,y_train,X_1_test,X_2_test,y_test,learning_rate=learning_rate,nepoch=1,decay=0.9,index_to_word=index_to_word,index_to_relation=index_to_relation)
        test_score(model,X_1_test,X_2_test,X_1_test_mask,X_2_test_mask,y_test,index_to_word=index_to_word,index_to_relation=index_to_relation)



if __name__ == '__main__':
    relation();
