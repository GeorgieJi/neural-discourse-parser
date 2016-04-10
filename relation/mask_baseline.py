'''
A Fast and Accurate Discourse Parser using Neural Networks
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

        W_att = np.random.uniform(-np.sqrt(1./hidden_dim*2),np.sqrt(1./hidden_dim*2),(hidden_dim,hidden_dim*2))
        v_att = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim))
        b_att = np.zeros((hidden_dim))

        # initialize the soft attention parameters
        # basically the soft attention is the single hidden layer 
        # no idea how to set the attention layer hidden node dim just set it as hidden dim for now


        # Created shared variable
        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))

        # Created attention variable
        self.W_att = theano.shared(name='W_att',value=W_att.astype(theano.config.floatX))
        self.v_att = theano.shared(name='v_att',value=v_att.astype(theano.config.floatX))
        self.b_att = theano.shared(name='b_att',value=b_att.astype(theano.config.floatX))


        self.params = [self.U,self.W,self.b,self.W_att,self.v_att,self.b_att]

        # store the theano graph 
        # self.theano = {}
        # self.__theano_build__()
 
    def recurrent(self,x_s,x_s_m,E):
        U, W, b, W_att, v_att, b_att = self.U, self.W, self.b, self.W_att, self.v_att, self.b_att
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

        h_s = T.concatenate([s_f,s_b],axis=1)

        def soft_attention(h_i,v_att,W_att,b_att):
            return v_att.dot(T.tanh(W_att.dot(h_i)+b_att))

        def weight_attention(h_i,a_j):
            return h_i*a_j

        h_att, updates = theano.scan(
                soft_attention,
                sequences=[h_s,v_att,W_att,b_att]
                )

        h_att = T.exp(h_att)
        h_att = h_att.flatten()
        h_att = h_att / h_att.sum()

        h_s_att, updates = theano.scan(
                weight_attention,
                sequences=[h_s,h_att]
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
        v_a = gru_layer.recurrent(x_a,x_a_m,self.E)
        v_b = gru_layer.recurrent(x_b,x_b_m,self.E)

        edu_pair_fea = T.concatenate([v_a,v_b],axis=0)

        # build hidden_layer for edu pair
        mlp_layer_1 = HiddenLayer(hidden_dim*4,label_dim)
        # mlp_layer_2 = HiddenLayer(hidden_dim,label_dim)
        ep_fea_2 = mlp_layer_1.forward(edu_pair_fea)
        # ep_fea_3 = mlp_layer_2.forward(ep_fea_2)

        # softmax 
        o = T.nnet.softmax(ep_fea_2)[0]

        # trick for prevent nan
        eps = np.asarray([1.0e-10]*label_dim,dtype=theano.config.floatX)
        o = o + eps
        om = o.reshape((1,o.shape[0]))
        prediction = T.argmax(om,axis=1)
        o_error = T.nnet.categorical_crossentropy(om,y)

        # cost
        cost = T.sum(o_error)

        # collect the params from each model
        self.params = []
        self.params = self.params + [ self.E ] 
        self.params = self.params + gru_layer.params + mlp_layer_1.params 


        # please verify the parameters of model
        print 'please verify the parameter of model'
        print self.params
        print len(self.params)

        # updates
        updates = sgd_updates_adadelta(norm=0,params=self.params,cost=cost)


        # framework assign function
        self.predict = theano.function([x_a,x_a_m,x_b,x_b_m],prediction)
        self.predict_class = theano.function([x_a,x_a_m,x_b,x_b_m],prediction)
        self.ce_error = theano.function([x_a,x_a_m,x_b,x_b_m,y],cost)
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
    index_to_word = [x[0] for x in vocab]

    # load in frequent word in common sentences
    freq_word = load_freq_word('../freq_word/freq_word')
    index_to_word.extend(freq_word)
    # remove the reduplicate word
    index_to_word = list(set(index_to_word))


    index_to_word.append(unknown_token)
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
    X_1_train, X_1_train_mask = prepare_data(X_1_train,maxlen)
    X_2_train, X_2_train_mask = prepare_data(X_2_train,maxlen)
    y_train = (rels)

    # X_1_test, X_2_test , y_train
    X_1_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_ledus])
    X_2_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_redus])
    X_1_test, X_1_test_mask = prepare_data(X_1_test,maxlen)
    X_2_test, X_2_test_mask = prepare_data(X_2_test,maxlen)
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


    hidden_dim = 200
    
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
    print 'predict_class : ' , output
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
