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
# from utils import *

def train_with_sgd(model,X_1_train,X_2_train,X_1_train_mask,X_2_train_mask,y_train,X_1_test,X_2_test,y_test,learning_rate=0.001,nepoch=20,decay=0.9,index_to_word=[],index_to_relation=[]):
    
    num_examples_seen = 0
    print 'now learning_rate : ' , learning_rate;
    for epoch in range(nepoch):
        # For each training example ...

        tocount = 0
        tccount = 0
        tycount = 0

        for i in np.random.permutation(len(y_train)):
            # One SGT step
            num_examples_seen += 1
            # Optionally do callback
            model.sgd_step(X_1_train[i],X_1_train_mask[i],X_2_train[i],X_2_train_mask[i],y_train[i]) 
            print 'the number of example have seen for now : ' , num_examples_seen
            output = model.predict_class(X_1_train[i],X_1_train_mask[i],X_2_train[i],X_2_train_mask[i])
            print '>>>>> case'
            lwrds = [index_to_word[j] for j in X_1_train[i]]
            rwrds = [index_to_word[j] for j in X_2_train[i]]
            # a_att, b_att = model.comsen(X_1_train[i],X_2_train[i])
            print 'i-th :' , i;
            print 'the left edu : '
            print " ".join(lwrds)
            print 'the right edu : '
            print " ".join(rwrds)
            print 'predict : ' , model.predict(X_1_train[i],X_1_train_mask[i],X_2_train[i],X_2_train_mask[i])
            print 'ce_error : ' , model.ce_error(X_1_train[i],X_1_train_mask[i],X_2_train[i],X_2_train_mask[i],y_train[i])
            print 'predict_relation : ' , output
            print index_to_relation[output[0]]
            print 'true relation : ' , y_train[i]
            print index_to_relation[y_train[i][0]]

            ocount = 0
            ccount = 0
            ycount = 0

            for o,y in zip(output,y_train[i]):
                if o == y:
                    print 'correct prediction'
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


def test_score(model,X_1_test,X_2_test,X_1_test_mask,X_2_test_mask,y_test,index_to_word,index_to_relation):
    print 'now score the test dataset'
    scores = [];
    tocount = 0
    tccount = 0
    tycount = 0

    for i in range(len(y_test)):
        output = model.predict_class(X_1_test[i],X_1_test_mask[i],X_2_test[i],X_2_test_mask[i])
        ocount = 0
        ccount = 0
        ycount = 0
        lwrds = [index_to_word[j] for j in X_1_test[i]]
        rwrds = [index_to_word[j] for j in X_2_test[i]]
        
        print 'i-th : ' , i;
        print 'the left edu : , ' , " ".join(lwrds)
        print 'the right edu : ' , " ".join(rwrds)
        print 'ce_error : ' , model.ce_error(X_1_test[i],X_1_test_mask[i],X_2_test[i],X_2_test_mask[i],y_test[i])

        # print 
        print 'predict relation : ' , output
        print index_to_relation[output[0]]
        print 'true relation : ' , y_test[i]
        print index_to_relation[y_test[i][0]]


        for o,y in zip(output,y_test[i]):
            if y == o:
                print 'the correct prediction!'
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

def load_freq_word(path):

    sens = open(path).readlines()
    fword = [];
    for sen in sens:
        fword.append(sen.strip())

    return fword


def sgd_updates_adadelta(norm,params,cost,rho=0.95,epsilon=1e-9,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value(),dtype=theano.config.floatX)
        exp_sqr_grads[param] = theano.shared(value=(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)

    # this is place you should think of gradient clip using the l2-norm
    g2 = 0.
    clip_c = 1.
    for g in gparams:
        g2 += (g**2).sum()
    # is_finite = T.or_(T.isnan(g2), T.isinf(g2))
    new_grads = []
    for g in gparams:
        new_grad = T.switch(g2>(clip_c**2),g/T.sqrt(g2)*clip_c,g)
        new_grads.append(new_grad)
    gparams = new_grads

    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if norm == 1:
            if (param.get_value(borrow=True).ndim == 2) and param.name!='Words':
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param
        elif norm == 0:
            updates[param] = stepped_param
        else:
            updates[param] = stepped_param
    return updates


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

    parentpair = [ [left_tree[1]+'-'+right_tree[1] , leftspan, rightspan] ]
    parentpair.extend(leftpair)
    parentpair.extend(rightpair)
    # print '->' , parentpair
    
    return parentpair, leftspan + ' ' + rightspan


def extract_nucleus(tree_str):

    tree = parse_tree(tree_str)
    if tree == ')//TT_ERR':
        pairs = []
        pass
    else:
        pairs = traversal(tree)[0]

    return pairs

def relation_mapping(subrelation):

    relsmap = dict()

    relsmap['Attribution'] = ['attribution','attribution-negative']
    relsmap['Background'] = ['background','circumstance']
    relsmap['Cause'] = ['cause','result','consequence']
    relsmap['Comparsion'] = ['comparison','preference','analogy','proportion']
    relsmap['Condition'] = ['condition','hypothetical','contingency','otherwise']
    relsmap['Contrast'] = ['contrast','concession','antithesis']
    relsmap['Elaboration'] = ['elaboration-additional','elaboration-general-specific','elaboration-part-whole','elaboration-process-step','elaboration-object-attribute','elaboration-set-member','example','definition']
    relsmap['Enablement'] = ['purpose','enablement']
    relsmap['Evaluation'] = ['evaluation','interpretation','conclusion','comment']
    relsmap['Explanation'] = ['evidence','explanation-argumentative','reason']
    relsmap['Joint'] = ['list','disjunction']
    relsmap['Manner-Means'] = ['manner','means']
    relsmap['Topic-Comment'] = ['problem-solution','question-answer','statement-response','topic-comment','comment-topic','rhetorical-question']
    relsmap['Summary'] = ['summary','restatement']
    relsmap['Temporal'] = ['temporal-before','temporal-after','temporal-same-time','sequence','inverted-sequence']
    relsmap['TopicChange'] = ['topic-shift','topic-drift']
    relsmap['TextualOrganization'] = ['textualorganization']
    relsmap['Same-Unit'] = ['same-unit']

    mappingrel = 'RelationError'
    for mrel in relsmap:
        for srel in relsmap[mrel]:
            if srel in subrelation.lower():
                mappingrel = mrel

    if mappingrel == 'RelationError':
        raise Exception('Discourse Relation Mapping Error!')

    return mappingrel

#
#
#
def build_data(dir_path,maxlen):
    
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
    disrels = []

    for pair in pairs:

        pair[1] = pair[1].strip().replace('<P>',' p_end ')
        pair[2] = pair[2].strip().replace('<P>',' p_end ')

        wrdsa = nltk.word_tokenize(pair[1].strip().lower())
        wrdsb = nltk.word_tokenize(pair[2].strip().lower())

        # only focus on 18 discourse relations
        rel = relation_mapping(pair[0].lower()).lower()

        # replace the < P > with 'p-end'
        wrdsa = (" ".join(wrdsa)).replace('< p >','p-end').split()
        wrdsb = (" ".join(wrdsb)).replace('< p >','p-end').split()

        # wrdsa.insert(0,'B_O_E')
        # wrdsb.insert(0,'B_O_E')
        wrdsa.append('E_O_E')
        wrdsb.append('E_O_E')


        if len(wrdsa) < maxlen and len(wrdsb) < maxlen:
            senas.append(wrdsa)
            senbs.append(wrdsb)
            disrels.append(rel)
        else:
            continue

    
    return senas , senbs , disrels

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
        b_att = np.zeros(1)
        sv_att = np.random.uniform(-np.sqrt(1./hidden_dim*2),np.sqrt(1./hidden_dim*2),(hidden_dim*2))

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
        self.sv_att = theano.shared(name='sv_att',value=sv_att.astype(theano.config.floatX))


        # self.params = [self.U,self.W,self.b,self.W_att,self.v_att,self.b_att,self.sv_att]
        self.params = [self.U,self.W,self.b,self.sv_att,self.b_att]

        # store the theano graph 
        # self.theano = {}
        # self.__theano_build__()
 
    def recurrent(self,x_s,x_s_m,E):
        U, W, b, W_att, v_att, b_att, sv_att = self.U, self.W, self.b, self.W_att, self.v_att, self.b_att, self.sv_att
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

        # 
        # only use a very simple vector to do so 
        # 
        def score_attention(h_i):
            return T.tanh(sv_att.dot(h_i)+b_att)

        # wrong function !
        # def soft_attention(h_i,v_att,W_att,b_att):
        #     return v_att.dot(T.tanh(W_att.dot(h_i)+b_att))

        def weight_attention(h_i,a_j):
            return h_i*a_j

        h_att, updates = theano.scan(
                score_attention,
                sequences=[h_s]
                )

        # score exp
        h_att = T.exp(h_att)
        h_att = h_att.flatten()

        # using mask on h_attention score
        h_att = h_att * x_s_m

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


    hidden_dim = 300
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
