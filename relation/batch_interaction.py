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

def build_batch_set(x1,x1m,x2,x2m,y,batch_size):

    total_n = len(x1)
    print 'total number of dataset : ' , total_n

    if total_n < batch_size:
        print 'warning : the total number of dataset is less than the batch size'

    batch_num = 0
    remain_num = 0

    batch_num = total_n / batch_size
    remain_num = total_n % batch_size

    samples = []

    for i in range(batch_num):
        tmp = []
        tmp.append(x1[i*batch_size:(i+1)*batch_size])
        tmp.append(x1m[i*batch_size:(i+1)*batch_size])
        tmp.append(x2[i*batch_size:(i+1)*batch_size])
        tmp.append(x2m[i*batch_size:(i+1)*batch_size])
        tmp.append(y[i*batch_size:(i+1)*batch_size])
        samples.append(tmp)

    if remain_num != 0:
        tmp = []
        tmp.append(x1[batch_num*batch_size:])
        tmp.append(x1m[batch_num*batch_size:])
        tmp.append(x2[batch_num*batch_size:])
        tmp.append(x2m[batch_num*batch_size:])
        tmp.append(y[batch_num*batch_size:])
        samples.append(tmp)


    return samples

        


def train_with_sgd(model,train_samples,index_to_word=[],index_to_relation=[]):
    
    num_examples_seen = 0
    # For each training example ...
    
    tocount = 0
    tccount = 0
    tycount = 0
    
    for i in np.random.permutation(len(train_samples)):
        # One SGT step
        num_examples_seen += len(train_samples[i][0])
        X_1_train = train_samples[i][0]
        X_1_train_mask = train_samples[i][1]
        X_2_train = train_samples[i][2]
        X_2_train_mask = train_samples[i][3]
        y_train = train_samples[i][4]
        
        # Optionally do callback
        model.sgd_step(X_1_train,X_1_train_mask,X_2_train,X_2_train_mask,y_train) 
        output = model.predict_class(X_1_train,X_1_train_mask,X_2_train,X_2_train_mask)
        
        for i in range(len(X_1_train)):
            lwrds = [index_to_word[j] for j in X_1_train[i]]
            rwrds = [index_to_word[j] for j in X_2_train[i]]
            print '---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
            # a_att, b_att = model.comsen(X_1_train[i],X_2_train[i])
            print 'i-th :' , i; 
            print 'the left edu : '
            print " ".join(lwrds)
            print 'the right edu : '
            print " ".join(rwrds)
            print 'predict_relation : ' , output[i]
            print index_to_relation[output[i]]
            print 'true relation : ' , y_train[i]
            print index_to_relation[y_train[i]]
            if output[i] == y_train[i]:
                print 'correct prediction'
            else:
                print 'wrong prediction'
        
        print 'the number of example have seen for now : ' , num_examples_seen
        # 
        ocount = 0
        ccount = 0
        ycount = 0
        for o,y in zip(output,y_train):
            if o == y:
                ccount += 1
            ycount += 1
            ocount += 1
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

def test_score(model,train_samples,index_to_word=[],index_to_relation=[]):
    
    num_examples_seen = 0
    # For each training example ...
    
    tocount = 0
    tccount = 0
    tycount = 0
    print 'now score the test set performance ... '
    for i in range(len(train_samples)):
        # One SGT step
        num_examples_seen += len(train_samples[i][0])
        X_1_train = train_samples[i][0]
        X_1_train_mask = train_samples[i][1]
        X_2_train = train_samples[i][2]
        X_2_train_mask = train_samples[i][3]
        y_train = train_samples[i][4]
        
        # Optionally do callback
        # model.sgd_step(X_1_train,X_1_train_mask,X_2_train,X_2_train_mask,y_train) 
        # print 'the number of example have seen for now : ' , num_examples_seen
        output = model.predict_class(X_1_train,X_1_train_mask,X_2_train,X_2_train_mask)
        
        for i in range(len(X_1_train)):
            lwrds = [index_to_word[j] for j in X_1_train[i]]
            rwrds = [index_to_word[j] for j in X_2_train[i]]
            print '---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
            # a_att, b_att = model.comsen(X_1_train[i],X_2_train[i])
            print 'i-th :' , i; 
            print 'the left edu : '
            print " ".join(lwrds)
            print 'the right edu : '
            print " ".join(rwrds)
            print 'predict_relation : ' , output[i]
            print index_to_relation[output[i]]
            print 'true relation : ' , y_train[i]
            print index_to_relation[y_train[i]]
            if output[i] == y_train[i]:
                print 'correct prediction'
            else:
                print 'wrong prediction'
        
        # 
        ocount = 0
        ccount = 0
        ycount = 0
        for o,y in zip(output,y_train):
            if o == y:
                ccount += 1
            ycount += 1
            ocount += 1
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

    print 'Accuracy of test set: ' , precision
    print 'Recall of test set: ' , recall
    print 'Fmeasure of test set: ' , Fmeasure


    return model



def _test_score(model,X_1_test,X_2_test,y_test,index_to_word,index_to_relation):
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
        lwrds = [index_to_word[j] for j in X_1_test[i]]
        rwrds = [index_to_word[j] for j in X_2_test[i]]
        
        print 'i-th : ' , i;
        print 'the left edu : , ' , " ".join(lwrds)
        print 'the right edu : ' , " ".join(rwrds)
        print 'ce_error : ' , model.ce_error(X_1_test[i],X_2_test[i],y_test[i])

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
                 activation=None):

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
        lin_output = T.dot(x,W) + b[None,:]
        output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        return output

def ortho_weight(ndim):
    W = np.random.randn(ndim,ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def init_weight(nin,nout,scale=0.01):
    W = scale * np.random.randn(nin,nout)
    return W.astype('float32')


class gru_tensor:

    def __init__(self,word_dim,hidden_dim,bptt_truncate=-1):

        # assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # 
        W = np.concatenate([init_weight(word_dim,hidden_dim),init_weight(word_dim,hidden_dim)],axis=1)
        U = np.concatenate([ortho_weight(hidden_dim),ortho_weight(hidden_dim)],axis=1)
        b = np.zeros((2*hidden_dim)).astype('float32')
        Wx = init_weight(word_dim, hidden_dim)
        Ux = ortho_weight(hidden_dim)
        bx = np.zeros((hidden_dim)).astype('float32')

        # Created shared variable
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))
        self.Wx = theano.shared(name='Wx',value=Wx.astype(theano.config.floatX))
        self.Ux = theano.shared(name='Ux',value=Ux.astype(theano.config.floatX))
        self.bx = theano.shared(name='bx',value=bx.astype(theano.config.floatX))


        # self.params = [self.U,self.W,self.b,self.W_att,self.v_att,self.b_att,self.sv_att]
        self.params = [self.U,self.W,self.b,self.Ux,self.Wx,self.bx]
 
    def recurrent(self,x_s,x_m,E):
        U, W, b, Wx, Ux, bx = self.U, self.W, self.b, self.Wx, self.Ux, self.bx
        word_dim, hidden_dim = self.word_dim, self.hidden_dim


        # x_s -> x_s_emb
        # (n_samples,nsteps) -> (nsteps,n_samples)
        xs = x_s         
        xm = x_m

        n_steps = xs.shape[0] #
        word_dim = E.shape[0]
        n_samples = xs.shape[1]

        emb = E[:,xs.flatten()]
        emb = emb.reshape([n_steps, n_samples, word_dim])

        state_below_ = T.dot(emb,W) + b.dimshuffle('x','x',0)
        state_belowx = T.dot(emb,Wx) + bx.dimshuffle('x','x',0)

        def _slice(_x,n,dim):
            return _x[:,n*dim:(n+1)*dim]

        def _step_slice(m_, x_, xx_, h_, U, Ux):
            preact = T.dot(h_,U)
            preact += x_

            # reset and update gates
            r = T.nnet.hard_sigmoid(_slice(preact,0,hidden_dim))
            u = T.nnet.hard_sigmoid(_slice(preact,1,hidden_dim))

            # compute the hidden state proposal
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_

            # hidden state proposal
            h = T.tanh(preactx)

            # leaky integrate and obtain next hidden state
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_

            return h

        seqs = [xm, state_below_, state_belowx]
        init_states = [T.alloc(0., n_samples, hidden_dim)]
        _step = _step_slice
        shared_vars = [U, Ux]

        rval, updates = theano.scan(
                _step,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=shared_vars,
                n_steps=n_steps
                # strict=True
                )

        
        return rval


def load_data():
    """
    load the dataset 
    warp them with theano shared object
    """

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
        # rels = [ [1] [2] [3]]
        rels[i] = relation_to_index[rel]

    # test dataset
    for i,(edua,edub) in enumerate(zip(tst_ledus,tst_redus)):
        tst_ledus[i] = [w if w in word_to_index else unknown_token for w in edua]
        tst_redus[i] = [w if w in word_to_index else unknown_token for w in edub]
    for i,rel in enumerate(tst_rels):
        tst_rels[i] = relation_to_index[rel]

    # X_1_train , X_2_train , y_train
    # convert it into matrix


    X_1_train = np.asarray([[word_to_index[w] for w in sent ] for sent in ledus])
    X_2_train = np.asarray([[word_to_index[w] for w in sent ] for sent in redus])
    X_1_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_ledus])
    X_2_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_redus])

    # maxlen = max([get_maxlen(X_1_train),get_maxlen(X_2_train),get_maxlen(X_1_test),get_maxlen(X_2_test)])
    # print 'maxlen : ' , maxlen

    X_1_train, X_1_train_mask = prepare_data(X_1_train,maxlen)
    X_2_train, X_2_train_mask = prepare_data(X_2_train,maxlen)
    y_train = (rels)
    # X_1_test, X_2_test , y_train
    # for shared variable strategy we need X_1_train to be matrix 
    X_1_test, X_1_test_mask = prepare_data(X_1_test,maxlen)
    X_2_test, X_2_test_mask = prepare_data(X_2_test,maxlen)
    y_test = (tst_rels)

    print 'build_data'
    print len(X_1_train)
    print len(X_2_train)
    print len(X_1_train_mask)
    print len(X_2_train_mask)

    print "Example sentence '%s' " % " ".join(ledus[0])
    print "Example sentence '%s' " % " ".join(redus[0])
    print "Example sentence after Pre-processing : '%s' " % X_1_train[0]
    print "Example sentence after Pre-processing : '%s' " % X_2_train[0]
    print "Example label : ", y_train[0]
    print ""
    train_set = [X_1_train, X_1_train_mask, X_2_train, X_2_train_mask, y_train]
    test_set = [X_1_test, X_1_test_mask, X_2_test, X_2_test_mask, y_test]


    return train_set, test_set, index_to_word, word_to_index, vocabulary_size, index_to_relation, relation_to_index, maxlen



def shared_dataset_int(data,borrow=True):

    """
    Function that loads the dataset into shared variables

    the reason we store our dataset in shared variables is to allow 
    theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime 
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_data = theano.shared(np.asarray(data,dtype=theano.config.floatX),borrow=borrow)
    # when storing data on the GPU it has to be stored as floats 
    # therefore we will store the labels as "floatX" as well .
    # But during our computations we need them as ints (word index and labels)
    # therefore instead of returning "" we will have to cast it to int. 
    # This little hack lets ous get around this issue
    return T.cast(shared_data,'int64')

def shared_dataset_float(data,borrow=True):
    shared_data = theano.shared(np.asarray(data,dtype=theano.config.floatX),borrow=borrow)
    return shared_data


# batch preparation

def get_maxlen(seqs_x):

    return max([len(s) for s in seqs_x])

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

def mapping_(x,y,lx,ly,word_dim):

    T_y = y.T
    xx = T.tile(x,(1,ly))
    yy = T.tile(T_y,(1,lx))

    xx = xx.reshape((word_dim,lx*ly))
    yy = (yy.reshape((ly*lx,word_dim))).T

    cxy = (T.concatenate([xx,yy],axis=0)).T

    return cxy
    
def mask_mapping_(x,y,lx,ly):

    x = x.reshape((1,lx))
    y = y.reshape((1,ly))

    T_y = y.T
    xx = T.tile(x,(1,ly))
    yy = T.tile(T_y,(1,lx))

    xx = xx.reshape((1,lx*ly))
    yy = (yy.reshape((ly*lx,1))).T

    cxy = T.concatenate([xx,yy],axis=0)
    cxy = cxy[0]*cxy[1]

    return cxy

class soft_attention_tensor:

    def __init__(self,hidden_dim):

        # assign instance variables
        self.hidden_dim = hidden_dim

        W_att = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim))
        b_att = np.zeros(1)

        # 
        self.W_att = theano.shared(name='W_att',value=W_att.astype(theano.config.floatX))
        self.b_att = theano.shared(name='b_att',value=b_att.astype(theano.config.floatX))


        # collect parameter
        self.params = [self.W_att, self.b_att]

    def soft_att(self,x_s,x_s_m):
        
        # 
        W_att, b_att, hidden_dim = self.W_att , self.b_att, self.hidden_dim
        h_att = T.tanh( T.dot(x_s,W_att) + b_att[:,None,None])

        # the h_att -> (1,50,10) / trade off dim-shuffle
        h_att = h_att[0]
        # (50,100)
        h_att = T.exp(h_att)
        
        # multiple mask
        h_att = h_att * x_s_m
        h_att = h_att / h_att.sum(axis=0)
        h_s_att = x_s * h_att[:,:,None]
        a_s = h_s_att.sum(axis=0)

        return a_s


class framework:
    """
    build all theano graph here , in here we can combine as many as nerual layer we need !
    """
    def  __init__(self,word_dim,label_dim,vocab_size,maxlen,hidden_dim=128,word_embedding=None,bptt_truncate=-1):


        n_steps = maxlen
        # n_samples = batch_size

        # load dataset as shared variables
        # train_set

        index = T.lscalar() # index to a [mini]batch
        x_1 = T.matrix('x_1',dtype='int64')
        x_1_m = T.matrix('x_1_m',dtype='float32')
        x_2 = T.matrix('x_2',dtype='int64')
        x_2_m = T.matrix('x_2_m',dtype='float32')
        y = T.lvector('y') # the labels of relation discourse
        
        # convert the input shape from (n_samples,n_steps) -> (n_steps,n_samples)
        x1 = x_1.T
        x1m = x_1_m.T
        x2 = x_2.T
        x2m = x_2_m.T

        x1r = x1[::-1]
        x1mr = x1m[::-1]
        x2r = x2[::-1]
        x2mr = x2m[::-1]

        # be careful about matrix

        # the frameword only holds the global word embedding 
        if word_embedding is None:
            E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,vocab_size))
        else:
            E = word_embedding
        self.E = theano.shared(name='E',value=E.astype(theano.config.floatX))

        # Gru tensor version layer for forward and backword
        forward_gru = gru_tensor(word_dim,hidden_dim,bptt_truncate=-1)
        backward_gru = gru_tensor(word_dim,hidden_dim,bptt_truncate=-1)


        x_1_f = forward_gru.recurrent(x1,x1m,self.E)
        x_1_b = backward_gru.recurrent(x1r,x1mr,self.E)

        x_2_f = forward_gru.recurrent(x2,x2m,self.E)
        x_2_b = backward_gru.recurrent(x2r,x2mr,self.E)


        # v_a/v_b -> (50, 100, 160)
        # x1m/x2m -> (50,100)
        s_1 = T.concatenate([x_1_f,x_1_b],axis=2)
        s_2 = T.concatenate([x_2_f,x_2_b],axis=2)


        # build across mapping
        # build horizontal direction matrix and vertical direction matrix

        sa = soft_attention_tensor(hidden_dim*2)
        s_1a = sa.soft_att(s_1,x1m) # (50,100,1)
        s_2a = sa.soft_att(s_2,x2m) # (50,100,1)

        # sb = soft_attention_tensor(hidden_dim*4)
        
        # a_b_mapping = sb.soft_att(a_b_map,a_b_mask_map)
        # right edu attention

        edu_pair_fea = T.concatenate([s_1a,s_2a],axis=1) # (100,160+160)

        # build hidden_layer for edu pair
        mlp_layer = HiddenLayer(hidden_dim*4,label_dim)
        # mlp_layer_2 = HiddenLayer(hidden_dim,label_dim)
        ep_fea_2 = mlp_layer.forward(edu_pair_fea)
        # ep_fea_3 = mlp_layer_2.forward(ep_fea_2)

        # softmax 
        o = T.nnet.softmax(ep_fea_2)

        # trick for prevent nan
        eps = T.ones_like(o) * 1.0e-10
        om = o + eps

        # 
        prediction = T.argmax(om,axis=1)
        o_error = T.nnet.categorical_crossentropy(om,y)

        # cost
        cost = T.sum(o_error)

        # collect the params from each model
        self.params = []
        self.params += [ self.E ]
        self.params += sa.params
        self.params += forward_gru.params
        self.params += backward_gru.params
        self.params += mlp_layer.params 


        # please verify the parameters of model
        print 'please verify the parameter of model'
        print self.params
        print len(self.params)

        # updates
        updates = sgd_updates_adadelta(norm=0,params=self.params,cost=cost)

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        # framework assign function
        self.predict = theano.function([x_1,x_1_m,x_2,x_2_m],prediction)
        self.predict_class = theano.function([x_1,x_1_m,x_2,x_2_m],prediction)
        self.ce_error = theano.function([x_1,x_1_m,x_2,x_2_m,y],cost)

        self.batch_ = theano.function([x_1,x_1_m,x_2,x_2_m],[x_1_f,x_1_b,s_1,s_1a,prediction])
        self.check_ = theano.function([x_1,x_1_m,x_2,x_2_m],[x1,x1m,x2,x2m])

        self.sgd_step = theano.function(
                [x_1,x_1_m,x_2,x_2_m,y],
                [],
                updates = updates
                )


def relation():

    # load data set
    train_set , test_set , index_to_word, word_to_index, vocabulary_size, index_to_relation, relation_to_index, maxlen = load_data()
    X_1_train, X_1_train_mask, X_2_train, X_2_train_mask, y_train = train_set
    X_1_test, X_1_test_mask, X_2_test, X_2_test_mask, y_test = test_set

    
    print 'load_data'
    print len(X_1_train)
    print len(X_2_train)
    print len(X_1_train_mask)
    print len(X_2_train_mask)



    # build Embedding matrix
    label_size = 18
    wvdic = load_word_embedding('../data/glove.6B.300d.txt')
    word_dim = wvdic.values()[0].shape[0]
    E = build_we_matrix(wvdic,index_to_word,word_to_index,word_dim)

    hidden_dim = 300
    batch_size = 30
    print 'now build model ...'
    print 'hidden dim : ' , hidden_dim
    print 'word dim : ' , word_dim
    print 'vocabulary size : ' , len(index_to_word)

    model = framework(word_dim,label_size,vocabulary_size,maxlen,hidden_dim=hidden_dim,word_embedding=E,bptt_truncate=-1)


    # Print SGD step time
    t1 = time.time()
    print X_1_train[0]
    print X_2_train[0]

    model.sgd_step(X_1_train[:10],X_1_train_mask[:10],X_2_train[:10],X_2_train_mask[:10],y_train[:10])
    learning_rate = 0.000005
    t2 = time.time()
    print "SGD Step time : %f milliseconds " % ((t2-t1)*1000.)
    sys.stdout.flush()

    # 
    NEPOCH = 100
    train_samples = build_batch_set(X_1_train, X_1_train_mask, X_2_train, X_2_train_mask, y_train, batch_size)
    test_samples = build_batch_set(X_1_test, X_1_test_mask, X_2_test, X_2_test_mask, y_test, batch_size)
    for epoch in range(NEPOCH):

        print 'this is epoch : ' , epoch

        # train_update(model,train_set,test_set,index_to_word,index_to_relation)
        # test_update(model,test_set,index_to_word,index_to_relation)

        train_with_sgd(model,train_samples,index_to_word=index_to_word,index_to_relation=index_to_relation)
        test_score(model,test_samples,index_to_word=index_to_word,index_to_relation=index_to_relation)



if __name__ == '__main__':
    relation();
