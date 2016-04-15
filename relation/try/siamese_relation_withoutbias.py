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
# from theano.compile.nanguardmode import NanGuardMode
from collections import OrderedDict

sys.path.append('../tree')
from tree import *

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

        wrdsa.insert(0,'B_E')
        wrdsb.insert(0,'B_E')
        wrdsa.append('E_E')
        wrdsb.append('E_E')


        if len(wrdsa) < 50 and len(wrdsb) < 50:
            senas.append(wrdsa)
            senbs.append(wrdsb)
            disrels.append(rel)
        else:
            continue
    
    return senas , senbs , disrels



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
        self.label_dim = label_dim

        # initialize the network parameters

        if word_embedding is None:
            E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,vocab_size))
        else:
            E = word_embedding

        U = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(3,hidden_dim,word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(3,hidden_dim,hidden_dim))
        b = np.zeros((3,hidden_dim))
        
        V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(label_dim,hidden_dim*2))
        c = np.zeros(label_dim)

        # Created shared variable
        self.E = theano.shared(name='E',value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V',value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c',value=c.astype(theano.config.floatX))


        self.params = [self.E,self.U,self.W,self.V,self.c]
        

        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b , self.c

        x_a = T.ivector('x_a')
        x_b = T.ivector('x_b')
        y = T.lvector('y')

        def forward_step(x_t,s_t_prev):
            # Word embedding layer
            x_e = E[:,x_t]
            # GRU layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e)+W[0].dot(s_t_prev))
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e)+W[1].dot(s_t_prev))
            c_t = T.tanh(U[2].dot(x_e)+W[2].dot(s_t_prev*r_t))
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            # directly return the hidden state as intermidate output 
            return [s_t]


        # sentence a vector (states)
        a_s , updates = theano.scan(
                forward_step,
                sequences=x_a,
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))
            
        # sentence b vector (states)
        b_s , updates = theano.scan(
                forward_step,
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

        # in case the o contains 0 which cause inf
        eps = np.asarray([1.0e-10]*self.label_dim,dtype=theano.config.floatX)
        o = o + eps
        om = o.reshape((1,o.shape[0]))
        prediction = T.argmax(om,axis=1)
        o_error = T.nnet.categorical_crossentropy(om,y)


        # cost 
        cost = T.sum(o_error)

        # updates
        updates = sgd_updates_adadelta(norm=0,params=self.params,cost=cost)

        # monitor parameter
        mV = V * T.ones_like(V)
        mc = c * T.ones_like(c)
        mU = U * T.ones_like(U)
        mW = W * T.ones_like(W)

        gV = T.grad(cost,V)
        gc = T.grad(cost,c)
        gU = T.grad(cost,U)
        gW = T.grad(cost,W)

        mgV = gV * T.ones_like(gV)
        mgc = gc * T.ones_like(gc)
        mgU = gU * T.ones_like(gU)
        mgW = gW * T.ones_like(gW)




        # Assign functions
        self.monitor = theano.function([x_a,x_b],[sena,senb,mV,mc,mU,mW])
        self.monitor_grad = theano.function([x_a,x_b,y],[mgV,mgc,mgU,mgW])
        self.predict = theano.function([x_a,x_b],om)
        self.predict_class = theano.function([x_a,x_b],prediction)
        self.ce_error = theano.function([x_a,x_b,y],cost)
        # self.bptt = theano.function([x,y],[dE,dU,dW,db,dV,dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        # find the nan
        self.sgd_step = theano.function(
                [x_a,x_b,y],
                [],
                updates=updates
                # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )


def index_to_class(index):
    label = '';
    if index == 0:
        label = 'Nucleus-Satellite'
    elif index == 1:
        label = 'Nucleus-Nucleus'
    elif index == 2:
        label = 'Satellite-Nucleus'
    return label

def train_with_sgd(model,X_1_train,X_2_train,y_train,X_1_test,X_2_test,y_test,learning_rate=0.001,nepoch=20,decay=0.9,index_to_word=[],index_to_relation=[]):
    
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
            model.sgd_step(X_1_train[i],X_2_train[i],y_train[i]) 
            print 'the number of example have seen for now : ' , num_examples_seen
            output = model.predict_class(X_1_train[i],X_2_train[i])
            print '>>>>> case'
            lwrds = [index_to_word[j] for j in X_1_train[i]]
            rwrds = [index_to_word[j] for j in X_2_train[i]]
            print 'i-th :' , i;
            print 'the left edu : '
            print " ".join(lwrds)
            print 'the right edu : '
            print " ".join(rwrds)
            print 'predict : ' , model.predict(X_1_train[i],X_2_train[i])
            print 'ce_error : ' , model.ce_error(X_1_train[i],X_2_train[i],y_train[i])
            print 'predict_relation : ' , output
            print index_to_relation[output[0]]
            print 'true relation : ' , y_train[i]
            print index_to_relation[y_train[i][0]]

            ocount = 0
            ccount = 0
            ycount = 0
            if False:
                test_score(model,X_1_test,X_2_test,y_test,index_to_word=index_to_word)

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


def test_score(model,X_1_test,X_2_test,y_test,index_to_word,index_to_relation):
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



def relation():

    ledus, redus , rels = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING');

    tst_ledus, tst_redus, tst_rels = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TEST')
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
    y_train = (rels)

    # X_1_test, X_2_test , y_train
    X_1_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_ledus])
    X_2_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_redus])
    y_test = (tst_rels)

    print "Example sentence '%s' " % " ".join(ledus[0])
    print "Example sentence '%s' " % " ".join(redus[0])
    print "Example sentence after Pre-processing : '%s' " % X_1_train[0]
    print "Example sentence after Pre-processing : '%s' " % X_2_train[0]
    print "Example label : ", y_train[0]
    print ""


    # build Embedding matrix
    label_size = 18
    wvdic = load_word_embedding('../data/glove.6B.200d.txt')
    word_dim = wvdic.values()[0].shape[0]

    E = build_we_matrix(wvdic,index_to_word,word_to_index,word_dim)

    model = Siamese_GRU(word_dim,label_size,vocabulary_size,hidden_dim=200,word_embedding=E,bptt_truncate=-1)

    # Print SGD step time
    t1 = time.time()

    output = model.predict_class(X_1_train[0],X_2_train[0])
    print 'predict_class : ' , output
    print 'ce_error : ' , model.ce_error(X_1_train[0],X_2_train[0],y_train[0])
    learning_rate = 0.000005

    model.sgd_step(X_1_train[0],X_2_train[0],y_train[0])
    t2 = time.time()

    print "SGD Step time : %f milliseconds " % ((t2-t1)*1000.)
    sys.stdout.flush()

    # 
    NEPOCH = 100

    for epoch in range(NEPOCH):

        print 'this is epoch : ' , epoch
        train_with_sgd(model,X_1_train,X_2_train,y_train,X_1_test,X_2_test,y_test,learning_rate=learning_rate,nepoch=1,decay=0.9,index_to_word=index_to_word,index_to_relation=index_to_relation)

        test_score(model,X_1_test,X_2_test,y_test,index_to_word=index_to_word,index_to_relation=index_to_relation)



if __name__ == '__main__':
    relation();
