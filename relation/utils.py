
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
            print 'weight L2 reg ' , model.L2_reg();
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

