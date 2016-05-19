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

    parentpair = [ [1 , leftspan, rightspan] ]
    parentpair.extend(leftpair)
    parentpair.extend(rightpair)
    # print '->' , parentpair
    
    return parentpair, leftspan + ' ' + rightspan

def traversal_basic(tree):
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
        leftpair , leftspan = traversal_basic(left_tree)

    if len(right_tree) == 4:
        rightpair , rightspan = traversal_basic(right_tree)

    if len(left_tree) == 3:
        leftspan = left_tree[2]

    if len(right_tree) == 3:
        rightspan = right_tree[2]

    parentpair.extend(leftpair)
    parentpair.extend([1,leftspan,rightspan])
    parentpair.extend(rightpair)
    # print '->' , parentpair
    
    return parentpair, ""


def verify_context(ledu,redu,pairs):

    sentence = pairs[0][1] + ' ' + pairs[0][2]
    return ( ledu +' '+ redu in sentence )


def generate_example(pairs,beduspairs):

    # extract basic edu
    # build the structure path

    # collect all edus
    edus = []
    bedus = []

    # pairs contains edu pair from bottom to top , the pairs[0] is the most top pair
    sentence = pairs[0][1] + ' ' + pairs[0][2]
    edus.append(sentence)

    for pair in pairs:
        edus.append(pair[1] + " " + pair[2])

    # collect all basic edu
    for item in beduspairs:
        # print pair
        if item != '' and item != 1:
            bedus.append(item)

    e_mark = 'S_R'
    edu_stack = copy.deepcopy(bedus)
    edu_stack.append(e_mark)
    
    order_edus = edu_stack

    levels = []
    update_edus = []
    examples = []

    sr_stack = []
    if len(order_edus) < 3:
        return []

    # initial the shift reduce stack
    sr_stack.append(order_edus.pop(0))
    sr_stack.append(order_edus.pop(0))

    while True:
        
        # perform shift reduce on a 3 items discourse tree
        # consider global strategy during building process
        # e.g. e1 .reduce  e2 .reduce e3 .reduce e4 []
        # how to generate negative sampl`1e
        
        
        if len(sr_stack) < 2:
            sr_stack.append(order_edus.pop(0))
            # print 'sr_stack : ' ,  sr_stack
            continue

        stk_1 = sr_stack[-2]
        stk_2 = sr_stack[-1]

        if len(sr_stack) <= 2 and len(order_edus) == 1 :
            example = [stk_1,stk_2,order_edus[0],'reduce']
            examples.append(example)
            break
        t_ = stk_1 + ' ' + stk_2 
        reduceflag = False

        if len(order_edus) != 1 : 
            top_edu = order_edus.pop(0)
            if ( t_ in edus ) and verify_context(t_,top_edu,pairs):
                reduceflag = True
        else:
            reduceflag = True
            top_edu = e_mark

        if reduceflag:
            # reduce
            example = [stk_1,stk_2,top_edu,'reduce']
            examples.append(example)
            sr_stack.pop()
            sr_stack.pop()
            n_edu = stk_1 + ' ' + stk_2
            if top_edu != e_mark:
                order_edus.insert(0,top_edu)
            order_edus.insert(0,n_edu)
            continue
        else:
            # shift
            example = [stk_1,stk_2,top_edu,'shift']
            examples.append(example)
            sr_stack.append(top_edu)
            continue


    return examples

def extract_edus(tree_str):
    tree = parse_tree(tree_str)
    # print 'parsed pairs : '
    pair = traversal(tree)[0]
    # generate_example(pairs)
    return pair

def extract_basic_edus(tree_str):
    tree = parse_tree(tree_str)
    pair = traversal_basic(tree)[0]
    return pair

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

    print 'trees number : ' , len(trees)
    # print trees[1]
    groups = []
    basic_groups = []
    for tree in trees:
        ftrees = extract_edus(tree)
        if ftrees[0][1] == '' and ftrees[0][2] == '':
            pass
        else:
            groups.append(ftrees)
            basic_groups.append(extract_basic_edus(tree))

    # print 'number of pairs', len(groups) , ''
    # print 'group ' , groups[242]
    # print 'bgroup ' , basic_groups[242]

    examples = []
    # pairs = generate_example(groups[242],basic_groups[242])
    for group,basic_group in zip(groups,basic_groups):
        # if len(basic_group) < 15:
        if True:
            examples.extend(generate_example(group,basic_group))
        pass

    edus0 = []
    edus1 = []
    edus2 = []
    acts = []

    for pair in examples:
        pair[0] = pair[0].strip().replace('<P>',' P_E ')
        pair[1] = pair[1].strip().replace('<P>',' P_E ')
        pair[2] = pair[2].strip().replace('<P>',' P_E ')
        wrds0 = nltk.word_tokenize(pair[0].strip().lower())
        wrds1 = nltk.word_tokenize(pair[1].strip().lower())
        wrds2 = nltk.word_tokenize(pair[2].strip().lower())
        # print wrds0
        # print wrds1
        # print wrds2
        # print pair[3]
        if True:
            edus0.append(wrds0)
            edus1.append(wrds1)
            edus2.append(wrds2)
            acts.append(pair[3])

        else:
            continue
    
    return edus0 , edus1 , edus2, acts


def build_tree_data(dir_path):
    
    files = os.listdir(dir_path);
    edus_path = [];
    for filename in files:
        if '.dis' in filename:
            # print filename;
            edus_path.append(filename);

    trees = [];
    for edu_path in edus_path:
        trees.append(open(dir_path+'/'+edu_path).readlines());

    print 'trees number : ' , len(trees)
    # print trees[1]
    groups = []
    basic_groups = []
    for tree in trees:
        ftrees = extract_edus(tree)
        if ftrees[0][1] == '' and ftrees[0][2] == '':
            pass
        else:
            groups.append(ftrees)
            basic_groups.append(extract_basic_edus(tree))

    # print 'number of pairs', len(groups) , ''
    # print 'group ' , groups[242]
    # print 'bgroup ' , basic_groups[242]

    tree_examples = []
    # pairs = generate_example(groups[242],basic_groups[242])
    for group,basic_group in zip(groups,basic_groups):
        # if len(basic_group) < 15:
        if True:
            tree_examples.append(generate_example(group,basic_group))
        pass

    edus0 = []
    edus1 = []
    edus2 = []
    acts = []

    for pair in examples:
        pair[0] = pair[0].strip().replace('<P>',' P_E ')
        pair[1] = pair[1].strip().replace('<P>',' P_E ')
        pair[2] = pair[2].strip().replace('<P>',' P_E ')
        wrds0 = nltk.word_tokenize(pair[0].strip().lower())
        wrds1 = nltk.word_tokenize(pair[1].strip().lower())
        wrds2 = nltk.word_tokenize(pair[2].strip().lower())
        # print wrds0
        # print wrds1
        # print wrds2
        # print pair[3]
        if True:
            edus0.append(wrds0)
            edus1.append(wrds1)
            edus2.append(wrds2)
            acts.append(pair[3])

        else:
            continue
    
    return edus0 , edus1 , edus2, acts


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
        lin_output = T.dot(x, self.W) + self.b
        output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        return output


class bid_GRU:
    """
    A implemention of Siamese Recurrent Architectures for Learning Sentence Similarity
    http://www.aaai.org/Conferences/AAAI/2016/Papers/15Mueller12195.pdf

    """
    def  __init__(self,word_dim,hidden_dim=128,bptt_truncate=-1):

        """
        Train 2 spearate GRU network to represent each sentence in pair as a fixed-length vector
        then calculate the 2 sentence vector Manhanttan distance 
        """
        # assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # initialize the network parameters

        U = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,hidden_dim))
        b = np.zeros((6,hidden_dim))
        

        # Created shared variable
        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))


        self.params = [self.U,self.W,self.b]
        

        # We store the Theano graph here
        # self.theano = {}
        # self.__theano_build__()

    def recurrent(self,x_s,E):
        U, W, b = self.U, self.W, self.b


        def forward_direction_step(x_t,s_t_prev):
            # Word embedding layer
            x_e = E[:,x_t]
            # GRU layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e)+W[0].dot(s_t_prev)) + b[0]
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e)+W[1].dot(s_t_prev)) + b[1]
            c_t = T.tanh(U[2].dot(x_e)+W[2].dot(s_t_prev*r_t)+b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            # directly return the hidden state as intermidate output 
            return [s_t]
        
        def backward_direction_step(x_t,s_t_prev):
            # Word embedding layer
            x_e = E[:,x_t]
            # GRU layer 2
            z_t = T.nnet.hard_sigmoid(U[3].dot(x_e)+W[3].dot(s_t_prev)) + b[3]
            r_t = T.nnet.hard_sigmoid(U[4].dot(x_e)+W[4].dot(s_t_prev)) + b[4]
            c_t = T.tanh(U[5].dot(x_e)+W[5].dot(s_t_prev*r_t)+b[5])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            # directly return the hidden state as intermidate output 
            return [s_t]


        # sentence a vector (states)
        s_f , updates = theano.scan(
                forward_direction_step,
                sequences=[x_s],
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))
            
        # sentence b vector (states)
        s_b , updates = theano.scan(
                backward_direction_step,
                sequences=[x_s[::-1]],
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))

        # semantic similarity 
        # s_sim = manhattan_distance(a_s[-1],b_s[-1])

        h_s = T.concatenate([s_f,s_b[::-1]],axis=1)

        return h_s


class soft_attention_layer:

    def __init__(self,hidden_dim):

        # assign instance variables
        self.hidden_dim = hidden_dim

        W_att = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim))
        b_att = np.zeros(1)

        # 
        self.W_att = theano.shared(name='W_att',value=W_att.astype(theano.config.floatX))
        self.b_att = theano.shared(name='b_att',value=W_att.astype(theano.config.floatX))

        # collect parameter
        self.params = [self.W_att,self.b_att]

    def soft_att(self,x_s):
        W_att , b_att = self.W_att , self.b_att

        def score_attention(h_i):
            return T.tanh(W_att.dot(h_i) + b_att)

        def weight_attention(h_i,a_j):
            return h_i*a_j

        h_att , updates = theano.scan(
                score_attention,
                sequences=[x_s]
                )

        h_att = T.exp(h_att)
        h_att = h_att.flatten()
        h_att = h_att / h_att.sum()

        h_s_att , updates =theano.scan(
                weight_attention,
                sequences=[x_s,h_att]
                )

        a_s = h_s_att.sum(axis=0)

        return a_s



class framework:

    """

    """
    def __init__(self,word_dim,label_dim,vocab_size,hidden_dim=128,word_embedding=None,bptt_truncate=-1):

        # the framework only holds the global word embedding
        if word_embedding is None:
            E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,vocab_size))
        else:
            E = word_embedding

        self.E = theano.shared(name='E',value=E.astype(theano.config.floatX))

        gru_layer = bid_GRU(word_dim,hidden_dim,bptt_truncate=-1)

        e_0  = T.lvector('e_0')
        e_1  = T.lvector('e_1')
        e_2  = T.lvector('e_2')

        y = T.lvector('y')

        # 3 symbolic vector
        v_e_0 = gru_layer.recurrent(e_0,self.E)
        v_e_1 = gru_layer.recurrent(e_1,self.E)
        v_e_2 = gru_layer.recurrent(e_2,self.E)

        # soft attention

        sa = soft_attention_layer(hidden_dim*2)

        s_v_e_0 = sa.soft_att(v_e_0)
        s_v_e_1 = sa.soft_att(v_e_1)
        s_v_e_2 = sa.soft_att(v_e_2)
        
        # disable 
        # v_e_0 = (v_e_0).mean(axis=0)
        # v_e_1 = (v_e_1).mean(axis=0)
        # v_e_2 = (v_e_2).mean(axis=0)

        edu_fea = T.concatenate([s_v_e_0,s_v_e_1,s_v_e_2],axis=0)

        # build hidden_layer for edu pair
        mlp_layer_1 = HiddenLayer(hidden_dim*6,label_dim)
        ep_fea_2 = mlp_layer_1.forward(edu_fea)

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

        self.params = []
        self.params += [self.E]
        self.params += gru_layer.params
        self.params += sa.params
        self.params += mlp_layer_1.params

        print 'please verify the parameter of model'
        print self.params
        print len(self.params)

        # update
        updates = sgd_updates_adadelta(norm=0,params=self.params,cost=cost)

        # framework assign function
        self.predict = theano.function([e_0,e_1,e_2],prediction)
        self.predict_class = theano.function([e_0,e_1,e_2],prediction)
        self.ce_error = theano.function([e_0,e_1,e_2,y],cost)

        self.sgd_step = theano.function(
                [e_0,e_1,e_2,y],
                [],
                updates=updates
                )

        pass


def train_with_sgd(model,X_1_train,X_2_train,X_3_train,y_train,learning_rate=0.001,nepoch=20,decay=0.9,index_to_word=[]):
    num_examples_seen = 0
    print 'now learning_rate : ' , learning_rate;
    for epoch in range(nepoch):
        # For each training example ...
        tocount = 0
        tccount = 0
        tycount = 0
        for i in range(len(y_train)):
            # One SGT step
            model.sgd_step(X_1_train[i],X_2_train[i],X_3_train[i],y_train[i])
            num_examples_seen += 1
            # Optionally do callback
            print '>>>>>'
            lwrds = [index_to_word[j] for j in X_1_train[i]]
            rwrds = [index_to_word[j] for j in X_2_train[i]]
            cwrds = [index_to_word[j] for j in X_3_train[i]]
            print 'i-th :' , i;
            print 'the stack_1 edu : ' , " ".join(lwrds)
            print 'the stack_2 edu : ' , " ".join(rwrds)
            print 'the top edu : ' , " ".join(cwrds)
            print 'predict : ' , model.predict(X_1_train[i],X_2_train[i],X_3_train[i])
            print 'ce_error : ' , model.ce_error(X_1_train[i],X_2_train[i],X_3_train[i],y_train[i])
            output = model.predict_class(X_1_train[i],X_2_train[i],X_3_train[i])
            print 'predict_class : ' , output
            print 'true label : ' , y_train[i]
            if y_train[i][0] == 0:
                print 'shift'
            else:
                print 'reduce'

            print 'the number of example have seen for now : ' , num_examples_seen
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


def test_score(model,X_1_test,X_2_test,X_3_test,y_test,index_to_word):
    print 'now score the test dataset'
    scores = [];
    tocount = 0
    tccount = 0
    tycount = 0

    # test on tree
    for i in range(len(y_test)):
        output = model.predict_class(X_1_test[i],X_2_test[i],X_3_test[i])
        ocount = 0
        ccount = 0
        ycount = 0

        lwrds = [index_to_word[j] for j in X_1_test[i]]
        rwrds = [index_to_word[j] for j in X_2_test[i]]
        cwrds = [index_to_word[j] for j in X_3_test[i]]

        print 'i-th : ' , i;
        print 'the stack_1 edu : , ' , " ".join(lwrds)
        print 'the stack_2 edu : ' , " ".join(rwrds)
        print 'the top edu : ' , " ".join(cwrds)
        print 'ce_error : ' , model.ce_error(X_1_test[i],X_2_test[i],X_3_test[i],y_test[i])

        # print 
        print 'predict_class : ' , output
        print 'true label : ' , y_test[i]


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


def tree_score():

    # using modeing predict , perform shift-reduce on each basic tree

    # score on the gold tree


    pass

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

def act_to_index(act):
    if act == 'shift':
        return 0
    elif act == 'reduce':
        return 1
    else:
        raise ValueError('unknown action type')
        pass


def load_data():
    
    #
    # * -- step1 -- build training data
    #

    # build training data set and test data set 
    # for shift reduce triple items
    # shift reduce 
    edus0, edus1, edus2 , acts = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING');
    tst_edus0, tst_edus1, tst_edus2, tst_acts = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TEST')

    print 'loading finish'
    #
    # * -- step2-- train a binary for structure classification
    # 

    print 'load in ' , len(acts) , 'training sample'
    print 'load in ' , len(tst_acts) , 'test sample'
    
    # if True:
    #     return 0
    
    token_list = []
    for edu0, edu1, edu2 in zip(edus0,edus1,edus2):
        token_list.extend(edu0)
        token_list.extend(edu1)
        token_list.extend(edu2)

    word_freq = nltk.FreqDist(token_list)
    print 'Found %d unique words tokens . ' % len(word_freq.items())

    vocabulary_size = 10*1000

    unknown_token = 'UNK'
    sr_mark = 's_r'

    vocab = word_freq.most_common(vocabulary_size-1)
    print 'vocab : '
    index_to_word = []
    index_to_word.append(unknown_token)
    index_to_word.append(sr_mark)
    index_to_word.extend([x[0] for x in vocab])
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print 'Using vocabulary size %d. ' % vocabulary_size
    print "the least frequent word in our vocabulary is '%s' and appeared %d times " % (vocab[-1][0],vocab[-1][1])

    # training dataset
    for i,(edu0,edu1,edu2) in enumerate(zip(edus0,edus1,edus2)):
        edus0[i] = [w if w in word_to_index else unknown_token for w in edu0]
        edus1[i] = [w if w in word_to_index else unknown_token for w in edu1]
        edus2[i] = [w if w in word_to_index else unknown_token for w in edu2]

    # test dataset
    for i,(edu0,edu1,edu2) in enumerate(zip(tst_edus0,tst_edus1,tst_edus2)):
        tst_edus0[i] = [w if w in word_to_index else unknown_token for w in edu0]
        tst_edus1[i] = [w if w in word_to_index else unknown_token for w in edu1]
        tst_edus2[i] = [w if w in word_to_index else unknown_token for w in edu2]

    # X_1_train , X_2_train , X_3_train , y_train
    X_1_train = np.asarray([[word_to_index[w] for w in sent ] for sent in edus0])
    X_2_train = np.asarray([[word_to_index[w] for w in sent ] for sent in edus1])
    X_3_train = np.asarray([[word_to_index[w] for w in sent ] for sent in edus2])

    # action to index
    y_train = []
    for act in acts:
        y_train.append([act_to_index(act)])

    # X_1_test, X_2_test , X_3_test, y_train
    X_1_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_edus0])
    X_2_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_edus1])
    X_3_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tst_edus2])

    # action to index
    y_test = []
    for act in tst_acts:
        y_test.append([act_to_index(act)])


    print " Example sentence '%s' " % " ".join(edus0[0])
    print " Example sentence '%s' " % " ".join(edus1[0])
    print " Example sentence '%s' " % " ".join(edus2[0])
    print " Example sentence after Pre-processing : '%s' " % X_1_train[0]
    print " Example sentence after Pre-processing : '%s' " % X_2_train[0]
    print " Example sentence after Pre-processing : '%s' " % X_3_train[0]
    print " Example label : ", y_train[0]
    print ""



def structure():

    # build Embedding matrix
    label_size = 2
    wvdic = load_word_embedding('../data/glove.6B.300d.txt')
    word_dim = wvdic.values()[0].shape[0]
    hidden_dim = 300

    E = build_we_matrix(wvdic,index_to_word,word_to_index,word_dim)

    model = framework(word_dim,label_size,vocabulary_size,hidden_dim=hidden_dim,word_embedding=E,bptt_truncate=-1)

    # Print SGD step time
    t1 = time.time()
    print model.predict(X_1_train[0],X_2_train[0],X_3_train[0])
    output = model.predict_class(X_1_train[0],X_2_train[0],X_3_train[0])
    print 'predict_class : ' , output
    print 'ce_error : ' , model.ce_error(X_1_train[0],X_2_train[0],X_3_train[0],y_train[0])
    learning_rate = 0.000005

    model.sgd_step(X_1_train[0],X_2_train[0],X_3_train[0],y_train[0])
    t2 = time.time()

    print "SGD Step time : %f milliseconds " % ((t2-t1)*1000.)
    sys.stdout.flush()

    # 
    NEPOCH = 100

    for epoch in range(NEPOCH):

        print 'this is epoch : ' , epoch
        train_with_sgd(model,X_1_train,X_2_train,X_3_train,y_train,learning_rate=learning_rate,nepoch=1,decay=0.9,index_to_word=index_to_word)

        test_score(model,X_1_test,X_2_test,X_3_test,y_test,index_to_word=index_to_word)
        tree_score()



if __name__ == '__main__':
    structure();
