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
from collections import OrderedDict

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
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        # 
        # step is the adadelta key! 
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


def pos_parser(inputfile,outputfile):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    print dirpath
    print 'cwd' , os.getcwd()
    cmd = ('java -cp lib/stanford-tagger/ TaggerDemo '+dirpath+'/lib/stanford-tagger/models/english-left3words-distsim.tagger %s %s')%(inputfile,outputfile)

    # print cmd
    os.system(cmd)
    posSentence = open(outputfile,'r').readlines()
    return posSentence

def preprocess(edus):

    # this is the version of pos segmenter
    with open('tmp/segs','w') as f:
        for p in edus:
            for edu in p:
                f.write(edu.strip()+'\n')
            f.write('\n')
    # 
    cp = '';
    filepath = 'lib/stanford-tagger/DSC_TaggerDemo.java';
    
    dirpath = os.path.dirname(os.path.abspath(__file__))
    # print dirpath
    cmd = 'javac -cp '+dirpath+'/lib/stanford-tagger %s'%(filepath)
    # print cmd
    os.system(cmd)
    eduspos = pos_parser('tmp/segs','tmp/segs.tag')
    return eduspos


def extract_pos_sen(sen):
    items = sen.split()
    sen_wrds = []
    sen_poss = []
    for item in items:
        # find the first / in reverse direction
        index = item.rfind('/')
        # the font part is the word and the left part is the POS
        wrd = item[:index]
        pos = item[index+1:]
        sen_wrds.append(wrd)
        sen_poss.append(pos)
    return " ".join(sen_wrds) , " ".join(sen_poss)



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
    train_p = [];
    train_y = [];

    print len(edus)
    # for edu in edus:
        # print '>>>>'
        # print edu

    print 'now preprocessing ... '
    posedus = preprocess(edus)

    edus = []
    tmp = []
    for l in posedus:
        if l != '\n':
            tmp.append(l)
        else:
            edus.append(tmp)
            tmp = []

    pas = []
    for p in edus:
        tokseq = []
        labels = []
        postag = []
        
        for edu in p:
            wrdsen , possen = extract_pos_sen(edu)
            toks = wrdsen.strip().lower().split()
            label = (len(toks)-1)*[0] + [1]

            tokseq.extend(toks)
            postag.extend(possen.strip().split())
            labels.extend(label)
            pass
        pas.append((tokseq,postag,labels))

    # 

    # pas = simpleedus(edus)
    # print 'this is pas'
    # print pas[0]
    for pa in pas:
        train_x.append(pa[0])
        train_p.append(pa[1])
        train_y.append(pa[2])
    return [train_x,train_p,train_y]



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
    
    def  __init__(self,word_dim,pos_dim,label_dim,vocab_size,pos_size,hidden_dim=128,word_embedding=None,POS_embedding=None,bptt_truncate=-1):

        # assign instance variables

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.label_dim = label_dim


        if word_embedding is None:
            E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim,vocab_size))
        else:
            # using pre-trained word vector
            E = word_embedding

        # the framework only holds the global POS embeddings
        if POS_embedding is None:
            P = np.random.uniform(-np.sqrt(1./pos_dim),np.sqrt(1./pos_dim),(pos_dim,pos_size))
        else:
            P = POS_embedding

        U = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,hidden_dim))
        
        Up = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,pos_dim))
        Wp = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(6,hidden_dim,hidden_dim))
        # combine hidden states from 2 layer 
        V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(label_dim,hidden_dim*4))
        b = np.zeros((6,hidden_dim))
        bp = np.zeros((6,hidden_dim))
        c = np.zeros(label_dim)

        # Created shared variable
        self.E = theano.shared(name='E',value=E.astype(theano.config.floatX))
        self.P = theano.shared(name='P',value=P.astype(theano.config.floatX))

        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W',value=W.astype(theano.config.floatX))
        
        self.Up = theano.shared(name='Up',value=Up.astype(theano.config.floatX))
        self.Wp = theano.shared(name='Wp',value=Wp.astype(theano.config.floatX))
        
        self.V = theano.shared(name='V',value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b',value=b.astype(theano.config.floatX))
        self.bp = theano.shared(name='bp',value=bp.astype(theano.config.floatX))
        self.c = theano.shared(name='c',value=c.astype(theano.config.floatX))


        # params
        self.params = [self.E,self.P,self.U,self.W,self.Up,self.Wp,self.V,self.b,self.bp,self.c]
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, P, V, U, W, Up, Wp, b, bp, c = self.E, self.P, self.V, self.U, self.W, self.Up, self.Wp, self.b , self.bp, self.c

        x = T.ivector('x')
        p = T.ivector('p')
        y = T.ivector('y')

        def forward_direction_prop_step(x_t,s_t_prev):
            x_e = E[:,x_t]
            # update gate
            z_t = T.nnet.sigmoid(U[0].dot(x_e)+W[0].dot(s_t_prev) + b[0])
            # reset gate
            r_t = T.nnet.sigmoid(U[1].dot(x_e)+W[1].dot(s_t_prev) + b[1])
            # inner hidden state
            c_t = T.tanh(U[2].dot(x_e)+W[2].dot(s_t_prev*r_t)+b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            return [s_t]

        def backward_direction_prop_step(x_t,s_t_prev):
            x_e = E[:,x_t]
            # update gate
            z_t = T.nnet.sigmoid(U[3].dot(x_e)+W[3].dot(s_t_prev) + b[3])
            # reset gate
            r_t = T.nnet.sigmoid(U[4].dot(x_e)+W[4].dot(s_t_prev) + b[4])
            # inner hidden state
            c_t = T.tanh(U[5].dot(x_e)+W[5].dot(s_t_prev*r_t)+b[5])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            return [s_t]
        
        def f_pos_step(x_t,s_t_prev):
            x_e = P[:,x_t]
            z_t = T.nnet.sigmoid(Up[0].dot(x_e)+Wp[0].dot(s_t_prev) + bp[0])
            r_t = T.nnet.sigmoid(Up[1].dot(x_e)+Wp[1].dot(s_t_prev) + bp[1])
            c_t = T.tanh(Up[2].dot(x_e)+Wp[2].dot(s_t_prev*r_t)+bp[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            return [s_t]

        def b_pos_step(x_t,s_t_prev):
            x_e = P[:,x_t]
            z_t = T.nnet.sigmoid(Up[3].dot(x_e)+Wp[3].dot(s_t_prev) + bp[3])
            r_t = T.nnet.sigmoid(Up[4].dot(x_e)+Wp[4].dot(s_t_prev) + bp[4])
            c_t = T.tanh(Up[5].dot(x_e)+Wp[5].dot(s_t_prev*r_t)+bp[5])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev
            return [s_t]
       
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
        
        # forward direction states
        f_p , updates = theano.scan(
                f_pos_step,
                sequences=p,
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))
            
        # backward direction states
        b_p , updates = theano.scan(
                b_pos_step,
                sequences=p[::-1], # the reverse direction input
                truncate_gradient=self.bptt_truncate,
                outputs_info=T.zeros(self.hidden_dim))


        self.f_s = f_s
        self.b_s = b_s
        self.f_p = f_p
        self.b_p = b_p

        # combine the forward GRU state and backward GRU state together 
        c_x = T.concatenate([f_s,b_s[::-1],f_p,b_p[::-1]],axis=1)
        
        def o_step(combined_s_t):
            o_t = T.nnet.softmax(V.dot(combined_s_t)+c)[0] 
            eps = np.asarray([1.0e-10]*self.label_dim,dtype=theano.config.floatX)
            o_t = o_t + eps
            return o_t

 
        # concatenate the hidden state from 2 GRU layer to do the output
        o , updates = theano.scan(
                o_step,
                sequences=c_x,
                truncate_gradient=self.bptt_truncate,
                outputs_info=None)

        prediction = T.argmax(o,axis=1)

        #prediction = T.argmax(o,axis=1)
        # print 'o,y' , o ,y 
        o_error = T.sum(T.nnet.categorical_crossentropy(o,y))

        cost = o_error

        # Assign functions
        self.predict = theano.function([x,p],o)
        # self.detect = theano.function([x],[o])
        self.predict_class = theano.function([x,p],prediction)
        self.ce_error = theano.function([x,p,y],cost)
        # self.bptt = theano.function([x,y],[dE,dU,dW,db,dV,dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # updates
        updates = sgd_updates_adadelta(norm=0,params=self.params,cost=cost)

        self.sgd_step = theano.function(
                [x,p,y],
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


def train_with_sgd(model,X_train,P_train,y_train,learning_rate=0.001,nepoch=20,decay=0.9,index_to_word=[]):
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
            model.sgd_step(X_train[i],P_train[i],y_train[i])
            num_examples_seen += 1
            # Optionally do callback
            print '>>>>>'

            wrds = []
            for j in X_train[i]:
                wrds.append(index_to_word[j])

            print 'i-th :' , i;
            print 'X_train[i] : ' , X_train[i]
            print 'the edus : ' ," ".join(wrds)
            print 'ce_error : ' , model.ce_error(X_train[i],P_train[i],y_train[i])
            # print 'predict : ' , model.predict(X_train[i])
            output = model.predict_class(X_train[i],P_train[i])
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
            
            if False:
                print 'for now the whole training data score :'
                print 'epoch : ' , epoch , ' examples seen : ' , num_examples_seen
                precision , recall , Fmeasure = data_score(model,X_train,y_train)
                print 'precision : ' , precision
                print 'recall : ' , recall
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


def test_score(model,X_test,P_test,y_test,index_to_word,epoch):
    
    print 'now score the test dataset'
    scores = [];
    tocount = 0
    tccount = 0
    tycount = 0

    for i in range(len(y_test)):

        output = model.predict_class(X_test[i],P_test[i])

        # np.savez
        ocount = 0
        ccount = 0
        ycount = 0

        print 'test dataset prediction :'
        print output
        print 'test ture label : '
        print y_test[i]

        for j,(o,y) in enumerate(zip(output,y_test[i])):
            if o == 1:
                ocount += 1
            if y == 1:
                ycount += 1
            if y == o and y == 1:
                ccount += 1

            if y == 1 and o != y :
                print 'wrong prediction on uncapture label:'
                print index_to_word[X_test[i][j]]

                l = 0
                r = len(X_test[i])-1
                w = 3
                if j-w < 0 :
                    l = 0
                else:
                    l = j-r
                if j+w > r:
                    pass
                else:
                    r = j+w

                context = " ".join(index_to_word[xj] for xj in X_test[i][l:r])
                print 'context : '
                print context


            if y == 0 and o != y:
                print 'wrong labeling : '
                print index_to_word[X_test[i][j]]
                

                l = 0
                r = len(X_test[i])-1
                w = 3
                if j-w < 0 :
                    l = 0
                else:
                    l = j-r
                if j+w > r:
                    pass
                else:
                    r = j+w

                context = " ".join(index_to_word[xj] for xj in X_test[i][l:r])
                print 'context : '
                print context


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

def data_score(model,X_test,y_test):
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
            if y == o:
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

    return precision , recall , Fmeasure

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
    

    edux , edup, eduy = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING');
    testx, testp, testy = build_data('../data/RSTmain/RSTtrees-WSJ-main-1.0/TEST')

    # build training set
    trnset = copy.deepcopy(edux)
    trnset_pos = copy.deepcopy(edup)
    trnset_label = copy.deepcopy(eduy)

    # build test set
    tstset = copy.deepcopy(testx)
    tstset_pos = copy.deepcopy(testp)
    tstset_label = copy.deepcopy(testy)

    # print testx[-3]

    print 'the size of Edus : ' , (len(trnset))
    print 'the size of label : ' , len(trnset_label)


    # Collect the word sentence information
    token_list = [];
    for sen in trnset:
        # print sen
        token_list.extend(sen)
    word_freq = nltk.FreqDist(token_list);
    print 'Found %d unique words tokens.' % len(word_freq.items())
    vocabulary_size = 10*1000
    unknown_token = 'UNK'
    vocab = word_freq.most_common(vocabulary_size-1);
    index_to_word = [x[0] for x in vocab]
    print 'vocab : '
    # print vocab
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    # Collect the part of speech information
    pos_list = []
    for sen in trnset_pos:
        pos_list.extend(sen)
    pos_freq = nltk.FreqDist(pos_list)
    print 'Found %d unique part of speech tokens. '% len(pos_freq.items())
    pos_num = len(pos_freq.items())
    pos_vocab = pos_freq.most_common(pos_num)
    index_to_pos = [x[0] for x in pos_vocab]
    pos_to_index = dict([(p,i) for i,p in enumerate(index_to_pos)])
    print index_to_pos
    print pos_to_index


    print 'Using vocabulary size %d.' % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times " % (vocab[-1][0],vocab[-1][1])

    # training dataset 
    for i,sent in enumerate(trnset):
        trnset[i] = [w if w in word_to_index else unknown_token for w in sent]
    for i,sent in enumerate(tstset):
        tstset[i] = [w if w in word_to_index else unknown_token for w in sent]
     
    # X_train , y_train
    X_train = np.asarray([[word_to_index[w] for w in sent ] for sent in trnset])
    P_train = np.asarray([[pos_to_index[w] for w in sent ] for sent in trnset_pos])
    y_train = np.asarray(trnset_label)

    # X_test , y_test
    X_test = np.asarray([[word_to_index[w] for w in sent ] for sent in tstset])
    P_test = np.asarray([[pos_to_index[w] for w in sent ] for sent in tstset_pos])
    y_test = np.asarray(tstset_label)

    print "\n Example sentence '%s' " % " ".join(edux[0])
    print "\n Example sentence after Pre-processing : '%s'" % (trnset[0])
    print "\n Example sentence part of speech after Pre-processing : '%s'" % (trnset_pos[0])
    print "\n Example label after labeling : '%s' " % (trnset_label[0])

    # build Embedding matrix
    label_size = 2
    wvdic = load_word_embedding('../data/glove.6B.300d.txt')
    word_dim = wvdic.values()[0].shape[0]
    E = build_we_matrix(wvdic,index_to_word,word_to_index,word_dim)

    # build part-of-speech matrix
    pvdic = load_word_embedding('../data/pos.200d.txt')
    pos_dim = pvdic.values()[0].shape[0]
    pos_size = len(pvdic)
    print pos_dim
    print pos_size
    P = build_we_matrix(pvdic,index_to_pos,pos_to_index,pos_dim)
    print 'the shape of POS matrix : ' , P.shape


    hidden_dim = 150
    print 'word dim : ' , word_dim
    print 'pos dim : ' , pos_dim
    print 'hidden dim : ' , hidden_dim

    # build model GRU and test it for first output
    # def  __init__(self,word_dim,label_dim,vocab_size,hidden_dim=128,word_embedding=None,bptt_truncate=-1):
    model = bidirectional_GRU(word_dim,pos_dim,label_size,vocabulary_size,pos_size,hidden_dim=hidden_dim,word_embedding=E,POS_embedding=P,bptt_truncate=-1)

    # Print SGD step time
    t1 = time.time()
    oo = model.predict(X_train[0],P_train[0])
    print oo
    print oo.shape
    print oo[0]
    print oo[0].shape
    print y_train[0]

    output = model.predict_class(X_train[0],P_train[0])
    print 'predict_class : ' , output
    print 'ce_error : ' , model.ce_error(X_train[0],P_train[0],y_train[0])
    learning_rate = 0.00005

    model.sgd_step(X_train[0],P_train[0],y_train[0])

    t2 = time.time()
    print "SGD Step time : %f milliseconds" % ((t2-t1)*1000.)
    sys.stdout.flush()

    # training simple edu tag

    NEPOCH = 100

    for epoch in range(NEPOCH):
        #
        print 'this is epoch : ' , epoch
        train_with_sgd(model,X_train,P_train,y_train,learning_rate=learning_rate,nepoch=1,decay=0.9,index_to_word=index_to_word)
        test_score(model,X_test,P_test,y_test,index_to_word,epoch)



if __name__ == '__main__':
    parser();
