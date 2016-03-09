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

def extract_edu(dir_path):
    files = os.listdir(dir_path);
    edus_path = [];
    for filename in files:
        if '.edus' in filename:
            # print filename;
            edus_path.append(filename);
    edus = [];
    for edu_path in edus_path:
        edus.extend(open(dir_path+'/'+edu_path).readlines());

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

        # Begin 1
        # End 2
        # Continious 3
        labels = [];
        for j,tok in enumerate(toks):
            if j == 0 :
                labels.append(1)
                # 
            elif j == len(toks)-1:
                labels.append(2)
            else :
                labels.append(3)

        edus_toks_label.append(labels)

    return edus_toks_label


class GRU:

    def  __init__(self,word_dim,hidden_dim=128,bptt_truncate=-1):

        # assign instance variables

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(3,hidden_dim,hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(3,hidden_dim,hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
        b = np.zeros((3,hidden_dim))
        c = np.zeros(word_dim)

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

        def forward_prop_step(x_t,s_t_prev):
            #
            #
            # Word embedding layer
            x_e = E[:,x_t]

            # GRU layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e)+W[0].dot(s_t_prev)) + b[0]
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e)+W[1].dot(s_t_prev)) + b[1]
            c_t = T.tanh(U[2].dot(x_e)+W[2].dot(s_t_prev*r_t)+b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t*s_t_prev

            # output 
            o_t = T.nnet.softmax(V.dot(s_t) + c)[0]

            return [o_t,s_t]

        [o,s] , updates = theano.scan(
                forward_prop_step,
                sequences=x,
                truncate_gradient=self.bptt_truncate,
                outputs_info=[
                    None,
                    dict(initial=T.zeros(self.hidden_dim))
                    ])
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
                updates=updates
                )

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



def parser():
    
    # extract_edu()
    # collecting edus from corpus
    trnedus = extract_edu('data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING');
    
    # trnset word vector / trnlabel , EB, EC, EE
    trnset = build_dataset(trnedus);
    trnset_label = build_datasetlabel(trnedus);

    print 'the size of Edus : ' , len(trnset)
    print 'the size of label : ' , len(trnset_label)

    token_list = [];
    for sen in trnset:
        token_list.extend(sen)

    word_freq = nltk.FreqDist(token_list);
    print 'Found %d unique words tokens.' % len(word_freq.items())

    vocabulary_size = 4000
    unknown_token = 'UNK'

    vocab = word_freq.most_common(vocabulary_size-1);
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print 'Using vocabulary size %d.' % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times " % (vocab[-1][0],vocab[-1][1])

    for i,sent in enumerate(trnset):
        trnset[i] = [w if w in word_to_index else unknown_token for w in sent]

    print '***********************'
    for edu,label in zip(trnset[:10],trnset_label[:10]):
        print '>>>>>>'
        print edu , label,

    
    # X_train , y_train
    # X_train = ?
    # y_train = ?

    print "\n Example sentence '%s' " % trnedus[0]
    print "\n Example sentence after Pre-processing : '%s'" % trnset[0]
    print "\n Example label after labeling : '%s' " % trnset_label[0]

    # build model GRU
    model = GRU(vocabulary_size,hidden_dim=128,bptt_truncate=-1)

    # Print SGD step time
    t1 = time.time()
    model.sgd_step(x_train[10],y_train[10],learning_rate)
    t2 = time.time()

    print "SGD Step time : %f milliseconds" % ((t2-t1)*1000.)
    sys.stdout.flush()

    # 




if __name__ == '__main__':
    parser();
