'''
A Fast and Accurate Dependency Parser using Neural Networks
'''


import sys
import os
import nltk

def extract_edu(dir_path):
    #
    # list of edus files name 
    #
    
    #print dir_path;
    files = os.listdir(dir_path);
    edus_path = [];
    for filename in files:
        if '.edus' in filename:
            # print filename;
            edus_path.append(filename);

    edus = [];
    for edu_path in edus_path:
        edus.extend(open(dir_path+'/'+edu_path).readlines());

    for edu in edus:
        print edu;

    return edus;

def process_edu(edu_list):
    ap_edus = [];
    for edu in edu_list:
        tokens = word_tokenize(edu);

#
#
#

def load_wordembedding():
    pass;

def build_dataset(trnedus):

    pass;




def parser():
    #extract_edu()
    trnedus = extract_edu('../data/RSTmain/RSTtrees-WSJ-main-1.0/TRAINING');
    
    # trnset word vector / trnlabel , EB, EC, EE
    trnset = build_dataset(trnedus);


if __name__ == '__main__':
    parser();
