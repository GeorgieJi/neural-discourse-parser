import sys;
import os;

RST_training = sys.argv[1];
RST_test = sys.argv[2];

traininglst = os.listdir(RST_training);
testlst = os.listdir(RST_test);

for disname in traininglst:
    if 'dis' in disname:
        print disname;
        doc = open(RST_training+'/'+disname).readlines();
        print '-----------------------------------------';
        for line in doc :
            print line.strip();
