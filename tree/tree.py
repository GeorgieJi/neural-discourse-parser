import sys


def parse_node(node):
    
    # main token 
    # ( Root )
    # ( Nucleus )
    # ( Staellite )

    # sub token
    # (span 1 5)
    # (leaf 1)
    # (rel2par Comparison)
    # (text _! _!)

    # for main token as basic unit
    # ( Nucleus (span 1 3) (rel2par Evaluation) unit1 unit2 )
    # ( Nucleus (span 1 3) (rel2par Evaluation) unit1 unit2 )



    pass

def parse_tree(tree):

    #
    # target the result = [ ['NS','e1','e2'] , ['NN','e3','e4'] , ['NS','e12','e34'] ]
    #

    line = " ".join(tree).replace("\n","")
    # print line
    items = line.split()
    # print items
    
    Nodes = []
    stack = []
    node = []
    while len(items) != 0:

        t = items.pop(0)
        stack.append(t)
        # print 'stack :' , stack;
        if t == ')':
            while True:
                it = stack.pop()
                node.append(it)
                if it == '(':
                    Nodes.append(node)
                    node.reverse()
                    print 'Node : ' , node
                    node = []
                    break
                else:
                    pass
        else:
            pass
                
    pass;

def process():
    tree = open('wsj_0699.out.dis').readlines()
    parse_tree(tree)

if __name__ == '__main__':
    process()
