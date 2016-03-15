import sys


def parse_node(node):
    
    # 3 type of node: leaf span root
    # ( Nucleus (leaf 1) (rel2par Comparsion) (text _!The European Community's

    f_node = []
    if node[2] == '(leaf':
        f_node.append(node[1])
        f_node.append(node[5][:-1])
        f_node.append(" ".join(node[7:-1])[2:-3])
        return f_node

    elif node[2] == '(span' and node[1] != 'Root':
        f_node.append(node[1])
        f_node.append(node[6][:-1])
        f_node.append(node[7])
        f_node.append(node[8])
        return f_node
    elif node[2] == '(span' and node[1] == 'Root':
        f_node.append(node[1])
        f_node.append('Root')
        f_node.append(node[5])
        f_node.append(node[6])
        return f_node
    else:
        raise ValueError('node format error, check the tree data')


def extract_pair(tree):
    """
    take the tree as input
    extract relation and structure pair
    """
    pass

def parse_tree(tree):

    """
    target the result = [ ['NS','e1','e2'] , ['NN','e3','e4'] , ['NS','e12','e34'] ]
    """

    line = " ".join(tree).replace("\n","")
    items = line.split()

    
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
                    node = parse_node(node)
                    stack.append(node)
                    node = []
                    break
                else:
                    pass
        else:
            pass

    tree = stack.pop()
    print 'this is the tree'
    print tree

    # Nodes is heiracaical structure tree

    pass;

def process():
    tree = open('wsj_0699.out.dis').readlines()
    parse_tree(tree)

if __name__ == '__main__':
    process()
