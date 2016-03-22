import theano
import numpy as np
import theano.tensor as T


o = T.matrix('o')
y = T.ivector('y')
o_error = T.nnet.categorical_crossentropy(o,y)
ce_error = theano.function([o,y],o_error)

p = np.random.random((4,2))
q = np.asarray([0,1,1,0])
print p
print q
result = ce_error(p,q)
print result
