import random
import pickle
import sys
import os

f = open(sys.argv[1], 'rb')
d = pickle.load(f)
random.shuffle(d)
random.shuffle(d)

d_train = d[:int(len(d) * 0.8)]
d_test = d[int(len(d) * 0.8):]

pickle.dump(d_train, open(sys.argv[1][:-4] + '_train.pkl', 'wb'))
pickle.dump(d_test, open(sys.argv[1][:-4] + '_test.pkl', 'wb')) 

