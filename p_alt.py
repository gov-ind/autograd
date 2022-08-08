from pdb import set_trace
from collections import defaultdict
import numpy as np
from create_exp import create_exp
from time import time
import sys

sys.setrecursionlimit(1000000)

gradients = []
gradient_map = {}
cc = 0

class Var():
    def __init__(self, val, leaf=True):
        self.val = val
        self.leaf = leaf

    def __add__(self, x):
        y = Var(self.val + x.val, leaf=False)
        
        self.update_gradients(1, x, 1, y)

        return y

    def __sub__(self, x):
        y = Var(self.val - x.val, leaf=False)

        self.update_gradients(1, x, -1, y)

        return y

    def __mul__(self, x):
        set_trace()
        y = Var(self.val * x.val, leaf=False)
        
        self.update_gradients(x.val, x, self.val, y)

        return y

    def __pow__(self, x):
        y = Var(pow(self.val, x), leaf=False)
        
        self.update_gradients(x * self.val, None, None, y)

        return y

    def update_gradients(self, grad, op, op_grad, output):
        #set_trace()
        global cc, gradients, gradient_map
        print(cc)
        #set_trace()
        if self.leaf:
            if self in gradient_map:
                gradients[gradient_map[self]] += grad
            else:
                gradient_map[self] = cc
                gradients = np.append(gradients, grad)
                cc += 1
        else:
            gradients *= grad
        if not op: return
        if op.leaf:
            if op in gradient_map:
                gradients[gradient_map[op]] += op_grad
            else:
                gradient_map[op] = cc
                gradients = np.append(gradients, op_grad)
                cc += 1
        else:
            gradients *= op_grad
 


#z = (w * x - y) ** 2 + w ** 2
w, x, y = 1, 2, 3

set_trace()

W = Var(1)
X = Var(2)
Y = Var(3)

''''L0 = W * X
L1 = L0 - Y
L2 = L1 ** 2
L3 = W ** 2
L4 = L2 + L3

set_trace()

dL0dW = L0.parents[0].derivatives[L0]
dL1dL0 = L1.parents[0].derivatives[L1]
dL2dL1 = L2.parents[0].derivatives[L2]
dL4dL2 = L4.parents[0].derivatives[L4]

dL4dL2 = L4.parents[1].derivatives[L4]
dL2dW = L3.parents[0].derivatives[L3]

dL4dW = dL0dW * dL1dL0 * dL2dL1 * dL4dL2 + dL2dW  * dL4dL2
'''

#set_trace()
#exp, var_exp = create_exp()

#set_trace()

'''exec(var_exp)

t1 = time()

exec(exp)

#aa = L4.backwards()

print(time() - t1)

set_trace()

exit()
'''

from time import time

t1 = time()

#L4 = (W * X - Y) ** 2 + W ** 2

import torch

W1 = torch.tensor([1.], requires_grad=True)
X1 = torch.tensor([2.], requires_grad=True)
Y1 = torch.tensor([3.], requires_grad=True)

L4 = W * X * Y
L = W1 * X1 * Y1

print(time() - t1)

#aa = L4.backwards()

set_trace()
