import torch
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
        set_trace()
        y = Var(self.val + x.val, leaf=False)
        
        self.update_gradients(1, x, 1, y)

        return y

    def __sub__(self, x):
        set_trace()
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
        #print(cc)
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
 

set_trace()

a1 = Var(1)
a2 = Var(2)
a3 = Var(1)
a4 = Var(5)
a5 = Var(2)
a6 = Var(3)

#L = a1 + a2 * a3 * (a4 - a5 + a6)

#exit()

exp, torch_exp, var_exp, var_torch = create_exp()

#set_trace()

#exec(var_exp)
exec(var_torch)

#t1 = time()

#exec(exp)
exec(torch_exp)

#aa = L4.backwards()

#print(time() - t1)

#set_trace()

#exit()


W1 = torch.tensor([1.], requires_grad=True)
X1 = torch.tensor([2.], requires_grad=True)
Y1 = torch.tensor([3.], requires_grad=True)

L4 = W * X * Y
L = W1 * X1 * Y1

print(time() - t1)

#aa = L4.backwards()

set_trace()
