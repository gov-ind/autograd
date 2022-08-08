from pdb import set_trace
from collections import defaultdict
import numpy as np
from create_exp import create_exp
from time import time
import sys
import torch

sys.setrecursionlimit(1000000)

class Var():
    def __init__(self, val, leaf=True):
        self.val = val
        #self.derivatives = defaultdict(lambda: 0)
        self.leaf = leaf

        if self.leaf:
            self.derivatives = np.array([1])
            self.d_idx = np.array([self])

    def __add__(self, x):
        y = Var(self.val + x.val, leaf=False)
        
        self.update_gradients(1, x, 1, y)

        return y

    def __sub__(self, x):
        y = Var(self.val - x.val, leaf=False)

        self.update_gradients(1, x, -1, y)

        return y

    def __mul__(self, x):
        #set_trace()
        y = Var(self.val * x.val, leaf=False)
        
        self.update_gradients(x.val, x, self.val, y)

        return y

    def __pow__(self, x):
        y = Var(pow(self.val, x), leaf=False)
        
        self.update_gradients(x * self.val, None, None, y)

        return y

    #def __eq__(self, x):
        #return self == x

    def get_grad(

    def update_gradients(self, grad, op, op_grad, output):
        #set_trace()

        if op:
            if grad == 1 and op_grad == 1:
                output.derivatives = np.concatenate((self.derivatives, op.derivatives))
                output.d_idx = npself.d_idx + op.d_idx
                return

            #intersect = set(self.d_idx) & set(op.d_idx)
            
            '''if len(intersect) > 0:
                for a in intersect:
                    i, j = self.d_idx.index(a), op.d_idx.index(a)
                    self.derivatives[i] += op.derivatives[j]
                    op.derivatives = np.delete(op.derivatives, j)
                    op.d_idx.remove(a)
'''
            output.derivatives = np.concatenate((self.derivatives * grad, op.derivatives * op_grad))
            output.d_idx = self.d_idx + op.d_idx
            #if 0 in output.derivatives:
                #set_trace()
        else:
            if grad == 1:
                output.derivatives = self.derivatives
            else:
                output.derivatives = self.derivatives * grad
            output.d_idx = self.d_idx
        

        '''for val in self.derivatives:
            output.derivatives[val] += self.derivatives[val] * grad
        if not op: return
        for val in op.derivatives:
            output.derivatives[val] += op.derivatives[val] * op_grad 
        '''


#z = (w * x - y) ** 2 + w ** 2
w, x, y = 1, 2, 3

#set_trace()

a1 = Var(1)
a2 = Var(2)
a3 = Var(1)
a4 = Var(5)
a5 = Var(2)
a6 = Var(3)

#L = a1 + a2 * a3 * (a4 - a5 + a6)

a3 = Var(3)

L = (a1 * a2 - a3) ** 2 + a1 ** 2

#set_trace()

a1_t = torch.tensor([1.], requires_grad=True)
a2_t = torch.tensor([2.], requires_grad=True)
a3_t = torch.tensor([1.], requires_grad=True)
a4_t = torch.tensor([5.], requires_grad=True)
a5_t = torch.tensor([2.], requires_grad=True)
a6_t = torch.tensor([3.], requires_grad=True)

L_t = a1_t + a2_t * a3_t * (a4_t - a5_t + a6_t)

#set_trace()

W = Var(1)
X = Var(2)
Y = Var(3)

''''L0 = W * X
L1 = L0 - Y
L2 = L1 ** 2
L3 = W ** 2
L4 = L2 + L3

#set_trace()

dL0dW = L0.parents[0].derivatives[L0]
dL1dL0 = L1.parents[0].derivatives[L1]
dL2dL1 = L2.parents[0].derivatives[L2]
dL4dL2 = L4.parents[0].derivatives[L4]

dL4dL2 = L4.parents[1].derivatives[L4]
dL2dW = L3.parents[0].derivatives[L3]

dL4dW = dL0dW * dL1dL0 * dL2dL1 * dL4dL2 + dL2dW  * dL4dL2
'''

#set_trace()

for _ in range(10):
    exp, torch_exp, var_exp, var_torch = create_exp()

    t1 = time()

    exec(var_exp)
    exec(exp)

    print(time() - t1)

    t1 = time()

    exec(var_torch)
    exec(torch_exp)

    L_t.backward()

    print(f'Torch: {time() - t1}')

    set_trace()

    #print(len(np.where(L.derivatives == 0)[0]))

#set_trace()

exit()

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
