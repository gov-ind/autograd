from pdb import set_trace
from collections import defaultdict
import numpy as np
from create_exp import create_exp
from time import time
import sys
import torch

torch.set_printoptions(precision=100)
sys.setrecursionlimit(1000000)

#a1731.val * -1 *  ( a5783.val * 1 )  * a5199.val * a6858.val * a9759.val *  ( a850.val + a2023.val - a7766.val - a8396.val * a1329.val - a9230.val + a7775.val * a2961.val * a9607.val * a5852.val * a1778.val )  * a261.val * a5615.val * a7424.val * a1114.val * a3677.val

#x3 = torch.tensor([17760000.], requires_grad=True)
#a1 = torch.tensor([2.], requires_grad=True)
#L = -a1 - x3 * a1

class Var():
    def __init__(self, val, leaf=True):
        self.val = val
        self.leaf = leaf

        if self.leaf:
            self.grads = np.array([1])
            #self.grad_idxs = np.array([self])
            self.grad_idxs = self
        else:
            self.grad_idxs = []

    def __add__(self, x):
        y = Var(self.val + x.val, leaf=False)
        
        self.update_gradients(1, x, 1, y)

        return y

    def __sub__(self, x):
        y = Var(self.val - x.val, leaf=False)

        self.update_gradients(1, x, -1, y)

        return y

    def __mul__(self, x):
        y = Var(self.val * x.val, leaf=False)
        
        self.update_gradients(x.val, x, self.val, y)

        return y

    def __pow__(self, x):
        y = Var(pow(self.val, x), leaf=False)
        
        self.update_gradients(x * self.val ** (x - 1), None, None, y)

        return y

    def traverse(self, node, leafs=[]):
        #set_trace() 
        if type(node) == list:
            self.traverse(node[0], leafs)
            if len(node) > 1:
                self.traverse(node[1], leafs)
        else:
            leafs.append(node)
            #yield node

    def traverse(self, root, leafs=[]):
        nodes = [root]

        while len(nodes) != 0:
            node = nodes[-1]
            nodes = nodes[:-1]
        
            if type(node) == list:
                nodes.append(node[0])
                if len(node) > 1:
                    nodes.append(node[1])
            else:
                leafs.append(node)

        leafs.reverse()

    def collect_grads(self):
        #set_trace()
        grads = defaultdict(lambda: 0)
        leafs = []
        self.traverse(self.grad_idxs, leafs)
        for i, var in enumerate(leafs):
            grads[var] += self.grads[i]
        return grads

    def update_gradients(self, grad, op, op_grad, output):
        if op:
            if grad == 0:
                if op_grad == 1:
                    output.grads = op.grads
                    output.grad_idxs = op.grad_idxs
                    return
            if grad == 1:
                if op_grad == 0:
                    output.grads = self.grads
                    output.grad_idxs = self.grad_idxs
                    return
                if op_grad == 1:
                    #output.grads = self.grads
                    output.grads = np.concatenate((self.grads,
                                                   op.grads))
                else:
                    #output.grads = self.grads
                    output.grads = np.concatenate((self.grads,
                                                   op.grads * op_grad))
            else:
                #output.grads = self.grads
                output.grads = np.concatenate((
                    self.grads * grad,
                    op.grads * op_grad
                ))
            #set_trace()
            output.grad_idxs.append(self.grad_idxs)
            output.grad_idxs.append(op.grad_idxs)          
            #output.grad_idxs = np.concatenate((self.grad_idxs, op.grad_idxs))
        else:
            if grad == 1:
                output.grads = self.grads
            else:
                output.grads = self.grads * grad
            output.grad_idxs = self.grad_idxs

times = []
torch_times = []
trials = 100

for _ in range(trials):
    exp, torch_exp, var_exp, var_torch = create_exp()


    start = time()

    exec(var_exp)
    exec(exp)

    grads = L.collect_grads()

    elapsed = time() - start
    print(elapsed)
    times.append(elapsed)

    start = time()

    exec(var_torch)
    exec(torch_exp)

    L_t.backward()

    elapsed = time() - start
    print(f'Torch: {elapsed}')
    torch_times.append(elapsed)

    vars = [a.split(' =')[0] for a in var_exp.split(';')[:-1]]
    i = len(vars)

    for var in vars:
        #print(i)
        grad_torch = eval(var + '_t').grad[0]
        grad = grads[eval(var)]

        #if float(grad_torch) != grad: set_trace()
        i -= 1

print(f'Av time: {sum(times) / trials}')
print(f'Av Torch time: {sum(torch_times) / trials}')
