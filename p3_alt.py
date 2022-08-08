from pdb import set_trace
from collections import defaultdict
import sys

sys.setrecursionlimit(1000000)

class Var():
    def __init__(self, val, parents=[]):
        self.val = val
        self.derivatives = defaultdict(lambda: 0)
        self.parents = parents

        if len(self.parents) == 0:       
            self.derivatives[self] = 1

    def __add__(self, x):
        y = Var(self.val + x.val, parents=[self, x])
        
        self.update_gradients(1, x, 1, y)

        return y

    def __sub__(self, x):
        y = Var(self.val - x.val, parents=[self, x])

        self.update_gradients(1, x, -1, y)

        return y

    def __mul__(self, x):
        y = Var(self.val * x.val, parents=[self, x])
        
        self.update_gradients(x.val, x, self.val, y)

        return y

    def __pow__(self, x):
        y = Var(pow(self.val, x), parents=[self])
        
        self.update_gradients(x * self.val, None, None, y)

        return y

    def update_gradients(self, grad, op, op_grad, output):
        for val in self.derivatives:
            output.derivatives[val] += self.derivatives[val] * grad
        if not op: return
        for val in op.derivatives:
            output.derivatives[val] += op.derivatives[val] * op_grad



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

from create_exp import create_exp
from time import time

exp, var_exp = create_exp()

#set_trace()

exec(var_exp)

t1 = time()

exec(exp)

#aa = L4.backwards()

print(time() - t1)

set_trace()

exit()


from time import time

t1 = time()

L4 = (W * X - Y * X * X) ** 2 + (X * X) ** 2 + W + W * X * Y + W ** 2

#L4 = (W * X - Y) ** 2 + W ** 2

print(time() - t1)

#aa = L4.backwards()



set_trace()
