from pdb import set_trace
from collections import defaultdict
import sys
sys.setrecursionlimit(100000000)
from create_exp import create_exp
from time import time

class Var():
    def __init__(self, val, parents=[]):
        self.val = val
        self.derivatives = {}
        self.parents = parents

    def __add__(self, x):
        y = Var(self.val + x.val, parents=[self, x])
        self.derivatives[y] = 1
        x.derivatives[y] = 1
        return y

    def __sub__(self, x):
        y = Var(self.val - x.val, parents=[self, x])
        self.derivatives[y] = 1
        x.derivatives[y] = -1
        return y

    def __mul__(self, x):
        y = Var(self.val * x.val, parents=[self, x])
        self.derivatives[y] = x.val
        x.derivatives[y] = self.val
        return y

    def __pow__(self, x):
        y = Var(pow(self.val, x), parents=[self])
        self.derivatives[y] = x * self.val
        return y

    def backwards(self, child=None):
        if len(self.parents) == 0:
            return { self: self.derivatives[child] }

        derivatives = defaultdict(lambda: 0)
        for parent in self.parents:
            for name, derivative in parent.backwards(self).items():
                derivatives[name] += (
                    (self.derivatives[child] if child else 1) * derivative
                )

        return derivatives


#z = (w * x - y) ** 2 + w ** 2
w, x, y = 1, 2, 3

set_trace()

W = Var(1)
X = Var(2)
Y = Var(3)

'''L0 = W * X
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

set_trace()
exp, var_exp = create_exp()

#set_trace()

exec(var_exp)

t1 = time()

exec(exp)

aa = L4.backwards()

print(time() - t1)

set_trace()

exit()

L4 = (W * X - Y) ** 2 + W ** 2
L4 = (W * X - Y * X * X) ** 2 + (X * X) ** 2 + W + W * X * Y + W ** 2

from time import time

t1 = time()

aa = L4.backwards()

print(time() - t1)

set_trace()
