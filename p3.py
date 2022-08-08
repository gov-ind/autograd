from pdb import set_trace
from collections import defaultdict

class Var():
    def __init__(self, val, name='', parents=[]):
        self.name = name
        self.val = val
        self.derivatives = {}
        self.parents = parents

    def __add__(self, x):
        #set_trace()
        y = Var(self.val + x.val, parents=[self, x])
        self.derivatives[y] = 1
        x.derivatives[y] = 1
        return y

    def __sub__(self, x):
        #set_trace()
        y = Var(self.val - x.val, parents=[self, x])
        self.derivatives[y] = 1
        x.derivatives[y] = -1
        return y

    def __mul__(self, x):
        #set_trace()
        y = Var(self.val * x.val, parents=[self, x])
        self.derivatives[y] = x.val
        x.derivatives[y] = self.val
        return y

    def __pow__(self, x):
        #set_trace()
        y = Var(pow(self.val, x), parents=[self])
        self.derivatives[y] = x * self.val
        return y

    def backward(self, child=None):
        #set_trace()
        if len(self.parents) == 0:
            return { self.name: self.derivatives[child] }

        ders = defaultdict(lambda: 0)
        for parent in self.parents:
            _ders = parent.backward(self)
            for name, der in _ders.items():
                ders[name] += (self.derivatives[child] if child else 1) * der

        return ders


#z = (w * x - y) ** 2 + w ** 2
w, x, y = 1, 2, 3

set_trace()

W = Var(1, 'w')
X = Var(2, 'x')
Y = Var(3, 'y')

L0 = W * X
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

aa = L4.backward()

set_trace()
