from pdb import set_trace
from collections import defaultdict

class Var():
    def __init__(self, val, parents=[]):
        self.val = val
        self.derivatives = defaultdict(lambda: 0)
        self.parents = parents

        if len(self.parents) == 0:       
            self.derivatives[self] = 1

    def __add__(self, x):
        set_trace()
        y = Var(self.val + x.val, parents=[self, x])
        
        if False:
        #if len(self.parents) == 0:
            y.derivatives[self] = 1
            y.derivatives[x] = 1

        else:
            for val in self.derivatives:
                y.derivatives[val] += self.derivatives[val] * 1
            for val in x.derivatives:
                y.derivatives[val] += x.derivatives[val] * 1

        return y

    def __sub__(self, x):
        set_trace()
        y = Var(self.val - x.val, parents=[self, x])

        if False:
        #if len(self.parents) == 0:
            y.derivatives[self] = 1
            y.derivatives[x] = -1

        else:
            for val in self.derivatives:
                y.derivatives[val] += self.derivatives[val] * 1
            for val in x.derivatives:
                y.derivatives[val] += x.derivatives[val] * -1

        return y

    def __mul__(self, x):
        set_trace()
        y = Var(self.val * x.val, parents=[self, x])
        
        if False:
        #if len(self.parents) == 0:
            y.derivatives[self] = x.val
            y.derivatives[x] = self.val

        else:
            for val in self.derivatives:
                y.derivatives[val] += self.derivatives[val] * x.val
            for val in x.derivatives:
                y.derivatives[val] += x.derivatives[val] * self.val

        return y

    def __pow__(self, x):
        set_trace()
        y = Var(pow(self.val, x), parents=[self])
        
        if False:
        #if len(self.parents) == 0:
            y.derivatives[self] = x * self.val

        else:
            for val in self.derivatives:
                y.derivatives[val] += self.derivatives[val] * x * self.val

        return y


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

L4 = (W * X - Y) ** 2 + W ** 2

#aa = L4.backwards()

set_trace()
