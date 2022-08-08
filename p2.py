from pdb import set_trace

class Var():
    def __init__(self, val, parent=None):
        self.val = val

    def __add__(self, x):
        set_trace()
        y = Var(self.val + x.val, self)
        self.forward_der = 1
        x.forward_der = 1
        return y

    def __sub__(self, x):
        set_trace()
        y = Var(self.val - x.val, self)
        self.forward_der = 1
        x.forward_der = 1
        return y

    def __mul__(self, x):
        set_trace()
        y = Var(self.val * x.val, self)
        self.forward_der = x.val
        x.forward_der = self.val
        return y

    def __pow__(self, x):
        set_trace()
        y = Var(pow(self.val, x.val), self)
        self.forward_der = x.val * self.val
        return y


#z = (w * x - y) ** 2 + w ** 2
w, x, y = 1, 2, 3

set_trace()

W = Var(w)
X = Var(x)
Y = Var(y)

L0 = W * X
L1 = L0 - Y
L2 = L1 ** 2
L3 = W ** 2
L4 = L2 + L3
