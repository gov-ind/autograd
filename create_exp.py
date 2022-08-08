
from random import randint, choice, seed
import numpy as np

#seed(0)
N = 100000

def create_exp():
    vars = [f'a{i}' for i in range(N)]
    ops = [' + ', ' - ', ' * ', ' ** ', ' ( ', ' ) ']
    in_parens = False
    exp = choice(vars)
    exp_torch = exp + '_t'
    exp_tf = exp + '_tf'
    prev = exp

    var_set = set([exp])

    for _ in range(N):
        op = choice(ops)
        if op == ' ) ' and prev != ' ) ' and prev != ' ( ' and in_parens:
            exp += op
            exp_torch += op
            exp_tf += op
            prev = op
            
            in_parens = False
            continue
 
        if op == ' ( ' and prev != ' ( ' and prev != ' ) ' and not in_parens:
            var = choice(vars)
            pre_op = choice(ops[:3])
            exp += pre_op + op + var
            exp_torch += pre_op + op + var + '_t'
            exp_tf += pre_op + op + var + '_tf'            

            prev = var
            in_parens = True

            var_set.add(var)

            continue

        '''if op == ' ** ':
            var = str(randint(1, 2))
            exp += op + var
            exp_torch += op + var
            prev = var

            #var_set.add(var)

            continue
'''
        if op == ' + ' or op == ' - ' or op == ' * ':
            var = choice(vars)
            exp += op + var
            exp_torch += op + var + '_t'
            exp_tf += op + var + '_tf'

            prev = var

            var_set.add(var)

            op = choice(ops)

    if in_parens:
        exp += ')'
        exp_torch += ')'
        exp_tf += ')'

    var_exp = ''
    var_torch = ''
    var_tf = ''
    del_var = ''
    del_var_torch = ''
    del_var_tf = ''

    for var in list(var_set):
        val = str(randint(1, 5))
        #val = str(np.random.randn(1)[0])
        var_exp += var + ' = Var(' + val + ');'
        del_var += 'del ' + var + ';'
        var_torch += var + '_t = torch.tensor([' + val + '.], requires_grad=True);'
        del_var_torch = 'del ' + var + '_t;'

        var_tf += var + '_tf = tf.Variable(' + val + '.0);'
        del_var_tf = 'del ' + var + '_tf;'

    return 'L = ' + exp, 'L_t = ' + exp_torch, 'L_tf = ' + exp_tf, var_exp, var_torch, var_tf, del_var, del_var_torch, del_var_tf
