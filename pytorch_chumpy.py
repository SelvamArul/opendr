import torch as th

import chumpy as ch
from chumpy import ch, utils
from chumpy.utils import row, col
from chumpy.utils import *
from chumpy import optimization
import numpy as np

th_x= th.nn.Parameter(th.ones(1, requires_grad=True))
th_y= th.nn.Parameter(th.ones(1, requires_grad=True) + 1)
optimizer = th.optim.SGD([th_y, th_x], lr=1e-5, momentum=0.5)

for i in range(100):
    #print ('x', th_x)
    #print ('y', th_y)

    th_a = (th_x**2) + (th_y**2)
    th_b = (th_x + th_y)
    th_c = (th_x**2) + th_y

    ch_a = ch.array(th_a.detach().numpy())
    ch_b = ch.array(th_b.detach().numpy())
    ch_c = ch.array(th_c.detach().numpy())

    #print ('ch_a', ch_a)
    #print ('ch_b', ch_b)
    #print ('ch_c', ch_c)

    ch_d = ch_a + ch_b + ch_c
    ch_e = (ch_a ** 2) + (ch_b ** 2) + ch_c

    th_d = th.from_numpy(ch_d.r)
    th_e = th.from_numpy(ch_e.r)

    #print ('ch_d', ch_d)
    #print ('ch_e', ch_e)
    th_d.requires_grad=True
    th_e.requires_grad=True

    #print ('th_d', th_d.requires_grad)
    #print ('th_e', th_e.requires_grad)

    th_f = th_d + th_e
    #print('th_f', th_f)

    th_l = (50 - th_f) **2

    print ('loss {:.2f}'.format (th_l.item()))
    _l = th_l.item()

    optimizer.zero_grad()
    th_l.backward()
    #print ('pytorch grads')
    #print (th_d.grad)
    #print (th_e.grad)

    obj = [ch_d, ch_e]
    obj = ch.concatenate([f.ravel() for f in obj])
    free_variables = [ch_a, ch_b, ch_c]
    obj = optimization.ChInputsStacked(obj=obj, free_variables=free_variables, x=np.concatenate([freevar.r.ravel() for freevar in free_variables]))

    _j = obj.J

    _x = _j.toarray().T
    rhs = np.array([th_d.grad.item(), th_e.grad.item()]).reshape(2,1)
    
    #print ('---',_x.shape, rhs.shape)
    _x = _x.dot(rhs)
    #print (_x)
    grad_ch_a = _x[0]
    grad_ch_b = _x[1]
    grad_ch_c = _x[2]

    #grad_ch_a = _x[:,0]
    #grad_ch_b = _x[:,1]
    #grad_ch_c = _x[:,2]

    _a = th.from_numpy(grad_ch_a).float()
    th_a.backward(_a)
    _b = th.from_numpy(grad_ch_b).float()
    th_b.backward(_b)
    _c = th.from_numpy(grad_ch_c).float()
    th_b.backward(_c)

    optimizer.step()

    #print (' Updated inputs')
    #print ('{:.2f} {:.2f}'.format( th_x.item(), th_y.item() ))
    #print ()
    if _l < 1e-3:
        break

