import chumpy as cp
import numpy as np

'''
x = cp.array(np.ones((3,3)))
y = cp.array(np.zeros((3,3)))
z = cp.array(np.zeros((3,3)) + 2)

t1 = x + y

t2 = z * t1
print ('t2', t2.shape)
t3 = t2.sum()

print ('Init', t3)
loss = (25 - t3) ** 2
print ('loss', loss)


#import sys
#sys.exit()

methods = ['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SGDMom'] #Nelder-mead is the finite difference simplex method
method = 5

def cb(_):
    pass

#options = {'disp': True, 'maxiter': 1000, 'lr':2e-3, 'momentum' : 0.4, 'decay' :0.9, 'tol' : 1e-7}
options = {'disp': True, 'maxiter': 1000, 'lr':2e-3, 'momentum' : 0.4, 'decay' :0.9, 'tol' : 1e-7}
#import ipdb
#ipdb.set_trace()
cp.minimize(loss, bounds=None, method=methods[method], x0=[x,y,z], callback=cb, options=options)
#print (loss, t2)
#print (hasattr(loss, "J"))
print ('Optimized ', x, y, z, ((x+y) * z).sum() )
'''
x = cp.array( np.ones((3,1)) )
y = cp.array( np.zeros((3,1)) )
w = cp.array( np.ones((3,1)) )
t = cp.vstack([x, y, w])

z = cp.array( np.zeros((9,1)) + 1.1 )
t1 = t + z
t3 = t1.sum()
print ('Init', t3)
loss = (20 - t3) ** 2
print ('loss', loss)

methods = ['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SGDMom'] #Nelder-mead is the finite difference simplex method
method = 5

def cb(_):
    pass

#options = {'disp': True, 'maxiter': 1000, 'lr':2e-3, 'momentum' : 0.4, 'decay' :0.9, 'tol' : 1e-7}
options = {'disp': True, 'maxiter': 1000, 'lr':2e-3, 'momentum' : 0.4, 'decay' :0.9, 'tol' : 1e-7}
#import ipdb
#ipdb.set_trace()
cp.minimize(loss, bounds=None, method=methods[method], x0=[x,y], callback=cb, options=options)
print ('Optimized ', t, z, (t1).sum() )
