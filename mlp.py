import chumpy as cp


x = cp.array(3)
y = cp.array(4)
z = cp.array(5)

t1 = x + y

t2 = z * t1
print ('t2', t2.shape)
t3 = t2.sum()

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

