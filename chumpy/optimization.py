#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['minimize']

import time
import math
import sys
import time
import numpy as np
from numpy.linalg import norm

from chumpy import ch, utils
from chumpy.utils import row, col

import scipy.sparse as sp
import scipy.sparse
import scipy.optimize
from scipy.sparse.linalg.interface import LinearOperator
import collections
import chumpy.minimize_ras as min_ras

from torchnet.logger import VisdomLogger, VisdomPlotLogger 
import visdom


# import probLineSearch as pls
# import ipdb


def vstack(x):
    x = [a if not isinstance(a, LinearOperator) else a.dot(np.eye(a.shape[1])) for a in x]
    return sp.vstack(x, format='csc') if any([sp.issparse(a) for a in x]) else np.vstack(x)
def hstack(x):
    x = [a if not isinstance(a, LinearOperator) else a.dot(np.eye(a.shape[1])) for a in x]
    return sp.hstack(x, format='csc') if any([sp.issparse(a) for a in x]) else np.hstack(x)


# Nelder-Mead
# Powell
# CG
# BFGS
# Newton-CG
# Anneal
# L-BFGS-B
# TNC
# COBYLA
# SLSQP
# dogleg
# trust-ncg

def chFuncProb(fun, grad, var_f, var_df, args):
    def funValues(X):
        f = fun(X, *args)
        df = grad(X, *args)

        return f,df, var_f, var_df
    return funValues


def minimize_sgdmom(obj, free_variables, lr=0.01, momentum=0.9, decay=0.9, tol=1e-9, on_step=None, maxiters=None, gt=None):

    env_name = 'test'
    port = 8097
    vis = visdom.Visdom(server='http://localhost', port=port, env=env_name)
    vis.close(env=env_name)

    obj_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='obj'))
    p_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='p'))
    dp_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='dp'))
    lr_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='lr'))
    j_logger = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='jacobian'))
    dp_norm_logger = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='dp_norm'))

    verbose = False
    labels = {}
    if isinstance(obj, list) or isinstance(obj, tuple):
        obj = ch.concatenate([f.ravel() for f in obj])
    elif isinstance(obj, dict):
        labels = obj
        obj = ch.concatenate([f.ravel() for f in list(obj.values())])

    unique_ids = np.unique(np.array([id(freevar) for freevar in free_variables]))
    num_unique_ids = len(unique_ids)

    if num_unique_ids != len(free_variables):
       raise Exception('The "free_variables" param contains duplicate variables.')
 
    obj = ChInputsStacked(obj=obj, free_variables=free_variables, x=np.concatenate([freevar.r.ravel() for freevar in free_variables]))
    # obj.show_difference()

    # import sys
    # sys.exit()
    def call_cb():
        if on_step is not None:
            on_step(obj)

        report_line = ""
        if len(labels) > 0:
            report_line += '%.2e | ' % (np.sum(obj.r**2),)
        for label in sorted(labels.keys()):
            objective = labels[label]
            report_line += '%s: %.2e | ' % (label, np.sum(objective.r**2))
        if len(labels) > 0:
            report_line += '\n'
        import sys
        sys.stderr.write(report_line)

    call_cb()

    import ipdb; ipdb.set_trace()

    # pif = print-if-verbose.
    # can't use "print" because it's a statement, not a fn
    verbose = False
    pif = lambda x: print(x) if verbose else 0

    # optimization parms
    k_max = maxiters

    k = 0
    p = col(obj.x.r)

    tm = time.time()
    pif('computing Jacobian...')
    J = obj.J
    if sp.issparse(J):
        assert(J.nnz > 0)
    print('Jacobian (%dx%d) computed in %.2fs' % (J.shape[0], J.shape[1], time.time() - tm))
    print ('p', p)
    if J.shape[1] != p.size:
        import pdb; pdb.set_trace()
    assert(J.shape[1] == p.size)

    tm = time.time()
    pif('updating A and g...')

    stop = False
    dp = np.array([[0]])

    bestParams = p
    bestEval = obj.r
    numWorse = 0
    lrWorse = 0
    while (not stop) and (k < k_max):
        print ('------',k,'-------------------------------')
        k += 1

        pif('beginning iteration %d' % (k,))
        arrJ = J
        if sp.issparse(J):
            arrJ = J.toarray()
            
        dp = col(lr*np.array(arrJ)) + momentum*dp

        #print (f'dp {dp}             p {p}')
        #print ('lr ', lr)

        if p.shape != dp.shape:
            import ipdb
            ipdb.set_trace()
        p_new = p - dp
        if k > 25:
            lr = lr*decay
        
        obj.x = p_new.ravel()

        #if norm(dp) < tol:
        #    print('stopping due to small update (%f) < (%f) ' % (norm(dp), tol))
        #    stop = True

        J = obj.J.copy()
        _loss = obj.r
        print ('Best {}  loss {}'.format(bestEval, _loss))
        if bestEval > _loss:
            numWorse = 0
            lrWorse = 0
            bestEval = obj.r.copy()
            bestParams = p.copy()
        else:
            numWorse += 1
            lrWorse += 1
            if numWorse >= 100:
                print("Stopping due to increasing evaluation error.")
                stop = True
                obj.x = bestParams.ravel()
                obj.r
            if lrWorse > 10:
                lrWorse = 0
                lr *= 0.9

        p = col(obj.x.r)
        call_cb()

        # visualize optimization process
        obj_logger.log(k, float(_loss), name='loss')
        obj_logger.log(k, bestEval, name='best')
        p_logger.log(k, float(p[0]),  name='p0')
        p_logger.log(k, float(p[1]),  name='p1')
        dp_logger.log(k, float(dp[0]),  name='dp0')
        dp_logger.log(k, float(dp[1]),  name='dp1')
        lr_logger.log(k, float(lr), name='lr')
        # import ipdb; ipdb.set_trace()
        j_logger.log(k, float(J[0][0]), name='J0')
        j_logger.log(k, float(J[0][1]), name='J1')
        if gt is not None:
            p_logger.log(k, float(gt[0]),  name='gt_p0')
            p_logger.log(k, float(gt[1]),  name='gt_p1')

        if k >= k_max:
            pif('stopping because max number of user-specified iterations (%d) has been met' % (k_max,))
    return obj.free_variables



# 
def minimize_Adagrad(obj, free_variables, lr=0.01, momentum=0.9, decay=0.9, tol=1e-7,
                    on_step=None, maxiters=None, gt=None, params=None,
                    resnet_loss=0, self_obj=0):
    
    if not isinstance(gt, collections.Mapping) or not isinstance(params, collections.Mapping):
        import sys
        sys.exit('gt and params should be a dict of translation and quaterions keys')

    
    gt_translation = gt['t']
    gt_quaterions = gt['q']

    ch_params_trans = params['t']
    ch_params_q = params['q']

    eps = 1e-8
    env_name = 'adagrad_test_WIP1'
    port = 8097
    vis = visdom.Visdom(server='http://localhost', port=port, env=env_name)
    vis.close(env=env_name)

    obj_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='obj'))
    p_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='p'))
    dp_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='dp'))
    lr_logger =  VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='lr'))
    j_logger = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='jacobian'))
    cache_logger  = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='cache'))
    dp_norm_logger = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='dp_norm'))
    t_error_logger = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='t_error'))
    q_error_logger = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='q_error'))
    angle_error_logger = VisdomPlotLogger('line', env=env_name, port=port, opts=dict(title='angle_error'))


    verbose = False
    labels = {}
    if isinstance(obj, list) or isinstance(obj, tuple):
        obj = ch.concatenate([f.ravel() for f in obj])
    elif isinstance(obj, dict):
        labels = obj
        obj = ch.concatenate([f.ravel() for f in list(obj.values())])

    unique_ids = np.unique(np.array([id(freevar) for freevar in free_variables]))
    num_unique_ids = len(unique_ids)

    if num_unique_ids != len(free_variables):
        import ipdb; ipdb.set_trace()
        raise Exception('The "free_variables" param contains duplicate variables.')
 
    obj = ChInputsStacked(obj=obj, free_variables=free_variables, x=np.concatenate([freevar.r.ravel() for freevar in free_variables]))

    # import ipdb; ipdb.set_trace()
    # obj.show_difference()

    # import sys
    # sys.exit()
    def call_cb():
        if on_step is not None:
            on_step(obj)

        report_line = ""
        if len(labels) > 0:
            report_line += '%.2e | ' % (np.sum(obj.r**2),)
        for label in sorted(labels.keys()):
            objective = labels[label]
            report_line += '%s: %.2e | ' % (label, np.sum(objective.r**2))
        if len(labels) > 0:
            report_line += '\n'
        import sys
        sys.stderr.write(report_line)

    call_cb()

    # obj.show_difference()
    # obj.show_label()
    # pif = print-if-verbose.
    # can't use "print" because it's a statement, not a fn
    verbose = False
    pif = lambda x: print(x) if verbose else 0

    # optimization parms
    k_max = maxiters

    k = 0
    p = col(obj.x.r)
    
    pif('computing Jacobian...')
    tm = time.time()
    if resnet_loss:
        J, _cnn_loss = self_obj.jacobian_wrt_rendering()
    else:
        J = obj.J
    if sp.issparse(J):
        assert(J.nnz > 0)
    # print('Jacobian (%dx%d) computed in %.2fs' % (J.shape[0], J.shape[1], time.time() - tm))
    # print ('p', p)
    if J.shape[1] != p.size:
        import ipdb; ipdb.set_trace()
    assert(J.shape[1] == p.size)

    stop = False
    cache = ch.zeros_like( J )
    M = ch.zeros_like( J )
    R = ch.zeros_like( J )

    bestParams = p

    if resnet_loss:
        obj.r
        bestEval =  _cnn_loss
    else:
        bestEval = obj.r
    
    numWorse = 0
    lrWorse = 0

    gamma = 0.5
    beta_1 = 0.9
    beta_2 = 0.999

    while (not stop) and (k < k_max):
        # print ('---------', k,'----------------------------')
        k += 1
        arrJ = J
        if sp.issparse(J):
            arrJ = J.toarray()

        tm = time.time()
        # Adagrad
        cache += arrJ ** 2
        
        # RMSProp
        # cache = gamma * cache +  (1- gamma) * (arrJ ** 2)
        
        # Adagrad , RMSProp
        dp = col( lr * arrJ / ( ch.sqrt( cache ) + eps) )

        # Adam
        # import ipdb; ipdb.set_trace()
        # M = beta_1 * M + (1-beta_1) * arrJ
        # R = beta_2 * R + (1-beta_2) * arrJ ** 2
        # M_hat = M / (1-beta_1 ** k)
        # R_hat = R / (1-beta_2 ** k)
        # dp = col( lr * M_hat / (ch.sqrt(R_hat) + eps ) )

        # print ('lr ', lr)
        
        # if k < 3:
        #     dp = dp * 0.01
        p_new = p - dp
        if k > 15:
            lr = lr*decay
        
        
        obj.x = p_new.ravel()
        # print('Params updated in %.2fs' %  (time.time() - tm))
        if norm(dp) < tol:
            print('stopping due to small update (%f) < (%f) ' % (norm(dp), tol))
            stop = True
        # if abs(float(dp.max())) < 5e-4:
        #     print (' stopping do to small max (%f) < (%f) ' % (float(dp.max()), 5e-4))
        #     stop = True
        tm = time.time()
        if resnet_loss:
            J, _cnn_loss = self_obj.jacobian_wrt_rendering()
            # print('_cnn_loss ', _cnn_loss)
        else:
            J = obj.J.copy()
        
        # print('Jacobian (%dx%d) computed in %.2fs' %  (J.shape[0], J.shape[1], time.time() - tm))
        # print ('Jacobian \n', J)
        # import ipdb; ipdb.set_trace()
        if resnet_loss:
            obj.r
            _loss =  _cnn_loss
        else:
            _loss = obj.r
        
        # print ('Best {}  loss {}'.format(bestEval, _loss))
        if bestEval > _loss:
            numWorse = 0
            lrWorse = 0
            bestEval = _loss
            bestParams = p.copy()
        else:
            numWorse += 1
            lrWorse += 1
            if numWorse >= 15:
                print("Stopping due to increasing evaluation error.")
                stop = True
                obj.x = bestParams.ravel()
                obj.r
            if lrWorse > 10:
                lrWorse = 0
                lr *= 0.95

        p = col(obj.x.r)
        call_cb()
        # import ipdb; ipdb.set_trace()
        # visualize optimization process
        obj_logger.log(k, float(_loss), name='loss')
        obj_logger.log(k, bestEval, name='best')
        lr_logger.log(k, float(lr), name='lr')
        dp_norm_logger.log(k, float(norm(dp)), name='dp_norm')
        dp_norm_logger.log(k, float(dp.min()), name='dp_min')
        dp_norm_logger.log(k, float(dp.max()), name='dp_max')

        cache_t = cache.T
        for _i in range(p.shape[0]):
            # p_logger.log(k, float(p[_i]),  name='p{}'.format(_i))
            dp_logger.log(k, float(dp[_i]),  name='dp{}'.format(_i))
            j_logger.log(k, float(J[0][_i]), name='J{}'.format(_i))
            cache_logger.log(k, float(cache_t[_i]), name='Cache{}'.format(_i))
            # if gt is not None:
            #     p_logger.log(k, float(gt[_i]),  name='gt_p{}'.format(_i))

        _t_total = 0
        _q_total = 0

        for _k in ch_params_trans.keys():

            if _k not in gt_quaterions.keys():
                print (_k, 'is not in ground truth')
                continue
            
            # import ipdb; ipdb.set_trace()
            for _i in range( gt_quaterions[_k].shape[0] ):
                p_logger.log(k, float(gt_quaterions[_k][_i]), name = 'gt_q{}_{}'.format(_k, _i))
                p_logger.log(k, float(ch_params_q[_k][_i]), name = 'q{}_{}'.format(_k, _i))
            
            for _i in range( gt_translation[_k].shape[0] ):
                p_logger.log(k, float(gt_translation[_k][_i]), name = 'gt_t{}_{}'.format(_k, _i))
                p_logger.log(k, float(ch_params_trans[_k][_i]), name = 't{}_{}'.format(_k, _i))
            # p_logger.log(k, float( gt_translation[_k]), name = 'gt_p{}'.format(_k))

            

            
            _t_norm = np.linalg.norm(gt_translation[_k] - ch_params_trans[_k])
            t_error_logger.log(k, float(_t_norm), name="t_{}".format(_k))
            if _t_norm < 0.05:
                pass
                # import ipdb; ipdb.set_trace()
            _t_total += _t_norm

            _q = ch_params_q[_k].copy().x
            if _q[0] < 0:
                _q *= -1
            else:
                _q *= 1
            
            _q_gt = gt_quaterions[_k].copy()
            if _q_gt[0] < 0:
                _q_gt *= -1
             
            _q_norm = np.linalg.norm(_q_gt - _q)

            q_error_logger.log(k, float(_q_norm), name="q_{}".format(_k))

            # compute angle between quaternions
            
            _q_gt_inv = _q_gt.copy()
            _q_gt_inv[1:] *= -1

            _q /= np.linalg.norm(_q)

            _q_res = quaternion_product(_q_gt_inv, _q)
            
            # clamp between -1 and 1
            _q_0 = max(min(_q_res[0], 1), -1)
            
            _theta = 2 * np.arccos(_q_0)
            _theta  = (_theta / np.pi) * 180      
            angle_error_logger.log(k, float(_theta), name="q_{}".format(_k))
            _q_total += _q_norm

        # text_logger.log('T loss: {}'.format(float(_t_total)))
        # text_logger.log('Q loss: {}'.format(float(_q_total)))
        # print ('T loss: {}'.format(float(_t_total)))
        # print ('Q loss: {}'.format(float(_q_total)))
        if k >= k_max:
            obj.x = bestParams.ravel()
            obj.r
            print('stopping because max number of user-specified iterations (%d) has been met' % (k_max,))
        # print (self_obj.ch_params_q)
        # for _k, _v in self_obj.ch_params_q.items():
        #     print ( np.linalg.norm(_v))
    # print ('Optimized')
    # for _k, _v in ch_params_trans.items():
        # print (_k, _v)
    return obj.free_variables



def quaternion_product(quaternion1, quaternion0):
	w0, x0, y0, z0 = quaternion0[0],quaternion0[1],quaternion0[2],quaternion0[3]
	w1, x1, y1, z1 = quaternion1[0],quaternion1[1],quaternion1[2],quaternion1[3]
	res = np.ones(4)
	res[0] = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
	res[1] = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
	res[2] = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
	res[3] = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
	return res


def gradCheckSimple(fun, var, delta):
    f0 = fun.r
    oldvar = var.r[:].copy()
    var[:] = var.r[:] + delta
    f1 = fun.r
    diff = (f1 - f0)/np.abs(delta)
    var[:] = oldvar.copy()
    return diff

def gradCheck(fun, vars, delta):

    jacs = np.concatenate([fun.dr_wrt(wrt).toarray()[0] for wrt in vars])
    approxjacs = []
    for idx, freevar in enumerate(vars):
        approxjacs = approxjacs + [gradCheckSimple(fun, freevar, delta[idx])]
    approxjacs = np.concatenate(approxjacs)
    check = jacs/approxjacs
    return jacs, approxjacs, check

def scipyGradCheck(fun, x0):

    if isinstance(fun, list) or isinstance(fun, tuple):
        fun = ch.concatenate([f.ravel() for f in fun])
    if isinstance(fun, dict):
        fun = ch.concatenate([f.ravel() for f in list(fun.values())])

    obj = fun
    free_variables = x0

    from chumpy.ch import SumOfSquares

    hessp = None
    hess = None
    if obj.size == 1:
        obj_scalar = obj
    else:
        obj_scalar = SumOfSquares(obj)

        def hessp(vs, p,obj, obj_scalar, free_variables):
            changevars(vs,obj,obj_scalar,free_variables)
            if not hasattr(hessp, 'vs'):
                hessp.vs = vs*0+1e16
            if np.max(np.abs(vs-hessp.vs)) > 0:

                J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                hessp.J = J
                hessp.H = 2. * J.T.dot(J)
                hessp.vs = vs
            return np.array(hessp.H.dot(p)).ravel()
            #return 2*np.array(hessp.J.T.dot(hessp.J.dot(p))).ravel()

        if method.lower() != 'newton-cg':
            def hess(vs, obj, obj_scalar, free_variables):
                changevars(vs,obj,obj_scalar,free_variables)
                if not hasattr(hessp, 'vs'):
                    hessp.vs = vs*0+1e16
                if np.max(np.abs(vs-hessp.vs)) > 0:
                    J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                    hessp.H = 2. * J.T.dot(J)
                return hessp.H

    def changevars(vs, obj, obj_scalar, free_variables):
        cur = 0
        changed = False
        for idx, freevar in enumerate(free_variables):
            sz = freevar.r.size
            newvals = vs[cur:cur+sz].copy().reshape(free_variables[idx].shape)
            if np.max(np.abs(newvals-free_variables[idx]).ravel()) > 0:
                free_variables[idx][:] = newvals
                changed = True

            cur += sz

        return changed

    def residuals(vs,obj, obj_scalar, free_variables):
        changevars(vs, obj, obj_scalar, free_variables)
        residuals = obj_scalar.r.ravel()[0]
        return residuals

    def scalar_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(scalar_jacfunc, 'vs'):
            scalar_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-scalar_jacfunc.vs)) == 0:
            return scalar_jacfunc.J

        changevars(vs, obj, obj_scalar, free_variables)

        if False: # faster, at least on some problems
            result = np.concatenate([np.array(obj_scalar.lop(wrt, np.array([[1]]))).ravel() for wrt in free_variables])
        else:
            jacs = [obj_scalar.dr_wrt(wrt) for wrt in free_variables]
            for idx, jac in enumerate(jacs):
                if sp.issparse(jac):
                    jacs[idx] = jacs[idx].toarray()
            result = np.concatenate([jac.ravel() for jac in jacs])

        scalar_jacfunc.J = result
        scalar_jacfunc.vs = vs
        return np.squeeze(np.asarray(result.ravel()))

    def ns_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(ns_jacfunc, 'vs'):
            ns_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-ns_jacfunc.vs)) == 0:
            return ns_jacfunc.J

        changevars(vs, obj, obj_scalar, free_variables)
        jacs = [obj.dr_wrt(wrt) for wrt in free_variables]
        result = hstack(jacs)

        ns_jacfunc.J = result
        ns_jacfunc.vs = vs
        return result

    err = scipy.optimize.check_grad(residuals, scalar_jacfunc, np.concatenate([free_variable.r.ravel() for free_variable in free_variables]), obj, obj_scalar, free_variables)
    print("Grad check (Root Sum Sq. of Diff.) error of real and finite difference gradients: " + str(err))

    return err



def minimize(fun, x0, method='dogleg', bounds=None, constraints=(), tol=None,
            callback=None, options=None, gt=None, params=None, resnet_loss=False,
            self_obj=None):

    if method == 'dogleg':
        if options is None: options = {}
        return _minimize_dogleg(fun, free_variables=x0, on_step=callback, **options)

    maxiter = None
    if options != None:
        maxiter = options['maxiter']

    if isinstance(fun, list) or isinstance(fun, tuple):
        fun = ch.concatenate([f.ravel() for f in fun])
    if isinstance(fun, dict):
        fun = ch.concatenate([f.ravel() for f in list(fun.values())])
    obj = fun
    free_variables = x0

    from chumpy.ch import SumOfSquares

    hessp = None
    hess = None
    if obj.size == 1:
        obj_scalar = obj
    else:
        obj_scalar = SumOfSquares(obj)
    
        def hessp(vs, p,obj, obj_scalar, free_variables):
            changevars(vs,obj,obj_scalar,free_variables)
            if not hasattr(hessp, 'vs'):
                hessp.vs = vs*0+1e16
            if np.max(np.abs(vs-hessp.vs)) > 0:

                J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                hessp.J = J
                hessp.H = 2. * J.T.dot(J)
                hessp.vs = vs
            return np.array(hessp.H.dot(p)).ravel()
            #return 2*np.array(hessp.J.T.dot(hessp.J.dot(p))).ravel()
            
        if method.lower() != 'newton-cg':
            def hess(vs, obj, obj_scalar, free_variables):
                changevars(vs,obj,obj_scalar,free_variables)
                if not hasattr(hessp, 'vs'):
                    hessp.vs = vs*0+1e16
                if np.max(np.abs(vs-hessp.vs)) > 0:
                    J = ns_jacfunc(vs,obj,obj_scalar,free_variables)
                    hessp.H = 2. * J.T.dot(J)
                return hessp.H
        
    def changevars(vs, obj, obj_scalar, free_variables):
        cur = 0
        changed = False
        for idx, freevar in enumerate(free_variables):
            sz = freevar.r.size
            newvals = vs[cur:cur+sz].copy().reshape(free_variables[idx].shape)
            if np.max(np.abs(newvals-free_variables[idx]).ravel()) > 0:
                free_variables[idx][:] = newvals
                changed = True

            cur += sz

        methods_without_callback = ('anneal', 'powell', 'cobyla', 'slsqp')
        if callback is not None and changed and method.lower() in methods_without_callback:
            callback(None)

        return changed

    def residuals(vs,obj, obj_scalar, free_variables):
        changevars(vs, obj, obj_scalar, free_variables)
        residuals = obj_scalar.r.ravel()[0]
        return residuals

    def scalar_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(scalar_jacfunc, 'vs'):
            scalar_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-scalar_jacfunc.vs)) == 0:
            return scalar_jacfunc.J

        changevars(vs, obj, obj_scalar, free_variables)

        if False: # faster, at least on some problems
            result = np.concatenate([np.array(obj_scalar.lop(wrt, np.array([[1]]))).ravel() for wrt in free_variables])
        else:
            jacs = [obj_scalar.dr_wrt(wrt) for wrt in free_variables]
            #import ipdb
            #ipdb.set_trace()
            for idx, jac in enumerate(jacs):
                if sp.issparse(jac):
                    jacs[idx] = jacs[idx].toarray()
            result = np.concatenate([jac.ravel() for jac in jacs])

        scalar_jacfunc.J = result
        scalar_jacfunc.vs = vs
        return np.squeeze(np.asarray(result.ravel()))
        
    def ns_jacfunc(vs,obj, obj_scalar, free_variables):
        if not hasattr(ns_jacfunc, 'vs'):
            ns_jacfunc.vs = vs*0+1e16
        if np.max(np.abs(vs-ns_jacfunc.vs)) == 0:
            return ns_jacfunc.J
            
        changevars(vs, obj, obj_scalar, free_variables)
        jacs = [obj.dr_wrt(wrt) for wrt in free_variables]
        result = hstack(jacs)

        ns_jacfunc.J = result
        ns_jacfunc.vs = vs
        return result

    if method == 'minimize':
        x1, fX, i = min_ras.minimize(np.concatenate([free_variable.r.ravel() for free_variable in free_variables]), residuals, scalar_jacfunc, args=(obj, obj_scalar, free_variables), on_step=callback, maxnumfuneval=maxiter)
    elif method == 'SGDMom':
        return minimize_Adagrad(obj=fun, free_variables=x0 , lr=options['lr'], momentum=options['momentum'], decay=options['decay'],
                    on_step=callback, maxiters=maxiter, gt=gt, params=params, 
                    resnet_loss=resnet_loss, self_obj=self_obj)
    else:
        print ('Invoking Scipy optimize')
        x1 = scipy.optimize.minimize(
            method=method,
            fun=residuals,
            callback=callback,
            x0=np.concatenate([free_variable.r.ravel() for free_variable in free_variables]),
            jac=scalar_jacfunc,
            hessp=hessp, hess=hess, args=(obj, obj_scalar, free_variables),
            bounds=bounds, constraints=constraints, tol=tol, options=options).x

    changevars(x1, obj, obj_scalar, free_variables)

    return free_variables

_giter = 0
class ChInputsStacked(ch.Ch):
    dterms = 'x', 'obj'
    terms = 'free_variables'

    def compute_r(self):
        return self.obj.r.ravel()
    
    def dr_wrt(self, wrt):
        
        if wrt is self.x:
            mtxs = []
            for freevar in self.free_variables:
                if isinstance(freevar, ch.Select):
                    
                    new_mtx = self.obj.dr_wrt(freevar.a)
                    if new_mtx is None:
                        import ipdb; ipdb.set_trace()
                        new_mtx = self.obj.dr_wrt(freevar.a)
                    try:
                        mtxs.append(new_mtx[:,freevar.idxs])
                    except:
                        mtxs.append(new_mtx.tocsc()[:,freevar.idxs])
                else:
                    mtxs.append(self.obj.dr_wrt(freevar, reverse_mode=False))
            _t = hstack(mtxs)
            return _t
            #return hstack([self.obj.dr_wrt(freevar) for freevar in self.free_variables])
    
    def on_changed(self, which):
        global _giter
        _giter += 1
        
        if 'x' in which:
            pos = 0
            for idx, freevar in enumerate(self.free_variables):
                sz = freevar.r.size
                rng = np.arange(pos, pos+sz)
                if isinstance(self.free_variables[idx], ch.Select):

                    if not hasattr(self.free_variables[idx].a, 'x'):
                        print ('self.free_variables[idx].a does not have attribute "x"' )
                        print ('This happens because self.free_variables[idx].a is expected to be of type "ch.Ch" ')
                        print (' but got type ', type(self.free_variables[idx].a))
                        print ('One common mistake is to have superfluous dimenions')
                        print ('chumpy allows flexible number of dimensions')
                        print ('x[r][c][0][0] works for an array of just two dimensions')
                        
                    newv = self.free_variables[idx].a.x.copy()
                    newv.ravel()[self.free_variables[idx].idxs] = self.x.r[rng]
                    self.free_variables[idx].a.__setattr__('x', newv, _giter)
                    #self.free_variables[idx].a.x = newv
                elif isinstance(self.free_variables[idx].x, np.ndarray):
                    #self.free_variables[idx].x = self.x.r[rng].copy().reshape(self.free_variables[idx].x.shape)
                    self.free_variables[idx].__setattr__('x', self.x.r[rng].copy().reshape(self.free_variables[idx].x.shape), _giter)
                else: # a number
                    #self.free_variables[idx].x = self.x.r[rng]
                    self.free_variables[idx].__setattr__('x', self.x.r[rng], _giter)
                #self.free_variables[idx] = self.obj.replace(freevar, Ch(self.x.r[rng].copy()))
                pos += sz
    

    @property
    def J(self):
        result = self.dr_wrt(self.x).copy()
        return np.atleast_2d(result) if not sp.issparse(result) else result
    
    def JT_dot(self, y):
        return self.J.T.dot(y)
    
    def J_dot(self, y):
        return self.J.dot(y)
    
    # Have not observed this to be faster than just using J directly
    def JTJ(self):
        if False:
            return self.J.T.dot(self.J)
        else:
            Js = [self.obj.dr_wrt(freevar) for freevar in self.free_variables]
            zeroArray=[None]*len(Js)
            A = [zeroArray[:] for i in range(len(Js))]
            for y in range(len(Js)):
                for x in range(len(Js)):
                    if y > x:
                        A[y][x] = A[x][y].T
                    else:
                        A[y][x] = Js[y].T.dot(Js[x])
            return vstack([hstack(A[y]) for y in range(len(Js))])

    
_solver_fns = {
    'cg': lambda A, x, M=None : scipy.sparse.linalg.cg(A, x, M=M, tol=1e-10)[0],
    'spsolve': lambda A, x : scipy.sparse.linalg.spsolve(A, x)
}



def _minimize_dogleg(obj, free_variables, on_step=None,
                     maxiter=200, max_fevals=np.inf, sparse_solver='spsolve',
                     disp=False, show_residuals=None, e_1=1e-18, e_2=1e-18, e_3=0., delta_0=None):

    """"Nonlinear optimization using Powell's dogleg method.

    See Lourakis et al, 2005, ICCV '05, "Is Levenberg-Marquardt
    the Most Efficient Optimization for Implementing Bundle
    Adjustment?":
    http://www.ics.forth.gr/cvrl/publications/conferences/0201-P0401-lourakis-levenberg.pdf
    """

    import warnings
    if show_residuals is not None:
        import warnings
        warnings.warn('minimize_dogleg: show_residuals parm is deprecaed, pass a dict instead.')

    labels = {}
    if isinstance(obj, list) or isinstance(obj, tuple):
        obj = ch.concatenate([f.ravel() for f in obj])
    elif isinstance(obj, dict):
        labels = obj
        obj = ch.concatenate([f.ravel() for f in list(obj.values())])


    niters = maxiter
    verbose = disp
    num_unique_ids = len(np.unique(np.array([id(freevar) for freevar in free_variables])))
    if num_unique_ids != len(free_variables):
        raise Exception('The "free_variables" param contains duplicate variables.')
        
    obj = ChInputsStacked(obj=obj, free_variables=free_variables, x=np.concatenate([freevar.r.ravel() for freevar in free_variables]))

    def call_cb():
        if on_step is not None:
            on_step(obj)

        report_line = ""
        if len(labels) > 0:
            report_line += '%.2e | ' % (np.sum(obj.r**2),)
        for label in sorted(labels.keys()):
            objective = labels[label]
            report_line += '%s: %.2e | ' % (label, np.sum(objective.r**2))
        if len(labels) > 0:
            report_line += '\n'
        sys.stderr.write(report_line)

    call_cb()

    # pif = print-if-verbose.
    # can't use "print" because it's a statement, not a fn
    pif = lambda x: print(x) if verbose else 0

    if isinstance(sparse_solver, collections.Callable):
        solve = sparse_solver
    elif isinstance(sparse_solver, str) and sparse_solver in list(_solver_fns.keys()):
        solve = _solver_fns[sparse_solver]
    else:
        raise Exception('sparse_solver argument must be either a string in the set (%s) or have the api of scipy.sparse.linalg.spsolve.' % ''.join(list(_solver_fns.keys()), ' '))

    # optimization parms
    k_max = niters
    fevals = 0

    k = 0
    delta = delta_0
    p = col(obj.x.r) 

    fevals += 1
    
    tm = time.time()
    pif('computing Jacobian...')
    J = obj.J

    if sp.issparse(J):
        assert(J.nnz > 0)
    pif('Jacobian (%dx%d) computed in %.2fs' % (J.shape[0], J.shape[1], time.time() - tm))
    
    if J.shape[1] != p.size:
        import pdb; pdb.set_trace()
    assert(J.shape[1] == p.size)
    
    tm = time.time()
    pif('updating A and g...')
    A = J.T.dot(J)    
    r = col(obj.r.copy())
    
    g = col(J.T.dot(-r))
    pif('A and g updated in %.2fs' % (time.time() - tm))
    
    stop = norm(g, np.inf) < e_1
    while (not stop) and (k < k_max) and (fevals < max_fevals):
        k += 1
        pif('beginning iteration %d' % (k,))
        d_sd = col((sqnorm(g)) / (sqnorm (J.dot(g))) * g)
        GNcomputed = False

        while True:
            # if the Cauchy point is outside the trust region,
            # take that direction but only to the edge of the trust region
            if delta is not None and norm(d_sd) >= delta:
                pif('PROGRESS: Using stunted cauchy')
                d_dl = np.array(col(delta/norm(d_sd) * d_sd))
            else:
                if not GNcomputed:
                    tm = time.time()
                    if scipy.sparse.issparse(A):
                        A.eliminate_zeros()
                        pif('sparse solve...sparsity infill is %.3f%% (hessian %dx%d), J infill %.3f%%' % (
                            100. * A.nnz / (A.shape[0] * A.shape[1]),
                            A.shape[0],
                            A.shape[1],
                            100. * J.nnz / (J.shape[0] * J.shape[1])))
                            
                        if g.size > 1:             
                            d_gn = col(solve(A, g))
                            if np.any(np.isnan(d_gn)) or np.any(np.isinf(d_gn)):
                                from scipy.sparse.linalg import lsqr
                                d_gn = col(lsqr(A, g)[0])
                        else:
                            d_gn = np.atleast_1d(g.ravel()[0]/A[0,0])
                        pif('sparse solve...done in %.2fs' % (time.time() - tm))
                    else:
                        pif('dense solve...')
                        try:
                            d_gn = col(np.linalg.solve(A, g))
                        except Exception:
                            d_gn = col(np.linalg.lstsq(A, g)[0])
                        pif('dense solve...done in %.2fs' % (time.time() - tm))
                    GNcomputed = True

                # if the gauss-newton solution is within the trust region, use it
                if delta is None or norm(d_gn) <= delta:
                    pif('PROGRESS: Using gauss-newton solution')
                    d_dl = np.array(d_gn)
                    if delta is None:
                        delta = norm(d_gn)

                else: # between cauchy step and gauss-newton step
                    pif('PROGRESS: between cauchy and gauss-newton')

                    # compute beta multiplier
                    delta_sq = delta**2
                    pnow = (
                        (d_gn-d_sd).T.dot(d_gn-d_sd)*delta_sq
                        + d_gn.T.dot(d_sd)**2
                        - sqnorm(d_gn) * (sqnorm(d_sd)))
                    B = delta_sq - sqnorm(d_sd)
                    B /= ((d_gn-d_sd).T.dot(d_sd) + math.sqrt(pnow))

                    # apply step
                    d_dl = np.array(d_sd + float(B) * (d_gn - d_sd))

                    #assert(math.fabs(norm(d_dl) - delta) < 1e-12)
            if norm(d_dl) <= e_2*norm(p):
                pif('stopping because of small step size (norm_dl < %.2e)' % (e_2*norm(p)))
                stop = True
            else:
                p_new = p + d_dl

                tm_residuals = time.time()
                obj.x = p_new
                fevals += 1

                r_trial = obj.r.copy()
                tm_residuals = time.time() - tm

                # rho is the ratio of...
                # (improvement in SSE) / (predicted improvement in SSE)  
                
                # slower
                #rho = norm(e_p)**2 - norm(e_p_trial)**2
                #rho = rho / (L(d_dl*0, e_p, J) - L(d_dl, e_p, J))              
                
                # faster
                sqnorm_ep = sqnorm(r)
                rho = sqnorm_ep - norm(r_trial)**2
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',category=RuntimeWarning)
                    if rho > 0:
                        rho /= predicted_improvement(d_dl, -r, J, sqnorm_ep, A, g)
                    
                improvement_occurred = rho > 0

                # if the objective function improved, update input parameter estimate.
                # Note that the obj.x already has the new parms,
                # and we should not set them again to the same (or we'll bust the cache)                
                if improvement_occurred:
                    p = col(p_new)
                    call_cb()

                    if (sqnorm_ep - norm(r_trial)**2) / sqnorm_ep < e_3:
                        stop = True
                        pif('stopping because improvement < %.1e%%' % (100*e_3,))


                else:  # Put the old parms back
                    obj.x = ch.Ch(p)
                    obj.on_changed('x') # copies from flat vector to free variables

                # if the objective function improved and we're not done,
                # get ready for the next iteration
                if improvement_occurred and not stop:
                    tm_jac = time.time()
                    pif('computing Jacobian...')
                    J = obj.J.copy()
                    tm_jac = time.time() - tm_jac
                    pif('Jacobian (%dx%d) computed in %.2fs' % (J.shape[0], J.shape[1], tm_jac))

                    pif('Residuals+Jac computed in %.2fs' % (tm_jac + tm_residuals,))

                    tm = time.time()
                    pif('updating A and g...')
                    A = J.T.dot(J)
                    r = col(r_trial)
                    g = col(J.T.dot(-r))
                    pif('A and g updated in %.2fs' % (time.time() - tm))
                    
                    if norm(g, np.inf) < e_1:
                        stop = True
                        pif('stopping because norm(g, np.inf) < %.2e' % (e_1))

                # update our trust region
                delta = updateRadius(rho, delta, d_dl)
                
                if delta <= e_2*norm(p):
                    stop = True
                    pif('stopping because trust region is too small')

            # the following "collect" is very expensive.
            # please contact matt if you find situations where it actually helps things.
            #import gc; gc.collect()
            if stop or improvement_occurred or (fevals >= max_fevals):
                break
        if k >= k_max:
            pif('stopping because max number of user-specified iterations (%d) has been met' % (k_max,))
        elif fevals >= max_fevals:
            pif('stopping because max number of user-specified func evals (%d) has been met' % (max_fevals,))

    return obj.free_variables


def sqnorm(a):
    return norm(a)**2

def updateRadius(rho, delta, d_dl, lb=.05, ub=.9):

    if rho > ub:
        delta = max(delta, 2.5*norm(d_dl))
    elif rho < lb:
        delta *= .25
    return delta


def predicted_improvement(d, e, J, sqnorm_e, JTJ, JTe):
    d = col(d)
    e = col(e)
    aa = .5 * sqnorm_e
    bb = JTe.T.dot(d)
    c1 = .5 * d.T
    c2 = JTJ
    c3 = d
    cc = c1.dot(c2.dot(c3))
    result = 2. * (aa - bb + cc)[0,0]
    return sqnorm_e - result


def main():
    pass


if __name__ == '__main__':
    main()

