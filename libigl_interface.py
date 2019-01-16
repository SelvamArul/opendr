'''
Interface for libigl viewer
'''
__author__ = 'arul'

import numpy as np
from iglhelpers import *
from typing import Callable, Iterable, Union, Optional, List
import random

def visualize_scene(vertices_list: List[np.ndarray], faces_list: List[np.ndarray], colors_list : List[np.ndarray] = [] ):
    
    C_np_list = []
    if len(colors_list) != 0:
        for _i in colors_list:
            C_np_list.append(np.array(_i[0]))
        C_np = np.vstack(C_np_list)
        #remove nans from the colors
        C_np[np.isnan(C_np)] = 0
        C_np[C_np>1] = 1
    

    
    v_scene_np_list = []
    num_vertices_end_list = [0]
    for _x, _i in enumerate(vertices_list):
        print ('vert', _i[0].shape, _i[0].dtype )
        print ( _i[0][:5] )
        v_scene_np_list.append(np.array(_i[0]))
        num_vertices_end_list.append(_i[0].shape[0] + num_vertices_end_list[-1] )

        # random color for vertices if no vertex coloring is provided
        if len(colors_list) == 0:
            _c = np.zeros_like(_i[0])
            for _t in range(3):
                _b = random.getrandbits(1)
                _c[:_t] = _b
                print (_b)
            if _x == 0:
                _c = np.ones_like(_i[0])
            C_np_list.append(_c)
    v_scene_np = np.vstack(v_scene_np_list)
    C_np = np.vstack( C_np_list )

    f_np_list = []
    for _x, _i in enumerate( faces_list ):
        print (type(_i), len(_i), _i[0][0].shape )
        _i[0][0] += num_vertices_end_list[_x]
        f_np_list.append(_i[0][0])
        print (np.min(_i[0][0]), np.max(_i[0][0]))
    f_np  = np.vstack(f_np_list)
    

    V = p2e(v_scene_np)
    F = p2e(f_np)    
    #C = p2e(C_np)

    viewer = igl.glfw.Viewer()
    bg_np =np.array([[1, 1, 1, 1]]).astype(np.float64).T
    bg = p2e(bg_np)

    def key_pressed(viewer, key, modifier):
        print (viewer.core.proj)
        print (viewer.core.view)

    viewer.callback_key_pressed = key_pressed

    viewer.core.background_color = bg
    viewer.data().set_mesh(V, F)
    #if len(C_np_list) != 0:
    #    viewer.data().set_colors(C)
    viewer.launch()
