__author__ = 'pol'

import matplotlib.pyplot as plt
import glfw
import generative_models
from utils import *
import OpenGL.GL as GL
from utils import *
plt.ion()
from OpenGL import contextdata
import sys
from opendr.filters import gaussian_pyramid

from pyrr import Matrix44, Matrix33, matrix44, matrix33, vector3, vector4

import load_obj

import chumpy as cp
#__GL_THREADED_OPTIMIZATIONS

#Main script options:r


from iglhelpers import *

glModes = ['glfw','mesa']
glMode = glModes[0]

def load_mesh(filename, has_vertex_coloring=False):
        try:
            vertex_data = load_obj.Obj.open(filename, has_vertex_coloring=has_vertex_coloring)
        except Exception as e:
            print("Warning: could not open {}: {}".format(filename, e))
            return
        return vertex_data

np.random.seed(1)

width, height = (224, 224)
numPixels = width*height
shapeIm = [width, height,3]
win = -1
clip_start = 0.01
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}

if glMode == 'glfw':
    #Initialize base GLFW context for the Demo and to share context among all renderers.
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS,32)
    glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
    win = glfw.create_window(width, height, "Demo",  None, None)
    glfw.make_context_current(win)

else:
    from OpenGL.raw.osmesa._types import *
    from OpenGL.raw.osmesa import mesa

winShared = None


#chLightAzimuthGT = ch.Ch([0])
#chLightElevationGT = ch.Ch([np.pi * 1.5])
#chLightIntensityGT = ch.Ch([1])
#chGlobalConstantGT = ch.Ch([0.0])



v_scene = []
f_list_scene = []
vc_scene = []
vn_scene = []
uv_scene = []
haveTextures_list_scene = []
textures_list_scene = []

HEIGHT_OF_TABLE=2.4613
def add_mesh(mesh_file, has_vertex_coloring=False, scaleTF=0, azimuthTF=0, positionTF=0, coloring=0):
    global v_scene
    global f_list_scene
    global vc_scene
    global vn_scene
    global uv_scene
    global haveTextures_list_scene
    global textures_list_scene
    static_mesh = load_mesh(mesh_file, has_vertex_coloring=has_vertex_coloring)

    #op_file = open('py_vertices.txt', 'w')

    #for i in static_mesh.vert:
        #j = [str(x) for x in i]
        #print (' '.join(j), file=op_file)
    #op_file.close()

    _v_transf, _vn_transf = np.array(static_mesh.vert, dtype=np.float32) , np.array(static_mesh.norm, dtype=np.float32)
    # transform the object
    if scaleTF or azimuthTF or positionTF:
        _v_transf, _vn_transf = transformObject([_v_transf], [_vn_transf], scaleTF, azimuthTF, positionTF)
        _v_transf, _vn_transf = _v_transf[0], _vn_transf[0]

    _v_transf = [_v_transf]
    _vn_transf = [_vn_transf]

    np_Colors = np.array(static_mesh.color, dtype=np.float32)
    if coloring:
        for _i in range(3):
            np_Colors[:,_i] = coloring[_i]
    print ('np_Colors', np.amin(np_Colors), np.amax(np_Colors))
    _uvCube = np.zeros([_v_transf[0].shape[0],2])
    _texturesListCube = [[None]]
    _haveTexturesCube = [[False]]

    #_vc_illuminated = computeGlobalAndDirectionalLighting(_vn_transf,
                                                          #np_Colors,
                                                          #chLightAzimuthGT,
                                                          #chLightElevationGT,
                                                          #chLightIntensityGT,
                                                          #chGlobalConstantGT)
    #import ipdb
    #ipdb.set_trace()

    
    
    _vc_illuminated = computeGlobalAndPointLighting(
              v = _v_transf,
              vn = _vn_transf,
              vc = [ cp.array(np_Colors) ],
              light_pos = np.array( [0,0, HEIGHT_OF_TABLE+1.5] ),
              globalConstant = np.array( [0, 0, 0] ),
              light_color = np.array( [1, 1, 1] ),
        )
    
    # uncomment to disable lighting computation
    # _vc_illuminated = [ cp.array(np_Colors) ]
    
    #computeGlobalAndPointLighting(v, vn, vc, light_pos, globalConstant, light_color):

    def format_faces(mesh_faces):
        '''
        in: mesh_faces: faces as read by load_obj
        usual faces are in the format: (vertex_index1, vertex_index2, vertex_index3)
        but load_obj loads them as vertex_index1,texture_index1, normal_index1
        This format gives more flexibility but we don't need this
        We need to faces format to match the opendr
        '''
        opendr_faces = []
        for i in range(0, len(mesh_faces), 3):
            opendr_faces.append([ mesh_faces[i][0], mesh_faces[i+1][0], mesh_faces[i+2][0] ])
        return opendr_faces



    opendr_faces = format_faces(static_mesh.face)

    opendr_faces_np =  np.array(opendr_faces, dtype=np.int64)


    # fix for faces indexing
    # This strange convention with .obj files
    # File has vertices 1 indexed
    # Application expects 0 indexing
    # Most readers does this silently.
    opendr_faces_np -= 1

    if len(v_scene) == 0:
        v_scene = [_v_transf]
        f_list_scene = [[[ opendr_faces_np ]]]
        vc_scene = [_vc_illuminated]
        vn_scene = [_vn_transf]
        uv_scene = [[_uvCube]]
        haveTextures_list_scene = [_haveTexturesCube]
        textures_list_scene = [_texturesListCube]
    else:
        v_scene += [_v_transf]
        f_list_scene += [[[ opendr_faces_np ]]]
        vc_scene += [_vc_illuminated]
        vn_scene += [_vn_transf]
        uv_scene += [[_uvCube]]
        haveTextures_list_scene += [_haveTexturesCube]
        textures_list_scene += [_texturesListCube]


HEIGHT_OF_TABLE=2.4613
scaleTF = ch.Ch([2, 2, 2])
azimuthTF = ch.Ch([ 2.0 * np.pi * (0 / 360.0) ])
positionTF = ch.Ch([0., 0., HEIGHT_OF_TABLE,])

print ('Adding drill')
add_mesh('data/opendr_models/drill_opendr_frame.obj',
         has_vertex_coloring=True,
         scaleTF=scaleTF,
         azimuthTF=azimuthTF,
         positionTF=positionTF,
         )
print ('Adding table')
add_mesh('data/opendr_models/table_opendr_frame.obj',
         has_vertex_coloring=True,
         coloring=[.8, 0.8, 0.8]
         )



v_scene_np_list = []
for _i in v_scene:
    v_scene_np_list.append(np.array(_i[0]))
num_vertices_in_first = v_scene_np_list[0].shape[0]
v_scene_np = np.vstack(v_scene_np_list)

f_np_list = []
for _i in f_list_scene:
    print (type(_i), len(_i), type(_i[0]), )
    f_np_list.append(_i[0][0])
num_faces_in_first = f_np_list[0].shape[0]
f_np  = np.vstack(f_np_list)

C_np_list = []
for _i in vc_scene:
    C_np_list.append(np.array(_i[0]))
C_np = np.vstack(C_np_list)


f_np[num_faces_in_first:] += num_vertices_in_first

#remove nans from the colors
C_np[np.isnan(C_np)] = 0
C_np[C_np>1] = 1.


#C_np[:,0] = 1.
#C_np[:,1] = 0.
#C_np[:,2] = 0.
V = p2e(v_scene_np)
F = p2e(f_np)
C = p2e(C_np)

#import ipdb
#ipdb.set_trace()


#print ('creating viewer')
#viewer = igl.glfw.Viewer()
bg_np =np.array([[1, 1, 1, 1]]).astype(np.float64).T
bg = p2e(bg_np)

# viewer callback
def key_pressed(viewer, key, modifier):
    print("Key: ", chr(key))
    print (viewer.core.proj)
    print (viewer.core.view)
    print('-----------------')

#viewer.callback_key_pressed = key_pressed

#viewer.core.background_color = bg
#viewer.data().set_mesh(V, F)
#viewer.data().set_colors(C)
#viewer.launch()


coord_transformation = np.eye(4).astype(np.float64)

axis_V_np =np.ascontiguousarray(coord_transformation[:,:3]).astype(np.float64)
axis_V_np = np.roll(axis_V_np,1, axis=0)

t = np.ones((4, 1)).astype(np.float64)
axis_V_np =np.append(axis_V_np, t, 1)


axis_E_np = np.array([[0,1],[0,2],[0,3]]).astype(np.int32)
axis_C_np = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]).astype(np.float64)
axis_V = p2e( np.ascontiguousarray(axis_V_np[:,:3]).astype(np.float64) )

#print ('axis_V')
#print (axis_V)

axis_E = p2e(axis_E_np)
axis_C = p2e(axis_C_np)
_C_np = np.array(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])).astype(np.float64)
_C = p2e(_C_np)

#viewer.data().set_points(axis_V, _C)
#viewer.data().set_edges(axis_V, axis_E, axis_C)

#viewer.launch()

#import ipdb
#ipdb.set_trace()



#sys.exit('Alles Gut!!!')

###########################################################################
##*************************************************************************
###########################################################################



















##COnfigure lighting
#lightParamsGT = {'chLightAzimuth': chLightAzimuthGT, 'chLightElevation': chLightElevationGT, 'chLightIntensity': chLightIntensityGT, 'chGlobalConstant':chGlobalConstantGT}

c0 = width/2  #principal point
c1 = height/2  #principal point
a1 = 3.657  #Aspect ratio / mm to pixels
a2 = 3.657  #Aspect ratio / mm to pixels


gtCamElevation = np.pi/4
gtCamHeight = 4 #meters

ZshiftGT =  ch.Ch([-1.25])
chCamElGT = ch.Ch([gtCamElevation])
chCamHeightGT = ch.Ch([gtCamHeight])
focalLenght = 35 ##milimeters
chCamFocalLengthGT = ch.Ch([35/1000])


cameraParamsGT = {'Zshift':ZshiftGT, 'chCamEl': chCamElGT, 'chCamHeight':chCamHeightGT, 'chCamFocalLength':chCamFocalLengthGT, 'a':np.array([a1,a2]), 'width': width, 'height':height, 'c':np.array([c0, c1])}

#Create renderer object

#print ('v_scene', v_scene)
#print ('vc_scene', vc_scene)
#print ('f_list_scene', f_list_scene)
#print ('vn_scene', vn_scene)
#print ('uv_scene', uv_scene)
#print ('haveTextures_list_scene', haveTextures_list_scene)
#print ('textures_list_scene', textures_list_scene)
#print ('frustum', frustum)

#print ('-------------------****-----------------------')
#for k, v in cameraParamsGT.items():
    #print ('{}   --> {}'.format(k,v))

renderer = createRenderer(glMode, cameraParamsGT, v_scene, vc_scene, f_list_scene, vn_scene, uv_scene, haveTextures_list_scene,
                               textures_list_scene, frustum, None)
#sys.exit('Alles Gut')
# Initialize renderer
renderer.overdraw = True
renderer.nsamples = 8
renderer.msaa = True  #Without anti-aliasing optimization often does not work.
renderer.initGL()
renderer.initGLTexture()
renderer.debug = False
winShared = renderer.win

#plt.figure()
#plt.title('GT object')
_gt = renderer.r
#plt.imshow(_gt)

rendererGT = ch.Ch(renderer.r.copy()) #Fix the GT position


#positionTF = ch.Ch([0.1, 0.1, HEIGHT_OF_TABLE,])

positionTF[0] = 0.03
positionTF[1] = 0.03

# azimuthTF[0] = 2.0 * np.pi * (2 / 360.0) 

#Vary cube scale:
#chPositionGT[0] = 0.01
#chPositionGT[1] = 0.01


#plt.figure()
#plt.title('Init object')
_x0 = renderer.r
#plt.imshow(_x0)
#plt.show(0.1)

#sys.exit('Alles Gut!!!')

difference = renderer - rendererGT

print ('=======================>','difference', type(difference))

gpModel = gaussian_pyramid(difference).sum()

#sys.exit('Alles Gut')
#plt.title('Init object')


global iter
iter = 0
def cb(_):
    pass

global method
methods = ['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SGDMom']
method = 1

options = {'disp': True, 'maxiter': 1000, 'lr':2e-4, 'momentum' : 0.4, 'decay' :0.9, 'tol' : 1e-7}

ch.minimize({'raw': gpModel}, bounds=None, method=methods[method], x0=[positionTF[0], positionTF[1]] , callback=cb, options=options)

plt.figure()
plt.title('Fitted object')
_xn = renderer.r
plt.imshow(_xn)


plt.figure()
plt.title('gt - init')
_gt_x0 = _gt - _x0
plt.imshow(_gt_x0)

plt.figure()
plt.title('gt - _xn')
_gt_xn = _gt - _xn
plt.imshow(_gt_xn)

plt.show(0.1)

print ('Optimized values')
print (positionTF[0], positionTF[1])
print (azimuthTF)
print ('Completed ')

#Clean up.
renderer.makeCurrentContext()
renderer.clear()
contextdata.cleanupContext(contextdata.getContext())
# glfw.destroy_window(renderer.win)
del renderer
