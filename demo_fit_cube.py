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


#__GL_THREADED_OPTIMIZATIONS

#Main script options:r


from iglhelpers import *

glModes = ['glfw','mesa']
glMode = glModes[0]

np.random.seed(1)

width, height = (228, 228)
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

gtCamElevation = np.pi/3
gtCamHeight = 0.5 #meters

chLightAzimuthGT = ch.Ch([0])
chLightElevationGT = ch.Ch([np.pi/3])
chLightIntensityGT = ch.Ch([1])
chGlobalConstantGT = ch.Ch([0.5])

chCamElGT = ch.Ch([gtCamElevation])
chCamHeightGT = ch.Ch([gtCamHeight])
focalLenght = 35 ##milimeters
chCamFocalLengthGT = ch.Ch([35/1000])

#Move camera backwards to match the elevation desired as it looks at origin:
# bottomElev = np.pi/2 - (gtCamElevation + np.arctan(17.5 / focalLenght ))
# ZshiftGT =  ch.Ch(-gtCamHeight * np.tan(bottomElev)) #Move camera backwards to match the elevation desired as it looks at origin.

ZshiftGT =  ch.Ch([-0.2])

# Baackground cube - add to renderer by default.
verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube = getCubeData()

print ('verticesCube', type(verticesCube), verticesCube.shape)
print ('facesCube', type(facesCube), facesCube.shape)
print ('normalsCube', type(normalsCube), normalsCube.shape)
print ('vColorsCube', type(vColorsCube), vColorsCube.shape)
print ('texturesListCube', type(texturesListCube), len(texturesListCube), type(texturesListCube[0]), texturesListCube)
print ('haveTexturesCube', type(haveTexturesCube), len(haveTexturesCube), type(haveTexturesCube[0]), haveTexturesCube)


uvCube = np.zeros([verticesCube.shape[0],2])

chCubePosition = ch.Ch([0, 0, 0])
chCubeScale = ch.Ch([10.])
chCubeAzimuth = ch.Ch([0])
chCubeVCColors = ch.Ch(np.ones_like(vColorsCube) * 1.) #white cube
C_np_list = []
C_np_list.append(np.array(chCubeVCColors))
v_transf, vn_transf = transformObject([verticesCube], [normalsCube], chCubeScale, chCubeAzimuth, chCubePosition)

v_scene = [v_transf]
f_list_scene = [[[facesCube]]]
vc_scene = [[chCubeVCColors]]
vn_scene = [vn_transf]
uv_scene = [[uvCube]]
haveTextures_list_scene = [haveTexturesCube]
textures_list_scene = [texturesListCube]



#v_scene = []
#f_list_scene = []
#vc_scene = []
#vn_scene = []
#uv_scene = []
#haveTextures_list_scene = []
#textures_list_scene = []

#Example object 1: forgroudn cube
verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube = getCubeData()
uvCube = np.zeros([verticesCube.shape[0],2])

chCubeVCColors = ch.Ch(np.ones_like(vColorsCube) * 0.5) #Gray cube

C_np_list.append(np.array(chCubeVCColors))

chPositionGT = ch.Ch([0.2, 0.0, 0.5])
# chPositionGT = ch.Ch([-0.23, 0.36, 0.])
chScaleGT = ch.Ch([0.5, 0.5, 0.5])
chColorGT = ch.Ch([1.0, 1.0, 1.0])
chAzimuthGT = ch.Ch([0.0])

objectParamsGT = {'chPosition':chPositionGT, 'chScale':chScaleGT, 'chColor':chColorGT, 'chAzimuth':chAzimuthGT}

v_transf, vn_transf = transformObject([verticesCube], [normalsCube], chScaleGT, chAzimuthGT, chPositionGT)

import ipdb
ipdb.set_trace()

vc_illuminated = computeGlobalAndDirectionalLighting(vn_transf, [chCubeVCColors], chLightAzimuthGT, chLightElevationGT, chLightIntensityGT, chGlobalConstantGT)



v_scene += [v_transf]
f_list_scene += [[[facesCube]]]
vc_scene += [vc_illuminated]
vn_scene += [vn_transf]
uv_scene += [[uvCube]]
haveTextures_list_scene += [haveTexturesCube]
textures_list_scene += [texturesListCube]

v_scene_np_list = []
for _i in v_scene:
    v_scene_np_list.append(np.array(_i[0]))
num_vertices_in_first = v_scene_np_list[0].shape[0]
print ('num_vertices_in_first', num_vertices_in_first)
v_scene_np = np.vstack(v_scene_np_list)

f_np_list = []
for _i in f_list_scene:
    print (type(_i), len(_i), type(_i[0]), )
    f_np_list.append(_i[0][0])
num_vertices_in_first = v_scene_np_list[0].shape[0]
num_faces_in_first = f_np_list[0].shape[0]
print ('num_vertices_in_first', num_vertices_in_first)
v_scene_np = np.vstack(v_scene_np_list)
f_np  = np.vstack(f_np_list)

C_np = np.vstack(C_np_list)


#print (f_np)
f_np[num_faces_in_first:] += num_vertices_in_first
#print (v_scene_np.shape)
#print (f_np)
V = p2e(v_scene_np)
F = p2e(f_np)
C = p2e(C_np)
viewer = igl.glfw.Viewer()
bg_np =np.array([[1, 1, 1, 1]]).astype(np.float64).T
bg = p2e(bg_np)

viewer.core.background_color = bg
viewer.data().set_mesh(V, F)
viewer.data().set_colors(C)
#viewer.launch()

#sys.exit('Alles Gut!!!')
coord_transformation = np.eye(4).astype(np.float64)

axis_V_np =np.ascontiguousarray(coord_transformation[:,:3]).astype(np.float64)
axis_V_np = np.roll(axis_V_np,1, axis=0)

flipXRotation = np.array([[1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0., 0.0],
            [0.0, 0., -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
if True:
    flipZYRotation = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0, 1.0, 0.0],
                                    [0.0, -1.0, 0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])

    t = np.ones((4, 1)).astype(np.float64)
    axis_V_np =np.append(axis_V_np, t, 1)

    print ('axis_V_np \n', axis_V_np.T)

    #axis_V_np = axis_V_np.T

    axis_V_np = (flipZYRotation @ axis_V_np.T).T
    axis_V_np = (flipXRotation @ flipZYRotation @ axis_V_np.T).T

    print ('axis_V_np \n', axis_V_np)

else:
    origin = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1] ]).T
    axis_V_np = np.ascontiguousarray(origin).astype(np.float64)

    tf_world_camera = Matrix44.look_at(
            (0.0, 3.0, .85),
            (0.0, 2.43, 0),
            (0.0, 0.6, -1.0),
        )
    print ('axis_V_np \n', axis_V_np)
    axis_V_np = (flipXRotation @ tf_world_camera @ origin).T
    print ('axis_V_np \n', axis_V_np)


axis_E_np = np.array([[0,1],[0,2],[0,3]]).astype(np.int32)
axis_C_np = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]).astype(np.float64)
axis_V = p2e( np.ascontiguousarray(axis_V_np[:,:3]).astype(np.float64) )
axis_E = p2e(axis_E_np)
axis_C = p2e(axis_C_np)
_C_np = np.array(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])).astype(np.float64)
_C = p2e(_C_np)

viewer.data().set_points(axis_V, _C)
viewer.data().set_edges(axis_V, axis_E, axis_C)

#viewer.launch()

#sys.exit('Alles Gut!!!')
##COnfigure lighting
lightParamsGT = {'chLightAzimuth': chLightAzimuthGT, 'chLightElevation': chLightElevationGT, 'chLightIntensity': chLightIntensityGT, 'chGlobalConstant':chGlobalConstantGT}

c0 = width/2  #principal point
c1 = height/2  #principal point
a1 = 3.657  #Aspect ratio / mm to pixels
a2 = 3.657  #Aspect ratio / mm to pixels

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

#sys.exit('Alles Gut')
print ('-------------------****-----------------------')
for k, v in cameraParamsGT.items():
    print ('{}   --> {}'.format(k,v))

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

#Vary cube scale:
chScaleGT[0] = 0.05
chScaleGT[1] = 0.05

#chPositionGT[0] = 0.01
#chPositionGT[1] = 0.01


#plt.figure()
#plt.title('Init object')
_x0 = renderer.r
#plt.imshow(_x0)

variances = ch.Ch([0.1])
#print ('variances', variances)


#negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances, useMask=True)) / numPixels
difference = renderer - rendererGT

print ('=======================>','difference', type(difference))

gpModel = gaussian_pyramid(difference)

#sys.exit('Alles Gut')
#plt.title('Init object')


global iter
iter = 0
def cb(_):
    pass

global method
methods = ['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SGDMom'] #Nelder-mead is the finite difference simplex method
method = -1

options = {'disp': True, 'maxiter': 100000}
options = {'disp': True, 'maxiter': 1000, 'lr':2e-3, 'momentum' : 0.4, 'decay' :0.9, 'tol' : 1e-7}
ch.minimize({'raw': gpModel}, bounds=None, method=methods[method], x0=[chScaleGT] , callback=cb, options=options)

#plt.figure()
#plt.title('Fitted object')
_xn = renderer.r
#plt.imshow(_xn)


#plt.figure()
#plt.title('gt - init')
_gt_x0 = _gt - _x0
#plt.imshow(_gt_x0)

plt.figure()
plt.title('gt - Gaussian Pyramid')
_gt_xn = _gt - _xn
plt.imshow(_gt_xn)


##plt.show(0.1)
print ('chScaleGT', chScaleGT)
print ('chPositionGT', chPositionGT)




import sys
sys.exit()










#########################################################
#########################################################
## Rerun the same code but with NLL error
#Vary cube scale:
chScaleGT[0] = 0.05
chScaleGT[1] = 0.05

#chPositionGT[0] = 0.01
#chPositionGT[1] = 0.01


plt.figure()
plt.title('Init object')
_x0 = renderer.r
plt.imshow(_x0)

variances = ch.Ch([0.1])
#print ('variances', variances)


negLikModel = -ch.sum(generative_models.LogGaussianModel(renderer=renderer, groundtruth=rendererGT, variances=variances, useMask=True)) / numPixels

print ('=======================>','difference', type(difference))


#sys.exit('Alles Gut')
#plt.title('Init object')


iter = 0
def cb(_):
    pass

methods = ['dogleg', 'minimize', 'BFGS', 'L-BFGS-B', 'Nelder-Mead'] #Nelder-mead is the finite difference simplex method
method = 3

options = {'disp': True, 'maxiter': 100000}
ch.minimize({'raw': negLikModel}, bounds=None, method=methods[method], x0=[chScaleGT] , callback=cb, options=options)

#plt.figure()
#plt.title('Fitted object')
_xn = renderer.r
#plt.imshow(_xn)


#plt.figure()
#plt.title('gt - init')
_gt_x0 = _gt - _x0
#plt.imshow(_gt_x0)

plt.figure()
plt.title('gt - NLL')
_gt_xn = _gt - _xn
plt.imshow(_gt_xn)


plt.show(0.1)
print ('chScaleGT', chScaleGT)
print ('chPositionGT', chPositionGT)






#Clean up.
renderer.makeCurrentContext()
renderer.clear()
contextdata.cleanupContext(contextdata.getContext())
# glfw.destroy_window(renderer.win)
del renderer
