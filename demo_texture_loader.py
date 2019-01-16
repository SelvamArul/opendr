'''template for loading textures
'''
__author__ = 'arul'

import matplotlib.pyplot as plt
import glfw
# import generative_models
from utils import *
import OpenGL.GL as GL
from utils import *
plt.ion()
from OpenGL import contextdata
import sys

#__GL_THREADED_OPTIMIZATIONS

#Main script options:r  

glModes = ['glfw','mesa']
glMode = glModes[0]

np.random.seed(1)

width, height = (128, 128)
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

# Background cube - add to renderer by default.
verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube = getCubeData()

uvCube = np.zeros([verticesCube.shape[0],2])

chCubePosition = ch.Ch([0, 0, 0])
chCubeScale = ch.Ch([10.0])
chCubeAzimuth = ch.Ch([0])
chCubeVCColors = ch.Ch(np.ones_like(vColorsCube) * 1) #white cube
v_transf, vn_transf = transformObject([verticesCube], [normalsCube], chCubeScale, chCubeAzimuth, chCubePosition)


def list_summary(l):
    print ( 'Length ', len(l))
    for i, o in enumerate(l):
        print ('Type of ', i, ' is ', type(o))
        if isinstance(o, np.ndarray) or isinstance(o, ch.Ch):
            print ('shape ', o.shape )

v_scene = [v_transf]
f_list_scene = [[[facesCube]]]
vc_scene = [[chCubeVCColors]]
vn_scene = [vn_transf]
uv_scene = [[uvCube]]
haveTextures_list_scene = [haveTexturesCube]
textures_list_scene = [texturesListCube]


from load_obj import Obj
from PIL import Image

def load_mesh(filename, has_vertex_coloring=False):
    try:
        vertex_data = Obj.open_as_np_array(filename, has_vertex_coloring=has_vertex_coloring)
    except Exception as e:
        print("Warning: could not open {}: {}".format(filename, e))
        return
    return vertex_data

#verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube = getCubeData()

meshes  = ['data/008_pudding_box/textured.obj', 'data/002_master_chef_can/textured.obj', ]
textures = ['data/008_pudding_box/texture_map.png' , 'data/002_master_chef_can/texture_map.png',]
for _i, _m in enumerate(meshes):

    #vert, textUVs, norm, face, color = load_mesh('data/002_master_chef_can/textured.obj')
    v_transf, textUVs, vn_transf, faces, VColors = load_mesh(meshes[_i])

    print ('Min max vertices ', np.min(v_transf), np.max(v_transf))
    print ('shape', v_transf.shape, faces.shape)
    

    texture_image = np.asarray( Image.open(textures[_i]) ).astype(np.float32)
    texture_image /= 255

    textUVs = textUVs[:,0:2]
    haveTexturesObj = [[True]]
    texturesListObj=[[texture_image]]

    # vc_illuminated = computeGlobalAndDirectionalLighting(vn_transf, VColors, chLightAzimuthGT, chLightElevationGT, chLightIntensityGT, chGlobalConstantGT)

    vc_illuminated = ch.Ch(np.ones_like( v_transf ))

    v_scene += [[v_transf]]
    f_list_scene += [[[faces]]]
    vc_scene += [[vc_illuminated]]
    vn_scene += [[vn_transf]]
    uv_scene += [[textUVs]]
    haveTextures_list_scene += [haveTexturesObj]
    textures_list_scene += [texturesListObj]



from libigl_interface import visualize_scene
import copy
visualize_scene( copy.deepcopy(v_scene), copy.deepcopy(f_list_scene))


# camera parameters

#Move camera backwards to match the elevation desired as it looks at origin:
# bottomElev = np.pi/2 - (gtCamElevation + np.arctan(17.5 / focalLenght ))
# ZshiftGT =  ch.Ch(-gtCamHeight * np.tan(bottomElev)) #Move camera backwards to match the elevation desired as it looks at origin.

ZshiftGT =  ch.Ch([-0.2])

gtCamElevation = np.pi/3
gtCamElevation = 0
gtCamHeight = 0. #meters

chLightAzimuthGT = ch.Ch([0])
chLightElevationGT = ch.Ch([np.pi/3])
chLightIntensityGT = ch.Ch([1])
chGlobalConstantGT = ch.Ch([0.5])

chCamElGT = ch.Ch([gtCamElevation])
chCamHeightGT = ch.Ch([gtCamHeight])
focalLenght = 35 ##milimeters
chCamFocalLengthGT = ch.Ch([35/1000])

#COnfigure lighting
lightParamsGT = {'chLightAzimuth': chLightAzimuthGT, 'chLightElevation': chLightElevationGT, 'chLightIntensity': chLightIntensityGT, 'chGlobalConstant':chGlobalConstantGT}

c0 = width/2  #principal point
c1 = height/2  #principal point
a1 = 3.657  #Aspect ratio / mm to pixels
a2 = 3.657  #Aspect ratio / mm to pixels

cameraParamsGT = {'Zshift':ZshiftGT, 'chCamEl': chCamElGT, 'chCamHeight':chCamHeightGT, 'chCamFocalLength':chCamFocalLengthGT, 'a':np.array([a1,a2]), 'width': width, 'height':height, 'c':np.array([c0, c1])}

#Create renderer object
renderer = createRenderer(glMode, cameraParamsGT, v_scene, vc_scene, f_list_scene, vn_scene, uv_scene, haveTextures_list_scene,
                               textures_list_scene, frustum, None)

print ('Reached here')
# Initialize renderer
renderer.overdraw = True
renderer.nsamples = 8
renderer.msaa = True  #Without anti-aliasing optimization often does not work.
renderer.initGL()
renderer.initGLTexture()
renderer.debug = False
winShared = renderer.win

plt.figure()
plt.title('Visualization')
plt.imshow(renderer.r)

rendererGT = ch.Ch(renderer.r.copy()) #Fix the GT position

plt.show(0.1)