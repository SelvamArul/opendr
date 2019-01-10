import pygame
from pygame.locals import *
from OpenGL.GL import *
import sys

def init_gl():
    window_size = width, height = (550, 400)
    pygame.init()
    pygame.display.set_mode(window_size, OPENGL | DOUBLEBUF)
    glEnable(GL_TEXTURE_2D)
    glMatrixMode(GL_PROJECTION)
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def load_texture(texture_url):
    tex_id = glGenTextures(1)
    tex = pygame.image.load(texture_url)
    tex_surface = pygame.image.tostring(tex, 'RGBA')
    tex_width, tex_height = tex.get_size()
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_surface)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex_id

if __name__ == "__main__":
    init_gl()
    texture1 = load_texture("texture1.png")
    texture2 = load_texture("texture2.png")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, texture1)
        glBegin(GL_QUADS)
        glTexCoord(0, 0)
        glVertex(50, 50, 0)
        glTexCoord(0, 1)
        glVertex(50, 100, 0)
        glTexCoord(1, 1)
        glVertex(100, 100, 0)
        glTexCoord(1, 0)
        glVertex(100, 50, 0)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

        glBindTexture(GL_TEXTURE_2D, texture2)
        glBegin(GL_QUADS)
        glTexCoord(0, 0)
        glVertex(450, 300, 0)
        glTexCoord(0, 1)
        glVertex(450, 350, 0)
        glTexCoord(1, 1)
        glVertex(500, 350, 0)
        glTexCoord(1, 0)
        glVertex(500, 300, 0)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

        pygame.display.flip()