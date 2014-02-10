#!/usr/bin/env python
"""Simple pyglet example to manually test event loop integration.

This is meant to run tests manually in ipython as:

In [5]: %gui pyglet

In [6]: %run gui-pyglet.py
"""

import pyglet
from pyglet.gl import *
from math import *
from test3D2 import rotate,rotatex,rotatey,rotatez,part3D

#load model
import util
#model=util.load("./data/test2/face1_3Dfullright_final.model")[0]
#model=util.load("./data/test3/face1_3Dortogonal6_final.model")[0]
#model=util.load("./data/test4/face1_test3Dfull0.model")[0]
#model=util.load("./data/test4/face1_test3Dperfect5.model")[0]
model=util.load("./data/test9/face1_3Dfull0.model")[0]
#model=util.load("./data/test4/face1_test3Donlyfrontal_final.model")[0]
#model=util.load("./data/test5/face1_test3Dnobis5.model")[0]
#model=util.load("./data/test6/face1_3Drot20.model")[0]
#model=util.load("./data/test7/face1_3Dsferefull_final.model")[0]
#model=util.load("init.model")[0]

window = pyglet.window.Window()
glEnable(GL_DEPTH_TEST)
#label = pyglet.text.Label('xxx',font_name='Times New Roman',
#                          font_size=36,
#                          x=window.width//2, y=window.height//2,
#                         anchor_x='center', anchor_y='center')

#render HOGS
import test3D2
import drawHOG
import numpy
txt=[]
for w in model["ww"]:
    aux=drawHOG.drawHOG(w.mask,border=2,val=1)[::-1,:]
    #aux=numpy.random.random(aux.shape)#numpy.ones(aux.shape)
    w.im=(aux/aux.max()*255).astype('uint8')
    ##image = pyglet.image.ImageData(w.im.shape[1],w.im.shape[0],"L",w.im)#
    #image = pyglet.image.load('000379.jpg')
    image = pyglet.image.create(w.im.shape[1],w.im.shape[0])
    image.set_data("L",w.im.shape[1],w.im.tostring())
    w.txt = image.get_texture()
    txt.append(w.txt)
    glEnable(w.txt.target)
    glBindTexture(w.txt.target, w.txt.id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height,
             0, GL_RGBA, GL_UNSIGNED_BYTE,
             image.get_image_data().get_data('RGBA', image.width * 4))


rect_w = float(image.width) / image.height
rect_h = 1

window.data={}
window.data["model"]=model
window.data["posz"]=-20
window.data["posx"]=0
window.data["posy"]=0
#posy=0
#posz=-4
#rx=0


@window.event
def on_close():
    window.close()

@window.event                       
def on_mouse_press(x, y, button, modifiers):
    pass
    #window.clear()
    #print x,y,button, modifiers
    #posx = x
    #rx = y

@window.event                       
def on_mouse_scroll(x, y, scr_x, scr_y):
    window.data["posz"]=window.data["posz"]+scr_y
    #window.clear()
    #print posz#x,y,scr_x,scr_y
    #posz=posz+scr_y
    #posx = x
    #rx = y

@window.event  
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    #window.clear()
    window.data["posx"]=window.data["posx"]-dy
    window.data["posy"]=window.data["posy"]+dx
    #print window.data["posz"]

@window.event                       
def on_mouse_motion(x, y, dx, dy):
    pass
    #window.clear()
    #print x,y,dy,dx
    #posx = x
    #rx = y

@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    #glFrontFace( GL_CCW )
    #glCullFace(GL_FRONT)
    #glEnable(GL_CULL_FACE)
    glTranslatef(0, 0, window.data["posz"])
    glRotatef(window.data["posx"],1,0,0)
    glRotatef(window.data["posy"],0,1,0)
    for idt,t in enumerate(txt):#window.data["model"]["ww"][:1]:
        #glTranslatef(w.x, w.y, window.data["posz"])
        #glBindTexture(w.txt.target, w.txt.id)
        #glBindTexture(texture.target, texture.id)
        m=window.data["model"]["ww"]
        p0=(-rect_w, -rect_h, -m[idt].lz/2)
        p1=( rect_w, -rect_h, -m[idt].lz/2)
        p2=( rect_w,  rect_h, -m[idt].lz/2)
        p3=(-rect_w,  rect_h, -m[idt].lz/2)
        p0=rotatey(p0,-m[idt].ax)
        p1=rotatey(p1,-m[idt].ax)
        p2=rotatey(p2,-m[idt].ax)
        p3=rotatey(p3,-m[idt].ax)
        p0=rotatex(p0,m[idt].ay)
        p1=rotatex(p1,m[idt].ay)
        p2=rotatex(p2,m[idt].ay)
        p3=rotatex(p3,m[idt].ay)
        tr=numpy.array([m[idt].x/2, -m[idt].y/2, -m[idt].z/2])
        p0=p0+tr
        p1=p1+tr
        p2=p2+tr
        p3=p3+tr
        glBindTexture(t.target, t.id)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(p0[0],p0[1],p0[2])
        glTexCoord2f(1.0, 0.0); glVertex3f(p1[0],p1[1],p1[2])
        glTexCoord2f(1.0, 1.0); glVertex3f(p2[0],p2[1],p2[2])
        glTexCoord2f(0.0, 1.0); glVertex3f(p3[0],p3[1],p3[2])
        glEnd()
    #label.draw()

def on_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(65.0, width/float(height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)

window.on_resize = on_resize # we need to replace so can't use @window.event
try:
    from IPython.lib.inputhook import enable_pyglet
    enable_pyglet()
except ImportError:
    pyglet.app.run()


