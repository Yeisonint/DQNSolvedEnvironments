# Libreria para cinematica inversa
import ikpy
from ikpy import plot_utils, geometry_utils
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
# Graficos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from IPython.display import HTML
from PIL import Image
# Operaciones
import numpy as np
import math
import random

# Creamos el objeto del brazo izquierdo
_left_arm_chain = Chain(name='left_arm', links=[
    URDFLink(
        name="shoulder_y",
        translation_vector=[0, 0, 0],
        orientation=[0, 0, 0],
        rotation=[1, 0, 0],
    ),
    URDFLink(
        name="shoulder_x",
        translation_vector=[-10, 0, 5],
        orientation=[0, 1.57, 0],
        rotation=[0, 1, 0],
    ),
    URDFLink(
        name="elbow",
        translation_vector=[25, 0, 0],
        orientation=[0, 0, 1.57],
        rotation=[0, 0, 1],
    ),
    URDFLink(
        name="wrist",
        translation_vector=[22, 0, 0],
        orientation=[0, 0, 0],
        rotation=[0, 1, 0],
    )
])

# Clase que almacenara 
class ArmEnv3D():
    def __init__(self, arm_obj = _left_arm_chain, actions = [-5.0,-0.5,0,0.5,5.0], limits = [(-20.0,100.0),(0.0,130.0),(0.0,130.0)], goal_ratio_percent = 5):
        self._ARM = arm_obj
        self._ACTIONS = [math.radians(a) for a in actions]
        self._ACTION_DIM = len(arm_obj.links)-1
        self._STATE_DIM = 3
        self.STATE = np.zeros(self._STATE_DIM)
        self.TARGET = np.zeros(self._STATE_DIM)
        self.DONE = False
        self.STEPS = 0
        self.MAX_STEPS = 500
        self.LIMITS = [(math.radians(lim[0]),math.radians(lim[1])) for lim in limits]
        self.GOAL_RATIO = [np.around(np.interp(goal_ratio_percent, [0,100], (0,lim[1]-lim[0])), decimals=2) for lim in limits]
    
    # Definimos la funcion pos2ang(pos) y ang2pos(ang)
    pos2ang = lambda self,pos : list(self._ARM.inverse_kinematics(geometry_utils.to_transformation_matrix(pos))[:self._ACTION_DIM])
    ang2pos = lambda self,ang : list(self._ARM.forward_kinematics(list(ang)+[0])[:3, 3])

    # Definimos la funcion isInsLim(pos) y isReachable(pos), esta verifica que una posicion este dentro de los limites
    isInsLim = lambda self,pos : np.all([True if lim[0] <= np.round(self.pos2ang(pos)[i],decimals=2) <= lim[1] else False for i,lim in enumerate(self.LIMITS)])
    isReachable = lambda self,pos: True if np.all(np.round(self.ang2pos(self.pos2ang(pos)),decimals=2) == np.round(pos,decimals=2)) else False

    # asCartesian(rtp) Convierte una posicion esferica (r,theta,phi) en cartesiana (x,y,z)
    # asSpherical(xyz) Convierte una posicion cartesiana (x,y,z) en esferica (r,theta,phi)
    asCartesian = lambda self,rtp: [rtp[0]*math.sin(rtp[1])*math.cos(rtp[2]),rtp[0]*math.sin(rtp[1])*math.sin(rtp[2]),rtp[0]*math.cos(rtp[1])]
    asSpherical = lambda self,xyz: [np.sqrt(np.sum(np.square(xyz))),math.acos(xyz[2]/np.sqrt(np.sum(np.square(xyz)))),math.atan2(xyz[1],xyz[0])]

    # Funcion reset(), devuelve el estado inicial
    def reset(self):
        stateok = False
        st=np.zeros((2,3))
        while not stateok:
            # Crea 2 estados, el inicial y el objetivo
            st[0] = self.ang2pos(np.radians([random.randrange(int(lim[0]), int(lim[1])) for lim in np.degrees(self.LIMITS)]))
            st[1] = self.ang2pos(np.radians([random.randrange(int(lim[0]), int(lim[1])) for lim in np.degrees(self.LIMITS)]))
            stateok = np.all([self.isReachable(st[0]),self.isInsLim(st[0]),self.isReachable(st[1]),self.isInsLim(st[1])])
        st=np.round(st,decimals=2)
        self.STATE = self.asSpherical(st[0])
        self.TARGET = self.asSpherical(st[1])
        return self.STATE

    def encodeAction(self, action , numOfActions, numOfOutputs):
        return [int((action/numOfActions**i)%numOfActions) for i in range(numOfOutputs)]

    # Funcion step(action), 
    def step(self,action):
        action_index = self.encodeAction(action,len(self._ACTIONS),self._ACTION_DIM)
        actList = [self._ACTIONS[i] for i in action_index]
        print(np.degrees(actList))
        tmpState = self.ang2pos(np.array(self.pos2ang(self.asCartesian(self.STATE)))+np.array(actList))
        distance = np.array(self.asCartesian(self.TARGET))-np.array(self.asCartesian(self.STATE))
        reward = -np.sqrt(np.sum(np.square(distance)))
        print(np.degrees(self.pos2ang(tmpState)))
        if self.isInsLim(np.round(tmpState,decimals=2)) and self.isReachable(np.round(tmpState,decimals=2)):
            self.STATE = self.asSpherical(tmpState)
            self.DONE = np.all([abs(d)<=self.GOAL_RATIO[i] for i,d in enumerate(distance)])
            return self.STATE,reward,self.DONE
        else:
            return self.STATE,reward-10,True
        
        


# # Leemos la posicion inicial del brazo y establecemos los grados de libertad
# DOF = len(_left_arm_chain.links)-1
# # Todo se trabaja en radianes, por ende, se pasan los grados a radianes
# ACTIONS = [math.radians(a) for a in [-5.0, -0.5 , 0, 0.5, 5]]
# LIMITS = [(math.radians(lim[0]),math.radians(lim[1])) for lim in [(-20.0,100.0),(0.0,130.0),(0.0,130.0)]]
# # Establecemos el rango de goal en 5% 
# GOAL_RATIO = [np.around(np.interp(5, [0,100], (0,lim[1]-lim[0])), decimals=2) for lim in LIMITS]
# initial_pos = list(np.around(_left_arm_chain.forward_kinematics([0] * len(_left_arm_chain.links))[:3, 3], decimals=2))
# initial_ang = list(_left_arm_chain.inverse_kinematics(_left_arm_chain.forward_kinematics([0] * len(_left_arm_chain.links)))[:DOF])
# current_pos = initial_pos
# current_ang = initial_ang

# # Definimos la funcion pos2ang(pos) y ang2pos(ang)
# pos2ang = lambda pos : list(np.around(_left_arm_chain.inverse_kinematics(geometry_utils.to_transformation_matrix(pos)))[:DOF])
# ang2pos = lambda ang : list(np.around(_left_arm_chain.forward_kinematics(list(ang)+[0])[:3, 3], decimals=2))

# # Definimos la funcion constPos(pos), esta verifica que una posicion este dentro de los limites
# constPos = lambda pos : np.all([True if lim[0] <= pos2ang(pos)[i] <= lim[1] else False for i,lim in enumerate(LIMITS)])

# # Definimos la funcion rewardFunc(distance), la distancia es el punto final menos el actual
# # np.interp(-90, [180,-180], [1,-1], period=360)=-0.5
# rewardFunc = lambda distance : -np.sqrt(np.sum(np.square(distance)))

# def step(action):
