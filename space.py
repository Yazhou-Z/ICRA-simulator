import numpy as np
import gym
from gym import spaces
class Space:
    def __init__(self):
        # actrually, its "orders"
        self.action_space = spaces.Box(high = 1, low = -1, shape = (8,),dtype = int)
        self.observation_space = spaces.Box(low = -180.0, high = 2000.0, shape = (17, ), dtype = np.float32)
'''
        self.action_space = spaces.Tuple([
        spaces.Discrete(3),
        spaces.Discrete(3),
        spaces.Discrete(3),
        spaces.Discrete(3),
        spaces.Discrete(2),
        spaces.Discrete(2),
        spaces.Discrete(2),
        spaces.Discrete(2)])
'''
'''
Observation
    Type
    Num     Observation     Min     Max
    0:14    car_info        -180.0  800.0
    15      time            0.0     180.0
    16:19   observ          0.0     1.0
    # to be continue:)
    

observ
    Type    (4, 4), float32
'''
