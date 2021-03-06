# -*- coding: utf-8 -*-
'''
kernal v1.0
'''
import numpy as np
from numpy.core.arrayprint import dtype_is_implied
import pygame
import random 
import gym
from gym import spaces

class bullet(object):
    def __init__(self, center, angle, speed, owner):
        self.center = center.copy()
        self.speed = speed
        self.angle = angle
        self.owner = owner

class record(object):
    def __init__(self, time, cars, compet_info, detect, vision, bullets):
        self.time = time
        self.cars = cars
        self.compet_info = compet_info
        self.detect = detect
        self.vision = vision
        self.bullets = bullets

class g_map(object):
    def __init__(self, length, width, areas, barriers, special_area):
        self.length = length
        self.width = width
        self.areas = areas
        self.barriers = barriers
        self.special_area = special_area

class record_player(object):
    def __init__(self):
        self.map_length = 800
        self.map_width = 500
        pygame.init()
        self.screen = pygame.display.set_mode((self.map_length, self.map_width)) # creates window
        pygame.display.set_caption('RM AI Challenge Simulator')
        self.gray = (180, 180, 180)
        self.red = (190, 20, 20)
        self.blue = (10, 125, 181)
        self.areas = np.array([[[708.0, 808.0, 348.0, 448.0],
                                [708.0, 808.0, 0.0, 100.0]],
                                [[0.0, 100.0, 0.0, 100.0],
                                [0.0, 100.0, 348.0, 448.0]]], dtype='float32')
        self.special_area = np.array([[23.0, 77.0, 145.0, 193.0],
                                [731.0, 785.0, 255.0, 303.0], 
                                [377.0, 431.0, 20.5, 68.5], 
                                [377.0, 431.0, 379.5, 427.5],
                                [163.0, 217.0, 259.0, 307.0], 
                                [591.0, 645.0, 141.0, 189.0]], dtype='float32')
        self.barriers = np.array([[150.0, 230.0, 214.0, 234.0],
                                  [578.0, 658.0, 214.0, 234.0],
                                  [0.0, 100.0, 100.0, 120.0],
                                  [708.0, 808.0, 328.0, 348.0],
                                  [354.0, 454.0, 93.5, 113.5],
                                  [354.0, 454.0, 334.5, 354.5],
                                  [150.0, 170.0, 348.0, 448.0],
                                  [638.0, 658.0, 0.0, 100.0],
                                  [386.3, 421.7, 206.3, 241.7]], dtype='float32') # barrier_horizcontal: B2, B8, B1, B9, B4, B6; barrier_vertical: B3, B7
        # load barriers imgs #
        self.barriers_img = []
        self.barriers_rect = []
        barrier_horizontal_tall = pygame.image.load('./imgs/barrier_horizontal.png')
        barrier_horizontal_tall = pygame.transform.scale(barrier_horizontal_tall, (100,20))
        barrier_horizontal_short = pygame.image.load('./imgs/barrier_horizontal.png')
        barrier_horizontal_short = pygame.transform.scale(barrier_horizontal_short, (80,20)) 
        barrier_vertical = pygame.image.load('./imgs/barrier_vertical.png')
        barrier_vertical = pygame.transform.scale(barrier_vertical, (20,100))
        barrier_small = pygame.image.load('./imgs/barrier_small.png')
        barrier_small = pygame.transform.scale(barrier_small, (25,25))
        barrier_small = pygame.transform.rotate(barrier_small, 45)
        for i in range(self.barriers.shape[0]+1):
            if i < 2:
                self.barriers_img.append(barrier_horizontal_short)
            elif i < 6:
                self.barriers_img.append(barrier_horizontal_tall)
            elif i < 8:
                self.barriers_img.append(barrier_vertical)
            else:
                self.barriers_img.append(barrier_small)
            self.barriers_rect.append(self.barriers_img[-1].get_rect())
            if i < 8:
                self.barriers_rect[-1].center = [self.barriers[i][0:2].mean(), self.barriers[i][2:4].mean()]
            else:
                self.barriers_rect[-1].center = [404,224]
        # load start imgs
        self.areas_img = []
        self.areas_rect = []
        for oi, o in enumerate(['red', 'blue']):
            for ti, t in enumerate(['start', 'start']):
                startpicture = pygame.image.load('./imgs/area_{}_{}.png'.format(t, o))                    
                startpicture = pygame.transform.scale(startpicture, (100,100))
                self.areas_img.append(startpicture)
                self.areas_rect.append(self.areas_img[-1].get_rect())
                self.areas_rect[-1].center = [self.areas[oi, ti][0:2].mean(), self.areas[oi, ti][2:4].mean()]
        # load special_area imgs
        picture = pygame.image.load('./imgs/special_area.png')
        picture = pygame.transform.scale(picture, (54,48))            
        self.special_area_img = [picture for i in range(6)]
        self.special_area_rect = []
        for i in range(6):
            self.special_area_rect.append(self.special_area_img[i].get_rect())
            self.special_area_rect[-1].center = [self.special_area[i][0:2].mean(), self.special_area[i][2:4].mean()] 

        self.chassis_img = pygame.image.load('./imgs/chassis_g.png')
        self.gimbal_img = pygame.image.load('./imgs/gimbal_g.png')
        self.bullet_img = pygame.image.load('./imgs/bullet_s.png')
        self.info_bar_img = pygame.image.load('./imgs/info_bar.png')
        self.bullet_rect = self.bullet_img.get_rect()
        self.info_bar_rect = self.info_bar_img.get_rect()
        self.info_bar_rect.center = [200, self.map_width/2]
        pygame.font.init()
        self.font = pygame.font.SysFont('info', 20)
        self.clock = pygame.time.Clock()

    def play(self, file):
        self.memory = np.load(file)
        i = 0
        stop = False
        flag = 0
        while True:
            self.time = self.memory[i].time
            self.cars = self.memory[i].cars
            self.car_num = len(self.cars)
            #self.compet_info = self.memory[i].compet_info
            self.detect = self.memory[i].detect
            self.vision = self.memory[i].vision
            self.bullets = self.memory[i].bullets
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_TAB]: self.dev = True
            else: self.dev = False
            self.one_epoch()
            if pressed[pygame.K_SPACE] and not flag:
                flag = 50
                stop = not stop
            if flag > 0: flag -= 1
            if pressed[pygame.K_LEFT] and i > 10: i -= 10
            if pressed[pygame.K_RIGHT] and i < len(self.memory)-10: i += 10
            if i < len(self.memory)-1 and not stop: i += 1
            self.clock.tick(200)

    def one_epoch(self):
        self.screen.fill(self.gray) # fill it with grey
        for i in range(len(self.barriers_rect)):
            self.screen.blit(self.barriers_img[i], self.barriers_rect[i])
        for i in range(len(self.areas_rect)):
            self.screen.blit(self.areas_img[i], self.areas_rect[i])
        for i in range(len(self.bullets)):
            self.bullet_rect.center = self.bullets[i].center
            self.screen.blit(self.bullet_img, self.bullet_rect)
        for n in range(self.car_num):
            chassis_rotate = pygame.transform.rotate(self.chassis_img, -self.cars[n, 3]-90)
            gimbal_rotate = pygame.transform.rotate(self.gimbal_img, -self.cars[n, 4]-self.cars[n, 3]-90)
            chassis_rotate_rect = chassis_rotate.get_rect()
            gimbal_rotate_rect = gimbal_rotate.get_rect()
            chassis_rotate_rect.center = self.cars[n, 1:3]
            gimbal_rotate_rect.center = self.cars[n, 1:3]
            self.screen.blit(chassis_rotate, chassis_rotate_rect)
            self.screen.blit(gimbal_rotate, gimbal_rotate_rect)
            select = np.where((self.vision[n] == 1))[0]+1
            select2 = np.where((self.detect[n] == 1))[0]+1
            info = self.font.render('{} | {}: {} {}'.format(int(self.cars[n, 6]), n+1, select, select2), True, self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3]+[-20, -60])
            info = self.font.render('{} {}'.format(int(self.cars[n, 10]), int(self.cars[n, 5])), True, self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3]+[-20, -45])
        self.screen.blit(self.head_img[0], self.head_rect[0])
        self.screen.blit(self.head_img[1], self.head_rect[1])
        info = self.font.render('time: {}'.format(self.time), False, (0, 0, 0))
        self.screen.blit(info, (8, 8))
        if self.dev:
            for n in range(self.car_num):
                wheels = self.check_points_wheel(self.cars[n])
                for w in wheels:
                    pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, w.astype(int), 3)
                armors = self.check_points_armor(self.cars[n])
                for a in armors:
                    pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, a.astype(int), 3)
            self.screen.blit(self.info_bar_img, self.info_bar_rect)
            for n in range(self.car_num):
                tags = ['owner', 'x', 'y', 'angle', 'yaw', 'heat', 'hp', 'freeze_time', 'is_supply', 
                        'can_shoot', 'bullet', 'stay_time', 'wheel_hit', 'armor_hit', 'car_hit']
                info = self.font.render('car {}'.format(n), False, (0, 0, 0))
                self.screen.blit(info, (8+n*100, 100))
                for i in range(self.cars[n].size):
                    info = self.font.render('{}: {}'.format(tags[i], int(self.cars[n, i])), False, (0, 0, 0))
                    self.screen.blit(info, (8+n*100, 117+i*17))
            '''
            info = self.font.render('red   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[0, 0], \
                                    self.compet_info[0, 1], self.compet_info[0, 3]), False, (0, 0, 0))
            self.screen.blit(info, (8, 372))
            info = self.font.render('blue   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[1, 0], \
                                self.compet_info[1, 1], self.compet_info[1, 3]), False, (0, 0, 0))
            self.screen.blit(info, (8, 389))
            '''
        pygame.display.flip()

    def check_points_wheel(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-22.5, -29], [22.5, -29], 
                       [-22.5, -14], [22.5, -14], 
                       [-22.5, 14], [22.5, 14],
                       [-22.5, 29], [22.5, 29]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]

    def check_points_armor(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-6.5, -30], [6.5, -30], 
             [-18.5,  -7], [18.5,  -7],
             [-18.5,  0], [18.5,  0],
             [-18.5,  6], [18.5,  6],
             [-6.5, 30], [6.5, 30]])
        return [np.matmul(x, rotate_matrix) + car[1:3] for x in xs]

class Move_Shoot:
    def __init__(self, area, time, activation):
       self.area = area
       self.time = time
       self.activation = activation 

class RefereeSystem:
    move = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    shoot = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    red_hp = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    blue_hp = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    red_bullet = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    blue_bullet = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    def __init__(self, special_area, time, cars):
        self.special_area = special_area
        self.red_blue_bonus_hp_bullet_area = np.zeros([4,4], dtype='float32') # red_hp, blue_hp, red_bullet, blue_bullet
        self.red_blue_bonus_hp_bullet_sctivation = np.zeros([1,4], dtype=int)
        self.time = time
        self.cars = cars
        # Initialize internal variables, define some variables at your convenience.
        pass

    def checkZone(self, car, bouns):
        if car[1] >= bouns.area[0] and car[1] <= bouns.area[1] and car[2] >= bouns.area[2] and car[2] <= bouns.area[3] and bouns.activation != None :
            bouns.activation = None
            return True
        return False
        # TODO: Perform geometricall checking for the zone area and centroid of robot
        # It Should be called after each robot's motion update

    def getMobility(self,car):
        if car[1] >= self.move.area[0] and car[1] <= self.move.area[1] and car[2] >= self.move.area[2] and car[2] <= self.move.area[3] \
            and self.move.activation :
            self.move.time += 1
            self.move.activation = None
            return True
        if self.move.time == 2000:
            self.move.time = 0
            return False
        if not self.move.time:
            self.move.time += 1
            return True
        return False        
        # TODO: Return move eligibility
        # It should be called before robot motion update

    def getShootabiliy(self, car):
        if car[1] >= self.shoot.area[0] and car[1] <= self.shoot.area[1] and car[2] >= self.shoot.area[2] and car[2] <= self.shoot.area[3] \
            and self.shoot.activation:
            self.shoot.time += 1
            self.shoot.activation = None
            return True
        if self.shoot.time == 2000:
            self.shoot.time = 0
            return False
        if not self.shoot.time:
            self.shoot.time += 1
            return True
        return False 
        # TODO: Return  shoot eligibility.
        # It should be called before robot shooting execution

    def _reset_bufzone(self):
        # set the areas randomly
        i = random.sample([0,1,2],3)
        punish_areas = random.sample([self.special_area[i[0]], self.special_area[5-i[0]]],2)
        self.move.area, self.shoot.area = punish_areas[0], punish_areas[1]
        bonus_hp_areas = random.sample([self.special_area[i[1]],self.special_area[5-i[1]]],2)
        self.red_hp.area, self.blue_hp.area = bonus_hp_areas[0], bonus_hp_areas[1]
        bonus_bullet_areas = random.sample([self.special_area[i[2]],self.special_area[5-i[2]]],2)
        self.red_bullet.area, self.blue_bullet.area = bonus_bullet_areas[0], bonus_bullet_areas[1]
        # activate the areas
        self.move.activation = 1
        self.shoot.activation = 1
        self.red_hp.activation = 1
        self.blue_hp.activation = 1
        self.red_bullet.activation = 1
        self.blue_bullet.activation = 1
        # TODO: Randomly shuffle the buffer zone while followinig the symmetric re

    def update(self):
        if not self.time % 60:
            self._reset_bufzone()
        
        # TODO: Count time reshuffle the buffer zone; unfreeze cars and enable shooting
        pass

class kernal(gym.Env): # gym.Env

    def __init__(self, car_num,  robot_id,render=False, record=True):# map, car, render
        
        self.robot_id = robot_id
        self.car_num = car_num
        self.render = render
        # below are params that can be challenged depended on situation
        self.bullet_speed = 12.5
        self.motion = 6
        self.rotate_motion = 4
        self.yaw_motion = 1
        self.camera_angle = 75 / 2
        self.lidar_angle = 180 / 2
        self.move_discount = 0.6
        # above are params that can be challenged depended on situation
        self.map_length = 808
        self.map_width = 448
        self.theta = np.rad2deg(np.arctan(45/60))
        self.record=record
        self.areas = np.array([[[708.0, 808.0, 348.0, 448.0],
                                [708.0, 808.0, 0.0, 100.0]],
                                [[0.0, 100.0, 0.0, 100.0],
                                [0.0, 100.0, 348.0, 448.0]]], dtype='float32')
        self.special_area = np.array([[591.0, 645.0, 141.0, 189.0],
                                [731.0, 785.0, 255.0, 303.0], 
                                [377.0, 431.0, 20.5, 68.5], 
                                [377.0, 431.0, 379.5, 427.5],
                                [163.0, 217.0, 259.0, 307.0], 
                                [23.0, 77.0, 145.0, 193.0]], dtype='float32')
        self.barriers = np.array([[150.0, 230.0, 214.0, 234.0],
                                  [578.0, 658.0, 214.0, 234.0],
                                  [0.0, 100.0, 100.0, 120.0],
                                  [708.0, 808.0, 328.0, 348.0],
                                  [354.0, 454.0, 93.5, 113.5],
                                  [354.0, 454.0, 334.5, 354.5],
                                  [150.0, 170.0, 348.0, 448.0],
                                  [638.0, 658.0, 0.0, 100.0],
                                  [386.3, 421.7, 206.3, 241.7]], dtype='float32') # barrier_horizcontal: B2, B8, B1, B9, B4, B6; barrier_vertical: B3, B7

        self.reward = 0.0
        self.action_space = spaces.Box(high = 1, low = -1, shape = (8,),dtype = int)
        self.observation_space = spaces.Box(low = -180.0, high = 2000.0, shape = (17, ), dtype = np.float32)


        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.map_length, self.map_width))
            pygame.display.set_caption('RM AI Challenge Simulator')
            self.gray = (180, 180, 180)
            self.red = (190, 20, 20)
            self.blue = (10, 125, 181)
            # load barriers imgs #
            self.barriers_img = []
            self.barriers_rect = []
        
            barrier_horizontal_tall = pygame.image.load('./imgs/barrier_horizontal.png')
            barrier_horizontal_tall = pygame.transform.scale(barrier_horizontal_tall, (100,20))
            barrier_horizontal_short = pygame.image.load('./imgs/barrier_horizontal.png')
            barrier_horizontal_short = pygame.transform.scale(barrier_horizontal_short, (80,20)) 
            barrier_vertical = pygame.image.load('./imgs/barrier_vertical.png')
            barrier_vertical = pygame.transform.scale(barrier_vertical, (20,100))
            barrier_small = pygame.image.load('./imgs/barrier_small.png')
            barrier_small = pygame.transform.scale(barrier_small, (25,25))
            barrier_small = pygame.transform.rotate(barrier_small, 45)
            for i in range(self.barriers.shape[0]+1):
                if i < 2:
                    self.barriers_img.append(barrier_horizontal_short)
                elif i < 6:
                    self.barriers_img.append(barrier_horizontal_tall)
                elif i < 8:
                    self.barriers_img.append(barrier_vertical)
                else:
                    self.barriers_img.append(barrier_small)

                self.barriers_rect.append(self.barriers_img[-1].get_rect())
                if i < 8:
                    self.barriers_rect[-1].center = [self.barriers[i][0:2].mean(), self.barriers[i][2:4].mean()]
                else:
                    self.barriers_rect[-1].center = [404,224]
            # load start imgs OK~
            self.areas_img = []
            self.areas_rect = []
            for oi, o in enumerate(['red', 'blue']):
                for ti, t in enumerate(['start', 'start']):
                    startpicture = pygame.image.load('./imgs/area_{}_{}.png'.format(t, o))
                    startpicture = pygame.transform.scale(startpicture, (100,100))
                    self.areas_img.append(startpicture)
                    self.areas_rect.append(self.areas_img[-1].get_rect())
                    self.areas_rect[-1].center = [self.areas[oi, ti][0:2].mean(), self.areas[oi, ti][2:4].mean()]
            # load special_area imgs
            picture = pygame.image.load('./imgs/special_area.png')
            picture = pygame.transform.scale(picture, (54,48))            
            self.special_area_img = [picture for i in range(6)]
            self.special_area_rect = []
            for i in range(6):
                self.special_area_rect.append(self.special_area_img[i].get_rect())
                self.special_area_rect[-1].center = [self.special_area[i][0:2].mean(), self.special_area[i][2:4].mean()] 
            # load another imgs
            self.chassis_img = pygame.image.load('./imgs/chassis_g.png')
            self.gimbal_img = pygame.image.load('./imgs/gimbal_g.png')
            self.bullet_img = pygame.image.load('./imgs/bullet_s.png')
            self.info_bar_img = pygame.image.load('./imgs/info_bar.png')
            self.bullet_rect = self.bullet_img.get_rect()
            self.info_bar_rect = self.info_bar_img.get_rect()
            self.info_bar_rect.center = [200, self.map_width/2]
            pygame.font.init()
            self.font = pygame.font.SysFont('info', 20)
            self.clock = pygame.time.Clock()
        self.orders = np.zeros((self.car_num, 8), dtype='int8')

    def reset(self): 
        self.time = 180.0
        self.orders = np.zeros((8,), dtype='int8')
        self.acts = np.zeros((self.car_num, 8),dtype='float32')
        self.obs = np.zeros((self.car_num, 17), dtype='float32')
        self.vision = np.zeros((self.car_num, self.car_num), dtype='int8')
        self.detect = np.zeros((self.car_num, self.car_num), dtype='int8')
        self.observ = np.zeros((self.car_num, self.car_num), dtype='float32')
        self.bullets = []
        self.epoch = 0 
        self.n = 0
        self.dev = False
        self.memory = []
        cars = np.array([[1, 50, 50, 0, 0, 0, 2000, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 50, 450, 0, 0, 0, 2000, 0, 0, 1, 0, 0, 0, 0, 0],
                         [1, 750, 50, 0, 0, 0, 2000, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 750, 450, 0, 0, 0, 2000, 0, 0, 1, 0, 0, 0, 0, 0]], dtype='float32')
        self.cars = cars[0:self.car_num]
        a = np.zeros((17, ), dtype='float32')
        return a

    def step(self, orders):
        self.orders = orders
        for _ in range(10):
            self.one_epoch() 
        self.cars.flatten()
        reward = self.compute_reward()
        done = bool( self.time <= 0 )# to be clearified
        cars = self.cars.flatten()
        observ = self.observ.flatten()
        observation = cars.tolist()
        observation.append(self.time)
        observation += observ.tolist()
        #state = np.array(observation, dtype=np.float32)
        #to be continue:)
        return np.array(observation, dtype=np.float32), reward, done, {}

    def compute_reward(self):
        #hyperparams to be designed...
        a1 = 1
        a2 = 1
        a = 1
        s0 = 1
        x0 = 1
        rew_KO = 1
        rew_win = 1
        #reward design
        #design score
        #红方：己方；蓝方：敌方
        red_hp = 0
        blue_hp = 0
        for n in range(self.car_num):
            if n % 2 != 0:
                red_hp += self.cars[n, 6]
            else:
                blue_hp += self.cars[n, 6]
        score = red_hp - blue_hp
        # design reward for shooting SQ
        Q = 240
        Q0 = (-(2*a*x0 + 4*a*Q) + ((4*a*Q+2*a*x0)**2 + 12*a*(-a*Q*Q+a*x0*x0-s0))**0.5)/-6/a
        #assume 1V1
        x = self.cars[0, 5] 
        sq = -a*(Q-x)**2 + 2*a*Q0*(Q-x) - a*(x0**2) + 2*a*x0*Q0
        # more work needed
        reward = a1*score + a2*(rew_KO +rew_win) + sq
        return reward

    def one_epoch(self):  
        referee = RefereeSystem(self.special_area, self.time, self.cars) 
        #for n in range(self.car_num):
        n=0
        if not self.epoch % 10:
            self.orders_to_acts(n)
        # move car one by one
        self.move_car(n)
        # check bouns
        for i in [referee.red_hp, referee.blue_hp, referee.red_bullet, referee.blue_bullet]:
            if referee.checkZone(self.cars[n], i):
                if i == referee.red_hp:
                    for ii in range(self.car_num):
                        if self.cars[ii,0] == 0:
                            self.cars[ii,6] += 200
                if i == referee.blue_hp:
                    for ii in range(self.car_num):
                        if self.cars[ii,0] == 1:
                            self.cars[ii,6] += 200
                if i == referee.red_bullet:
                    for ii in range(self.car_num):
                        if self.cars[ii,0] ==0:
                            self.cars[ii,10] +=100
                if i == referee.blue_bullet:
                    for ii in range(self.car_num):
                        if self.cars[ii,0] ==1:
                            self.cars[ii,10] +=100
        if not self.epoch % 20:# 10Hz为一周期结算
            if self.cars[n, 5] >= 360: # 5:枪口热量 6:血量
                self.cars[n, 6] -= (self.cars[n, 5] - 360) * 40
                self.cars[n, 5] = 360
            elif self.cars[n, 5] > 240:
                self.cars[n, 6] -= (self.cars[n, 5] - 240) * 4
            self.cars[n, 5] -= 12 if self.cars[n, 6] >= 400 else 24
        if self.cars[n, 5] <= 0: self.cars[n, 5] = 0
        if self.cars[n, 6] <= 0: self.cars[n, 6] = 0
        if referee.getShootabiliy(self.cars[n]): self.acts[n, 4] = 0
        if not self.acts[n, 5]: self.acts[n, 4] = 0 # 5:连发 6:单发
        if not self.epoch % 200: # 200epoch = 1s
                self.time -= 1
                referee.update()
        self.get_lidar_camera_vision()
        # move bullet one by one
        i = 0
        while len(self.bullets):
            if self.move_bullet(i):
                del self.bullets[i]
                i -= 1
            i += 1
            if i >= len(self.bullets): break
        self.epoch += 1
        bullets = []
        for i in range(len(self.bullets)):
            bullets.append(bullet(self.bullets[i].center, self.bullets[i].angle, self.bullets[i].speed, self.bullets[i].owner))
        #if self.record: self.memory.append(record(self.time, self.cars.copy(), self.compet_info.copy(), self.detect.copy(), self.vision.copy(), bullets))
        if self.record: self.memory.append(record(self.time, self.cars.copy(), self.detect.copy(), self.vision.copy(), bullets))
        if self.render: self.update_display()

    def move_car(self, n):
        referee = RefereeSystem(self.special_area, self.time, self.cars) 
        if not referee.getMobility:
            # move chassis
            if self.acts[n, 0]:
                p = self.cars[n, 3]
                self.cars[n, 3] += self.acts[n, 0]
                if self.cars[n, 3] > 180: self.cars[n, 3] -= 360
                if self.cars[n, 3] < -180: self.cars[n, 3] += 360
                if self.check_interface(n):
                    self.acts[n, 0] = -self.acts[n, 0] * self.move_discount
                    self.cars[n, 3] = p
        # move gimbal
        if self.acts[n, 1]:
            self.cars[n, 4] += self.acts[n, 1]
            if self.cars[n, 4] > 90: self.cars[n, 4] = 90
            if self.cars[n, 4] < -90: self.cars[n, 4] = -90
        # print(self.acts[n, 7])
    '''
        if self.acts[n, 7]:
            if self.car_num > 1:
                select = np.where((self.vision[n] == 1))[0]
                if select.size:
                    angles = np.zeros(select.size)
                    for ii, i in enumerate(select):
                        x, y = self.cars[i, 1:3] - self.cars[n, 1:3]
                        angle = np.angle(x+y*1j, deg=True) - self.cars[i, 3]
                        if angle >= 180: angle -= 360
                        if angle <= -180: angle += 360
                        if angle >= -self.theta and angle < self.theta:
                            armor = self.get_armor(self.cars[i], 2)
                        elif angle >= self.theta and angle < 180-self.theta:
                            armor = self.get_armor(self.cars[i], 3)
                        elif angle >= -180+self.theta and angle < -self.theta:
                            armor = self.get_armor(self.cars[i], 1)
                        else: armor = self.get_armor(self.cars[i], 0)
                        x, y = armor - self.cars[n, 1:3]
                        angle = np.angle(x+y*1j, deg=True) - self.cars[n, 4] - self.cars[n, 3]
                        if angle >= 180: angle -= 360
                        if angle <= -180: angle += 360
                        angles[ii] = angle
                    m = np.where(np.abs(angles) == np.abs(angles).min())
                    self.cars[n, 4] += angles[m][0]
                    if self.cars[n, 4] > 90: self.cars[n, 4] = 90
                    if self.cars[n, 4] < -90: self.cars[n, 4] = -90
            # move x and y
            if not referee.getMobility:
                if self.acts[n, 2] or self.acts[n, 3]:
                    angle = np.deg2rad(self.cars[n, 3])
                    # x
                    p = self.cars[n, 1]
                    self.cars[n, 1] += (self.acts[n, 2]) * np.cos(angle) - (self.acts[n, 3]) * np.sin(angle)
                    if self.check_interface(n):
                        self.acts[n, 2] = -self.acts[n, 2] * self.move_discount
                        self.cars[n, 1] = p
                    # y
                    p = self.cars[n, 2]
                    self.cars[n, 2] += (self.acts[n, 2]) * np.sin(angle) + (self.acts[n, 3]) * np.cos(angle)
                    if self.check_interface(n):
                        self.acts[n, 3] = -self.acts[n, 3] * self.move_discount
                        self.cars[n, 2] = p
            # fire or not
            if self.acts[n, 4] and self.cars[n, 10]:
                if not referee.getShootabiliy:
                    if self.cars[n, 9]:
                        self.cars[n, 10] -= 1
                        self.bullets.append(bullet(self.cars[n, 1:3], self.cars[n, 4]+self.cars[n, 3], self.bullet_speed, n))
                        self.cars[n, 5] += self.bullet_speed
                        self.cars[n, 9] = 0
                else:
                    self.cars[n, 9] = 1
            else:
                self.cars[n, 9] = 1
        elif self.cars[n, 7] < 0: assert False
        else:
            self.cars[n, 7] -= 1
            if self.cars[n, 7] == 0:
                self.cars[n, 8] == 0
    '''
    def move_bullet(self, n):
        '''
        move bullet No.n, if interface with wall, barriers or cars, return True, else False
        if interface with cars, cars'hp will decrease
        '''
        old_point = self.bullets[n].center.copy()
        self.bullets[n].center[0] += self.bullets[n].speed * np.cos(np.deg2rad(self.bullets[n].angle))
        self.bullets[n].center[1] += self.bullets[n].speed * np.sin(np.deg2rad(self.bullets[n].angle))
        # bullet wall check
        if self.bullets[n].center[0] <= 0 or self.bullets[n].center[0] >= self.map_length \
            or self.bullets[n].center[1] <= 0 or self.bullets[n].center[1] >= self.map_width: return True
        # bullet barrier check
        for b in self.barriers:
            if self.line_barriers_check(self.bullets[n].center, old_point): return True
        # bullet armor check
        for i in range(len(self.cars)):
            if i == self.bullets[n].owner: continue
            if np.abs(np.array(self.bullets[n].center) - np.array(self.cars[i, 1:3])).sum() < 52.5:
                points = self.transfer_to_car_coordinate(np.array([self.bullets[n].center, old_point]), i)
                if self.segment(points[0], points[1], [-18.5, -5], [-18.5, 6]) \
                or self.segment(points[0], points[1], [18.5, -5], [18.5, 6]) \
                or self.segment(points[0], points[1], [-5, 30], [5, 30]) \
                or self.segment(points[0], points[1], [-5, -30], [5, -30]):
                    self.cars[i, 6] -= 50
                    return True
                if self.line_rect_check(points[0], points[1], [-18, -29, 18, 29]): return True
        return False

    def update_display(self):
        assert self.render, 'only render mode need update_display'
        self.screen.fill(self.gray)
        for i in range(len(self.barriers_rect)):
            self.screen.blit(self.barriers_img[i], self.barriers_rect[i])
        for i in range(len(self.areas_rect)):
            self.screen.blit(self.areas_img[i], self.areas_rect[i]) 
        for i in range(len(self.special_area_rect)):
            self.screen.blit(self.special_area_img[i], self.special_area_rect[i])
        for i in range(len(self.bullets)):
            self.bullet_rect.center = self.bullets[i].center
            self.screen.blit(self.bullet_img, self.bullet_rect)
        for n in range(self.car_num):
            chassis_rotate = pygame.transform.rotate(self.chassis_img, -self.cars[n, 3]-90)
            gimbal_rotate = pygame.transform.rotate(self.gimbal_img, -self.cars[n, 4]-self.cars[n, 3]-90)
            chassis_rotate_rect = chassis_rotate.get_rect()
            gimbal_rotate_rect = gimbal_rotate.get_rect()
            chassis_rotate_rect.center = self.cars[n, 1:3]
            gimbal_rotate_rect.center = self.cars[n, 1:3]
            self.screen.blit(chassis_rotate, chassis_rotate_rect)
            self.screen.blit(gimbal_rotate, gimbal_rotate_rect)
        for i in range(len(self.special_area_rect)):
            self.screen.blit(self.special_area_img[i], self.special_area_rect[i])
            #self.screen.blit(self.special_area_img[1], self.special_area_rect[1])
        for n in range(self.car_num):
            select = np.where((self.vision[n] == 1))[0]+1
            select2 = np.where((self.detect[n] == 1))[0]+1
            info = self.font.render('{} | {}: {} {}'.format(int(self.cars[n, 6]), n+1, select, select2), True, self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3]+[-20, -60])
            info = self.font.render('{} {}'.format(int(self.cars[n, 10]), int(self.cars[n, 5])), True, self.blue if self.cars[n, 0] else self.red)
            self.screen.blit(info, self.cars[n, 1:3]+[-20, -45])
        info = self.font.render('time: {}'.format(self.time), False, (0, 0, 0))
        self.screen.blit(info, (8, 8))
        if self.dev: self.dev_window()
        pygame.display.flip()

    def dev_window(self):
        for n in range(self.car_num):
            wheels = self.check_points_wheel(self.cars[n])
            for w in wheels:
                pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, w.astype(int), 3)
            armors = self.check_points_armor(self.cars[n])
            for a in armors:
                pygame.draw.circle(self.screen, self.blue if self.cars[n, 0] else self.red, a.astype(int), 3)
        self.screen.blit(self.info_bar_img, self.info_bar_rect)
        for n in range(self.car_num):
            tags = ['owner', 'x', 'y', 'angle', 'yaw', 'heat', 'hp', 'freeze_time', 'is_supply', 
                    'can_shoot', 'bullet', 'stay_time', 'wheel_hit', 'armor_hit', 'car_hit']
            info = self.font.render('car {}'.format(n), False, (0, 0, 0))
            self.screen.blit(info, (8+n*100, 100))
            for i in range(self.cars[n].size):
                info = self.font.render('{}: {}'.format(tags[i], int(self.cars[n, i])), False, (0, 0, 0))
                self.screen.blit(info, (8+n*100, 117+i*17))
        '''
        info = self.font.render('red   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[0, 0], \
                                self.compet_info[0, 1], self.compet_info[0, 3]), False, (0, 0, 0))
        self.screen.blit(info, (8, 372))
        info = self.font.render('blue   supply: {}   bonus: {}   bonus_time: {}'.format(self.compet_info[1, 0], \
                                self.compet_info[1, 1], self.compet_info[1, 3]), False, (0, 0, 0))
        '''
        #self.screen.blit(info, (8, 389))
    '''
    def get_order(self): 
        # get order from controler
        pygame.init()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        pressed = pygame.key.get_pressed()
        
        if pressed[pygame.K_1]: self.n = 0
        if pressed[pygame.K_2]: self.n = 1
        if pressed[pygame.K_3]: self.n = 2
        if pressed[pygame.K_4]: self.n = 3
        
        self.orders[self.n] = 0
        
        order_name = random.choice([0,1,2,3,4,5,6,7])
        if order_name < 4:
            order  = random.choice([1,-1])
        else:
            order = random.choice([0,1])
        self.orders[self.n,order_name] += order
        
        if pressed[pygame.K_w]: self.orders[self.n, 0] += 1 # Forward
        if pressed[pygame.K_s]: self.orders[self.n, 0] -= 1 # Backward
        if pressed[pygame.K_q]: self.orders[self.n, 1] -= 1 # Left rotate
        if pressed[pygame.K_e]: self.orders[self.n, 1] += 1 # Right rotate
        if pressed[pygame.K_a]: self.orders[self.n, 2] -= 1 # Left shift
        if pressed[pygame.K_d]: self.orders[self.n, 2] += 1 # Right shift
        if pressed[pygame.K_b]: self.orders[self.n, 3] -= 1 # Gimbal left
        if pressed[pygame.K_m]: self.orders[self.n, 3] += 1 # Gimbal Right
        if pressed[pygame.K_SPACE]: self.orders[self.n, 4] = 1 # Shoot
        else: self.orders[self.n, 4] = 0 
        if pressed[pygame.K_f]: self.orders[self.n, 5] = 1 # Supply
        else: self.orders[self.n, 5] = 0
        if pressed[pygame.K_r]: self.orders[self.n, 6] = 1 # Shoot mode
        else: self.orders[self.n, 6] = 0
        if pressed[pygame.K_n]: self.orders[self.n, 7] = 1 
        else: self.orders[self.n, 7] = 0
        
        if pressed[pygame.K_TAB]: self.dev = True
        else: self.dev = False
        return False
    '''
    def orders_to_acts(self, n):
        for i in range(4):
            self.orders[i] = np.clip(self.orders[i], 0, 1)
        # turn orders to acts
        self.acts[n, 2] += self.orders[0] * 1.5 / self.motion
        if self.orders[0] == 0:
            if self.acts[n, 2] > 0: self.acts[n, 2] -= 1.5 / self.motion
            if self.acts[n, 2] < 0: self.acts[n, 2] += 1.5 / self.motion
        if abs(self.acts[n, 2]) < 1.5 / self.motion: self.acts[n, 2] = 0
        if self.acts[n, 2] >= 1.5: self.acts[n, 2] = 1.5
        if self.acts[n, 2] <= -1.5: self.acts[n, 2] = -1.5
        # x, y
        self.acts[n, 3] += self.orders[1] * 1 / self.motion
        if self.orders[1] == 0:
            if self.acts[n, 3] > 0: self.acts[n, 3] -= 1 / self.motion
            if self.acts[n, 3] < 0: self.acts[n, 3] += 1 / self.motion
        if abs(self.acts[n,3]) < 1 / self.motion: self.acts[n, 3] = 0
        if self.acts[n,3] >= 1: self.acts[n, 3] = 1
        if self.acts[n,3] <= -1: self.acts[n, 3] = -1
        # rotate chassis
        self.acts[n, 0] += self.orders[2] * 1 / self.rotate_motion
        if self.orders[2] == 0:
            if self.acts[n, 0] > 0: self.acts[n, 0] -= 1 / self.rotate_motion
            if self.acts[n, 0] < 0: self.acts[n, 0] += 1 / self.rotate_motion
        if abs(self.acts[n, 0]) < 1 / self.rotate_motion: self.acts[n, 0] = 0
        if self.acts[n, 0] > 1: self.acts[n, 0] = 1
        if self.acts[n, 0] < -1: self.acts[n, 0] = -1
        # rotate yaw
        self.acts[n, 1] += self.orders[3] / self.yaw_motion
        if self.orders[3] == 0:
            if self.acts[n, 1] > 0: self.acts[n, 1] -= 1 / self.yaw_motion
            if self.acts[n, 1] < 0: self.acts[n, 1] += 1 / self.yaw_motion
        if abs(self.acts[n, 1]) < 1 / self.yaw_motion: self.acts[n, 1] = 0
        if self.acts[n, 1] > 3: self.acts[n, 1] = 3
        if self.acts[n, 1] < -3: self.acts[n, 1] = -3
        self.acts[n, 4] = self.orders[4]
        self.acts[n, 6] = self.orders[5]
        self.acts[n, 5] = self.orders[6]
        self.acts[n, 7] = self.orders[7]

    def set_car_loc(self, n, loc):
        self.cars[n, 1:3] = loc

    def get_map(self):
        return g_map(self.map_length, self.map_width, self.areas, self.barriers, self.special_area)

    def cross(self, p1, p2, p3): # 叉乘
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1 * y2 - x2 * y1 

    def segment(self, p1, p2, p3, p4): # 判断p1, p2两线和p3, p4 两线是否交叉
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        if (max(p1[0], p2[0])>=min(p3[0], p4[0]) and max(p3[0], p4[0])>=min(p1[0], p2[0])
        and max(p1[1], p2[1])>=min(p3[1], p4[1]) and max(p3[1], p4[1])>=min(p1[1], p2[1])):
            if (self.cross(p1,p2,p3)*self.cross(p1,p2,p4)<=0 and self.cross(p3,p4,p1)*self.cross(p3,p4,p2)<=0): return True
            else: return False
        else: return False

    def line_rect_check(self, l1, l2, sq):
        # this part code came from: https://www.jianshu.com/p/a5e73dbc742a
        # check if line cross rect, sq = [x_leftdown, y_leftdown, x_rightup, y_rightup]
        p1 = [sq[0], sq[1]]
        p2 = [sq[2], sq[3]]
        p3 = [sq[2], sq[1]]
        p4 = [sq[0], sq[3]]
        if self.segment(l1,l2,p1,p2) or self.segment(l1,l2,p3,p4): return True
        else: return False

    def line_barriers_check(self, l1, l2):
        for b in self.barriers:
            sq = [b[0], b[2], b[1], b[3]]
            if self.line_rect_check(l1, l2, sq): return True
        return False

    def line_cars_check(self, l1, l2):
        for car in self.cars:
            if (car[1:3] == l1).all() or (car[1:3] == l2).all():
                continue
            p1, p2, p3, p4 = self.get_car_outline(car)
            if self.segment(l1, l2, p1, p2) or self.segment(l1, l2, p3, p4): return True
        return False

    def get_lidar_vision(self) : 
        for n in range(self.car_num):
            for i in range(self.car_num-1):
                x, y = self.cars[n-i-1, 1:3] - self.cars[n, 1:3]
                angle = np.angle(x+y*1j, deg=True)
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                angle = angle - self.cars[n, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                if abs(angle) < self.lidar_angle:
                    if self.line_barriers_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]) \
                    or self.line_cars_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]):
                        self.detect[n, n-i-1] = 0
                    else: self.detect[n, n-i-1] = 1
                else: self.detect[n, n-i-1] = 0

    def get_camera_vision(self): 
        for n in range(self.car_num):
            for i in range(self.car_num-1):
                x, y = self.cars[n-i-1, 1:3] - self.cars[n, 1:3]
                angle = np.angle(x+y*1j, deg=True)
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                angle = angle - self.cars[n, 4] - self.cars[n, 3]
                if angle >= 180: angle -= 360
                if angle <= -180: angle += 360
                if abs(angle) < self.camera_angle:
                    if self.line_barriers_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]) \
                    or self.line_cars_check(self.cars[n, 1:3], self.cars[n-i-1, 1:3]):
                        self.vision[n, n-i-1] = 0
                    else: self.vision[n, n-i-1] = 1
                else: self.vision[n, n-i-1] = 0

    def get_lidar_camera_vision(self):
        self.get_camera_vision()
        self.get_lidar_vision()
        for n in range(self.car_num):
            for i in range(self.car_num-1):
                if self.vision[n, n-i-1] == 1 or self.detect[n, n-i-1] == 1:
                    self.observ[n, n-i-1] = 1.0
                else:
                    self.observ[n, n-i-1] = 0.0
       
    def transfer_to_car_coordinate(self, points, n):
        pan_vecter = -self.cars[n, 1:3]
        rotate_matrix = np.array([[np.cos(np.deg2rad(self.cars[n, 3]+90)), -np.sin(np.deg2rad(self.cars[n, 3]+90))],
                                  [np.sin(np.deg2rad(self.cars[n, 3]+90)), np.cos(np.deg2rad(self.cars[n, 3]+90))]])
        return np.matmul(points + pan_vecter, rotate_matrix)

    def check_points_wheel(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-22.5, -29], [22.5, -29], 
                       [-22.5, -14], [22.5, -14], 
                       [-22.5, 14], [22.5, 14],
                       [-22.5, 29], [22.5, 29]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]

    def check_points_armor(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-6.5, -30], [6.5, -30], 
             [-18.5,  -7], [18.5,  -7],
             [-18.5,  0], [18.5,  0],
             [-18.5,  6], [18.5,  6],
             [-6.5, 30], [6.5, 30]])
        return [np.matmul(x, rotate_matrix) + car[1:3] for x in xs]

    def get_car_outline(self, car):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[-22.5, -30], [22.5, 30], [-22.5, 30], [22.5, -30]])
        return [np.matmul(xs[i], rotate_matrix) + car[1:3] for i in range(xs.shape[0])]

    def trans_special_barrier(self, points):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(90+45)), -np.sin(-np.deg2rad(90+45))],
                                  [np.sin(-np.deg2rad(90+45)), np.cos(-np.deg2rad(+90+45))]])
        vector = -np.array([404.0, 224.0])
        return np.matmul(points + vector, rotate_matrix)

    def check_interface(self, n):
        # car barriers assess
        wheels = self.check_points_wheel(self.cars[n])
        for w in wheels:
            if w[0] <= 0 or w[0] >= self.map_length or w[1] <= 0 or w[1] >= self.map_width:
                self.cars[n, 12] += 1
                return True
            for b in self.barriers[:8]:
                if w[0] >= b[0] and w[0] <= b[1] and w[1] >= b[2] and w[1] <= b[3]:
                    self.cars[n, 12] += 1
                    return True
        armors = self.check_points_armor(self.cars[n])
        for a in armors:
            if a[0] <= 0 or a[0] >= self.map_length or a[1] <= 0 or a[1] >= self.map_width:
                self.cars[n, 13] += 1
                self.cars[n, 6] -= 10
                return True
            for b in self.barriers[:8]:
                if a[0] >= b[0] and a[0] <= b[1] and a[1] >= b[2] and a[1] <= b[3]:
                    self.cars[n, 13] += 1
                    self.cars[n, 6] -= 10
                    return True
        # special_barrier car asses
        wheels_barrier = self.trans_special_barrier(wheels)
        for w in wheels_barrier:
            if w[0] >= -12.5 and w[0] <= 12.5 and w[1] >= -12.5 and w[1] <= 12.5:
                self.cars[n, 14] += 1
                return True
        armors_barrier = self.trans_special_barrier(armors)
        for a in armors_barrier:
            if a[0] >= -12.5 and a[0] <= 12.5 and a[1] >= -12.5 and a[1] <= 12.5:
                self.cars[n, 14] += 1
                self.cars[n, 6] -= 10
                return True
        # car car assess
        for i in range(self.car_num):
            if i == n: continue
            wheels_tran = self.transfer_to_car_coordinate(wheels, i) # 以car_i为参考系
            for w in wheels_tran:
                if w[0] >= -22.5 and w[0] <= 22.5 and w[1] >= -30 and w[1] <= 30:
                    self.cars[n, 14] += 1
                    return True
            armors_tran = self.transfer_to_car_coordinate(armors, i)
            for a in armors_tran:
                if a[0] >= -22.5 and a[0] <= 22.5 and a[1] >= -30 and a[1] <= 30:
                    self.cars[n, 14] += 1
                    self.cars[n, 6] -= 10
                    return True
        return False

    def get_armor(self, car, i):
        rotate_matrix = np.array([[np.cos(-np.deg2rad(car[3]+90)), -np.sin(-np.deg2rad(car[3]+90))],
                                  [np.sin(-np.deg2rad(car[3]+90)), np.cos(-np.deg2rad(car[3]+90))]])
        xs = np.array([[0, -30], [18.5, 0], [0, 30], [-18.5,  0]])
        return np.matmul(xs[i], rotate_matrix) + car[1:3]

    def save_record(self, file):
        np.save(file, self.memory)
            
            
''' 

important indexs
areas_index = [[{'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 0 bonus red
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 1 supply red
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 2 start 0 red
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}], # 3 start 1 red

               [{'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 0 bonus blue
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 1 supply blue
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 2 start 0 blue
                {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}]] # 3 start 1 blue
                            

barriers_index = [{'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 0 horizontal
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 1 horizontal
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 2 horizontal
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 3 vertical
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 4 vertical
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}, # 5 vertical
                  {'border_x0': 0, 'border_x1': 1,'border_y0': 2,'border_y1': 3}] # 6 vertical

armor编号：0：前，1：右，2：后，3左，车头为前

act_index = {'rotate_speed': 0, 'yaw_speed': 1, 'x_speed': 2, 'y_speed': 3, 'shoot': 4, 'shoot_mutiple': 5, 'supply': 6,
             'auto_aim': 7}

bullet_speed: 12.5


compet_info_index = {'red': {'supply': 0, 'bonus': 1, 'bonus_stay_time(deprecated)': 2, 'bonus_time': 3}, 
                     'blue': {'supply': 0, 'bonus': 1, 'bonus_stay_time(deprecated)': 2, 'bonus_time': 3}}
int, shape: (2, 4)

order_index = ['x', 'y', 'rotate', 'yaw', 'shoot', 'supply', 'shoot_mode', 'auto_aim']
int, shape: (8,)
    x, -1: back, 0: no, 1: head
    y, -1: left, 0: no, 1: right
    rotate, -1: anti-clockwise, 0: no, 1: clockwise, for chassis
    shoot_mode, 0: single, 1: mutiple
    shoot, 0: not shoot, 1: shoot
    yaw, -1: anti-clockwise, 0: no, 1: clockwise, for gimbal
    auto_aim, 0: not, 1: auto aim

car_index = {"owner": 0, 'x': 1, 'y': 2, "angle": 3, "yaw": 4, "heat": 5, "hp": 6, 
             "freeze_time": 7, "is_supply": 8, "can_shoot": 9, 'bullet': 10, 'stay_time': 11,
             'wheel_hit': 12, 'armor_hit': 13, 'car_hit': 14}
float, shape: (14,)

'''

    
