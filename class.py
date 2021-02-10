import random
import numpy as np
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