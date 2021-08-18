# Simulator😄

**This is a 2D simulator for RoboMaster AI Challenge, the environment for the training of reinforcement learning The simulation is able to achieve efficient collision detection and significantly accelerated reinforcement learning.**

[action](#action)

[observation](#observation)

[cars](#cars)

[acts](#acts)

[punish&bonus](#punish&bonus)

[map_parameter](#map_parameter)

![cb88efaa382167848c285826ea56124](https://user-images.githubusercontent.com/76484768/129831464-d8517a16-b431-4170-b3af-07fced416d3b.jpg =288*180)


<img src="https://user-images.githubusercontent.com/76484768/129831488-82dedc44-dd88-4dab-a2f7-318dcf039802" width = 20% height = 20% div align=right />

## action

Called `orders` in kernel.
```python
        self.action_space = spaces.Box(high = 1, low = -1, shape = (8,),dtype = int)
```
In function `orders_to_acts`, `np.clip` is used.

```python
for i in range(4):
	self.orders[i] = np.clip(self.orders[i], 0, 1)
```

||名称|范围|解释|手控按键|
|-|-|-|-|-|
|0|x|-1~1|-1：后退，0：不动，1：前进[3]|s/w|
|1|y|-1~1|-1：左移，0：不动，1：右移|q/e|
|2|rotate|-1~1|底盘，-1：左转，0：不动，1：右转|a/d|
|3|yaw|-1~1|云台，-1：左转，0：不动，1：右转|b/m|
|4|shoot|0~1|是否射击，0：否，1：是|space|
|5|supply|0~1|时候触发补给，0：否，1：是|f|
|6|shoot_mode|0~1|射击模式，0：单发，1：连发|r|
|7|auto_aim|0~1|是否启用自瞄，0：否，1：是|n|

## observation

（有待完善

```python
        self.observation_space = spaces.Box(low = -180.0, high = 2000.0, shape = (17, ), dtype = np.float32)
```

|num|name|min|max|
|-|-|-|-|
|[0, 14]|[car_info](#cars)|-180.0|800.0|
|15|time|0.0|180.0|
|[16, 19]|[observ](#observ)|0.0|1.0|

## cars

`float`，`shape`（car_mun，15），`car_num`为机器人的数量：

|引索|名称|类型|范围|解释|
|---|---|---|---|---|
|0|owner|int|0~1|队伍，0：红方，1：蓝方|
|1|x|float|0~800|x坐标[0]|
|2|y|float|0~500|y坐标|
|3|angle|float|-180~180|底盘绝对角度[1]|
|4|yaw|float|-90~90|云台相对底盘角度|
|5|heat|int|0~|枪口热度|
|6|hp|int|0~2000|血量|
|7|freeze_time|int||【已删除】|
|8|is_supply|bool||【已删除】|
|9|can_shoot|bool|0~1|决策频率高于出弹最高频率（10Hz）|
|10|bullet|int|0~|剩余子弹量|
|11|stay_time|int||【已删除】|
|12|wheel_hit|int|0~|轮子撞墙的次数|
|13|armor_hit|int|0~|装甲板撞墙的次数|
|14|car_hit|int|0~|轮子或装甲板撞车的次数|

## detect&vision&observ
shape: (car_num, car_num)

### detect
`get_lidar_vision`
### vision
`get_camera_vision`
### observ
`observ`: `detect || vision`

```python
#	   0  1  2  3
detect = [[0, 1, 0, 0], # 0
          [0, 0, 1, 1], # 1
          [0, 0, 0, 0], # 2
          [1, 0, 0, 0]] # 3
```

表示：

0号车能检测到1号车

1号车能检测到2号车和3号车

2号车检测不到任何车

3号车能检测到0号车

## acts

`acts`是一个较底层的action，类型`float`，`shape`为：（car_num，8）

|引索1|名称|解释|
|-|-|-|
|0|rotate_speed|底盘旋转速度|
|1|yaw_speed|云台旋转速度|
|2|x_speed|前进后退速度|
|3|y_speed|左右平移速度|
|4|shoot|是否发射|
|5|shoot_mutiple|是否连发|
|6|supply|是否触发补给|
|7|auto_aim|是否自动瞄准|

## punish&bonus

```python
class Move_Shoot:
    def __init__(self, area, time, activation):
       self.area = area
       self.time = time
       self.activation = activation
```
```python
class RefereeSystem:
    move = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    shoot = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    red_hp = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    blue_hp = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    red_bullet = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
    blue_bullet = Move_Shoot(np.zeros(4, dtype='float32'), 0, None)
	def __init__(self, special_area, time, cars):
	def checkZone(self, car, bouns):
	def getMobility(self,car):
	def getShootabiliy(self, car):
	def _reset_bufzone(self):
	def update(self):
```

# map_parameter

|name|||解释|
|---|---|---|---|
|length|int|808|地图长度|
|width|int|448|地图宽度|
|[special_area](#special_area)|float|(8, 4)|supply & punish area|
|[areas](#areas)|float|(4, 4)|start areas|
|[barriers](#barriers)|float|(9, 4)|障碍物的位置信息|

以地图左上角为原点

||名称|范围|解释|
|---|---|---|---|
|0|border_x0|0~808|左边界|
|1|border_x1|0~808|右边界|
|2|border_y0|0~448|上边界|
|3|border_y1|0~448|下边界|

## special_area

`shape` (6, 4)

`area1` ,`area2` centrosymmetric

[reset randomly](#punish&bonus)

|area1|area2|
|-|-|
|red_hp|blue_hp|
|red_bullet|blue_bullet|
|move|shoot|

### areas

`start_area`, `shape` (4, 4) 

|num|team|
|-|-|
|0, 1|red|
|2, 3|blue|

### barriers

`shape`（9，4）

||name|
|-|-|
|0|barrier_horizontal_short|
|1|barrier_horizontal_short|
|2|barrier_horizontal_tall|
|3|barrier_horizontal_tall|
|4|barrier_horizontal_tall|
|5|barrier_horizontal_tall|
|6|barrier_vertical|
|7|barrier_vertical|
|8|barrier_small|
