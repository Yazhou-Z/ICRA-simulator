import numpy as np
import math
import cmath
class kenal(object):
    def __init__(self, car_num, render=False, record=True):
        self.car_num = car_num
        self.render = render
        # below are params that can be challenged depended on situation
        self.bullet_speed = 12.5
        self.motion = 6
        self.rotate_motion = 4
        self.yaw_motion = 1
        self.camera_angle = 75 / 2
        self.lidar_angle = 120 / 2
        self.move_discount = 0.6
        # above are params that can be challenged depended on situation
        self.map_length = 800
        self.map_width = 500
        self.theta = np.rad2deg(np.arctan(45/60))
        self.record=record
        self.areas = np.array([[[580.0, 680.0, 275.0, 375.0],
                                [350.0, 450.0, 0.0, 100.0],
                                [700.0, 800.0, 400.0, 500.0],
                                [0.0, 100.0, 400.0, 500.0]],
                               [[120.0, 220.0, 125.0, 225.0],
                                [350.0, 450.0, 400.0, 500.0],
                                [0.0, 100.0, 0.0, 100.0],
                                [700.0, 800.0, 0.0, 100.0]]], dtype='float32')
        self.barriers = np.array([[350.0, 450.0, 237.5, 262.5],
                                  [120.0, 220.0, 100.0, 125.0],
                                  [580.0, 680.0, 375.0, 400.0],
                                  [140.0, 165.0, 260.0, 360.0],
                                  [635.0, 660.0, 140.0, 240.0],
                                  [325.0, 350.0, 400.0, 500.0],
                                  [450.0, 475.0, 0.0, 100.0]], dtype='float32')
        if render:
            global pygame
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.map_length, self.map_width))
            pygame.display.set_caption('RM AI Challenge Simulator')
            self.gray = (180, 180, 180)
            self.red = (190, 20, 20)
            self.blue = (10, 125, 181)
            # load barriers imgs
            self.barriers_img = []
            self.barriers_rect = []
            for i in range(self.barriers.shape[0]):
                self.barriers_img.append(pygame.image.load('./imgs/barrier_{}.png'.format('horizontal' if i < 3 else 'vertical')))
                self.barriers_rect.append(self.barriers_img[-1].get_rect())
                self.barriers_rect[-1].center = [self.barriers[i][0:2].mean(), self.barriers[i][2:4].mean()]
            # load areas imgs
            self.areas_img = []
            self.areas_rect = []
            for oi, o in enumerate(['red', 'blue']):
                for ti, t in enumerate(['bonus', 'supply', 'start', 'start']):
                    self.areas_img.append(pygame.image.load('./imgs/area_{}_{}.png'.format(t, o)))
                    self.areas_rect.append(self.areas_img[-1].get_rect())
                    self.areas_rect[-1].center = [self.areas[oi, ti][0:2].mean(), self.areas[oi, ti][2:4].mean()]
            # load supply head imgs
            self.head_img = [pygame.image.load('./imgs/area_head_{}.png'.format(i)) for i in ['red', 'blue']]
            self.head_rect = [self.head_img[i].get_rect() for i in range(len(self.head_img))]
            self.head_rect[0].center = [self.areas[0, 1][0:2].mean(), self.areas[0, 1][2:4].mean()]
            self.head_rect[1].center = [self.areas[1, 1][0:2].mean(), self.areas[1, 1][2:4].mean()]
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

            map_width, map_length = 810, 510
            self.sp_map = np.zero([map_width, map_length], dtype='uint32')
            self.sp_value = np.zero([map_width, map_length], dtype='float')
            self.sp_flag = np.zero([map_width, map_length], dtype='uint32')
            self.sp_ff = np.zero([map_width, map_length], dtype='uint32')
            self.sp_route=np.zero([map_width*map_length*5,2],dtype='uint32')
            self.sp_last=np.zero([map_width, map_length],dtype='uint32')
            self.sp_tag=0
            self.sp_route_th=0
            self.sp_target_global=None
            self.sp_angle_lowerbound=0
            self.sp_angle_upperbound=0
            self.sp_v_lowerbound=-0.001
            self.sp_v_upperbound=0.01
            self.sp_Penalty_value=np.zero(map_width*map_length*5,dtype='float')
            self.sp_circular_obstacles=[]
    def sp_init(self,car_th):
        map_width, map_length=810,510
        def sp_init_change_value(f):
            global r
            for i in range(math.floor(f[0]),math.ceil(f[1])):
                for j in range(math.floor(f[2]),math.ceil(f[3])):
                    if (self.sp_map[i][j]==0):
                        self.sp_map[i][j]=1
                        r+=1
                        seq[r]=[i,j]
        def add_circular_obstacles(x,y,xx,yy):
            self.sp_circular_obstacles+=[[(x+xx)/2,(y+yy)/2,math.sqrt((xx-x)*(xx-x)+(yy-y)*(yy-y))/2]]
        def sp_circular_obstacles(f):
            x,xx,y,yy=f[0],f[1],f[2],f[3]
            if (x<xx):swap(x,xx)
            if (y<yy):swap(y,yy)
            dd=min(xx-x,yy-y)
            while (xx-x>dd or yy-y>dd):
                add_circular_obstacles(x,y,x+dd,y+dd)
             if (xx-x>yy-y)x+=dd
                else:y+=dd
            add_circular_obstacles(xx-dd, yy-dd, xx, yy)
        self.sp_map=np.zero([map_width,map_length],dtype='uint32')
        self.sp_value=np.zero([map_width,map_length],dtype='float')
        self.sp_flag=np.zero([map_width,map_length],dtype='uint16')
        seq=np.zero([map_width*map_length*5,2],dtype='uint32')

        l,r=0,0
        c=[(0,1),(0,-1),(1,0),(-1,0)]
        for i in xrange(self.barriers.shape[0]):
            sp_init_change_value(self.barriers[i])
            sp_circular_obstacles(self.barriers[i])
        for i in xrange(cars.shape[0]):
            if (i!=car_th):
                sp_init_change_value(np.array([cars[i][0]-30,cars[i][0]+30,cars[i][1]-30,cars[i][1]+30]))

        while (l<r):
            l+=1
            for dx,dy in c:
                x,y=dx+seq[sp_l][0],dy+seq[l][1]
                if (self.sp_flag[x][y]==0):
                    self.sp_flag[x][y]=1
                    self.sp_value[x][y]=self.sp_value[[seq[l][0]][seq[l][1]]+1
                    r+=1
                    seq[r]=[x,y]

        for i in xrange(1,map_width+1):
            for j in range(1,map_length+1):
                if (self.sp_value[i][j]<=25):
                    self.sp_map[i][j]=1
                else:self.sp_value[i][j]-=25;
        self.sp_value=200/(self.sp_value+1e-5)+1

    def sp_calc(self,p_begin):
        map_width, map_length=810,510

        self.sp_map = np.zero([map_width, map_length], dtype='uint32')
        self.sp_value = np.zero([map_width, map_length], dtype='float')
        self.sp_flag = np.zero([map_width, map_length], dtype='uint32')
        self.sp_ff = np.zero([map_width, map_length], dtype='uint32')
        self.sp_last = np.zero([map_width, map_length,2], dtype='uint32')
        seq=np.zero([map_width*map_length*5,2],dtype='uint32')
        f=np.zero([map_width*map_length*5,2],dtype='float')
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):f[i][j]=1e15

        x,y=p_begin
        l,r,c=0,1,[(0,1),(0,-1),(1,0),(-1,0)]
        seq[r]=[x,y]
        f[x]=0
        while (l<r&&r<map_width*map_length*5-4):
            l+=1
            for dx,dy in c:
                x,y=seq[l][0]+dx,seq[l][1]+dy
                if (self.sp_map[x][y] == 0 && f[x][y] + 1e-5 > f[seq[l][0]][seq[l][1]] + self.sp_value[x][y]):
                    f[x][y] = f[seq[l][0]][seq[l][1]] + self.sp_value[x][y]
                    self.sp_last[x][y]=seq[l]
                    self.sp_ff[x][y]=1
                    if (self.sp_flag[x][y]==0):
                        self.sp_flag[x][u]=1
                        r+=1
                        seq[r]=[x,y]
            self.sp_flag[seq[l][0]][seq[l][1]]=0

    def sp_follow_the_road(self,p_begin,p_end):
        map_width, map_length=810,510

        self.sp_calc(p_begin)
        z,d=[],0
        def dfs(x,y):
            global z,d
            if (self.sp_last[i][j][1]!=0):dfs(last[x][y][0],last[x][y][1])
            d+=1
            z[d]=(x,y)
        if (ff[p_end[0]][p_end[1]]==0):
            return "No such path"
        dfs(p_end[0],p_end[1])
        x,y,xx,yy,now_x,now_y=0,0,[0]*map_width*map_length,[0]*map_width*map_length,z[1][0],z[1][1]
        zz=[(now_x,now_y)]
        for i in range(2,d+1):
            if (z[i][0]-z[i-1][0]!=0):
                x+=z[i][0]-z[i-1][0]
                if (y!=0):
                    now_x,now_y=now_x+x,now_y+y
                    zz+=[(now_x,now_y)]
                    x,y=0,0
            if (z[i][1]-z[i-1][1]!=0):
                y+=z[i][1]-z[i-1][1]
                if (x!=0):
                    now_x,now_y=now_x+x,now_y+y
                    zz+=[(now_x,now_y)]
                    x,y=0,0
        if (x!=0||y!=0)
            now_x,now_y=now_x+x,now_y+y
            zz+=[(now_x,now_y)]
            x,y=0,0
        for x,y in zz:print(x,y)
        self.sp_route=np.array(zz)
        return zz

    def sp_calc_Penalty_value(self,car_th):
        def norm(x, y):
            return sqrt(x * x + y * y)
        def calc_angle(a,b,c):
            x,y=b[0]-a[0],b[1]-a[1]
            xx,yy=c[0]-a[0],c[1]-a[1]
            return math.acos((x*xx+y*yy)/norm(x,y)/norm(xx,yy))

        n=self.sp_route.shape[0]
        if (self.sp_tag==1):
            for i in range(n-1,1,-1):
                angle=calc_angle(sp_route[i],sp_route[i+1],sp_route(i-1))
                self.sp_Penalty_value[i]=angle/self.sp_value[self.sp_route[i][0]][self.sp_route[i][1]]*1000
                if (i!=n-1):
                    k=self.sp_Penalty_value[i+1]-norm(sp_route[i][0]-sp_route[i+1][0],sp_route[i][1]-sp_route[i+1][1])
                    self.sp_Penalty_value[i]=min(self.sp_Penalty_value[i],k)
        return self.sp_Penalty_value[sp_route_th]\
               -norm(cars[car_th][1]-sp_route[sp_route_th+1][0],cars[car_th][2]-sp_route[sp_route_th+1][1])

    def sp_speed_changing(self,car_th=0,p_target_local=None):
        def norm(x,y):
            return sqrt(x*x+y*y)

        Penalty_value=self.calc_Penalty_value(car_th)
        #惩罚值越高代表全局地图在当前节点越难走，和最近的那个拐弯的角度和拐弯点的costmap有关
        max_speed=min(3,max(p_target_local[2],1/Penalty_value))#计算最大速度
        PI=math.acos(-1)

        car_x,car_y=self.cars[car_th][1],self.cars[car_th][2]
        car_theta=self.cars[car_th][3]*PI/180
        #小车当前的位置和底盘角度

        dx,dy=p_target_global[0]-car_x,p_target_global[1]-car_y
        vx,vy=self.acts[car_th][1],self.acts[car_th][2]
        cos_theta,sin_theta=math.cos(car_theta),math.sin(car_theta)
        new_dx,new_dy=dx*cos_theta+dy*sin_theta,dy*cos_theta-dx*sin_theta
        new_vx=new_dx/norm(new_dx,new_dy)*max_speed
        new_vy=new_dy/norm(new_dx,new_dy)*max_speed
        dvx,dvy=(new_vx-vx),(new_vy-vy)
        #获得在当前小车参考系下的新速度

        if (dvx<self.sp_v_lowerbound):orders[car_th][0]=1
        if (dvx>self.sp_v_upperbound):orders[car_th][0]=-1
        if (dvy<self.sp_v_lowerbound):orders[car_th][1]=1
        if (dvy>self.sp_v_upperbound):orders[car_th][1]=-1
        #如果超过了某个阈值，就启动加速/减速

        delta_theta=math.asin((vx*new_vy-vy*new_vx)/norm(vx,vy)/norm(new_vx,new_vy))
        cos_d_theta, sin_d_theta=math.cos(delta_theta),math.sin(delta_theta)
        #计算改变的角度
        if (cos_d_theta*cos_theta-sin_d_theta*sin_theta!=new_vx/norm(new_vx,new_vy)or
            sin_d_theta*cos_theta+cos_d_theta*sin_theta!=new_vy/norm(new_vx,new_vy)):
            if (delta_theta>0):delta_theta-=PI
            else:delta_theta+=PI
        #处理答案不在+-Pi/2里的情况
        delta_theta=fmod(delta_theta,PI/2)

        if (delta_theta<self.sp_angle_lowerbound):orders[car_th][2]=-1
        if (delta_theta>self.sp_angle_upperbound):orders[car_th][2]=1
        #如果超过了某个阈值，就启动底盘加速/减速

    def sp_RVO(self,car_th=0,p_target_local=None):#20fps
        def calc_V(vx,vy,theta):
            cos_theta=math.cos(theta)
            sin_theta=math.sin(theta)
            vx_new=cos_theta*vx-sin_theta*vy
            vy_new=sin_theta*vx+cos_theta*vy
            return [vx_new,vy_new]
        def norm(x,y):
            return math.sqrt(x*x+y*y)
        #调用RVO模块
        from RVO import RVO_update, reach, compute_V_des, reach, Tools
        from vis import visualize_traj_dynamic
        #将障碍物和小车都转换成圆形的物体输入
        ws_model = dict()
        ws_model['robot_radius'] = 0.4
        ws_model['circular_obstacles'] =self.sp_circular_obstacles
        ws_model['boundary'] = [400,250,400,250]
        X=[]
        for i in range(4):X+=[[cars[i][1],cars[i][2]]]
        goals=[[0,0],[0,0],[0,0],[0,0]]
        V_max=[3]*4
        #goals[car_th]=[self.sp_route[self.sp_route_th][0],self.sp_route[self.sp_route_th][1]]
        goals[car_th]=p_target_local
        V=[]
        for i in range(4):V+=[calc_V(acts[i][2],acts[i][3],cars[i][3])]
        V_des = compute_V_des(X, goal, V_max)
        for i in range(4):
            if (i!=car_th):V_des[i]=V[i]
        V = RVO_update(X, V_des, V, ws_model,local=2)
        p_target_local=V[car_th]
        p_target_local[0]+=cars[car_th][1]
        p_target_local[1]+=cars[car_th][2]
        p_target_local+=[norm(V[car_th][0],V[car_th][1])]
        return V[car_th]


    def sp_follower(self,car_th=0,p_target_global=None):#20fps
        def clc(x,y):#算范数的平方
            return (x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])

        #如果没有输入就随机一个初始节点
        if (p_target_global==None):p_target_global=lf.sp_target_global
        else:
            self.sp_target_global=p_target_global
            self.sp_tag=0
        if (p_target_global == None or clc(p_target_global, cars[car_th]) < 100):
            self.sp_target_global = p_target_global = (int(random() * 500) + 1, int(random() * 800) + 1)
            print("we will get:", p_target_global[0], p_target_global[1])
            self.sp_tag = 0

        if (self.sp_tag%20==0):#每隔20帧（也就是1秒）重新算一次最短路，保证最短路地图的有效性
            #recalculate the shortest path
            # 1 time/s
            self.sp_init(car_th)
            self.sp_follow_the_road((cars[car_th][2], cars[car_th][1]), p_target_global)

        #if still got the goal, then goto the next goal
        while (clc(self.sp_route[self.sp_route_th],cars[car_th])<100):
            self.sp_route_th+=1

        #use RVO to changing the speed
        self.sp_tag+=1
        p_target_RVO=self.sp_RVO(car_th,sp_route[self.sp_route_th])
        #(0,1)target point;(2) target velocity

        self.sp_speed_changing(car_th,p_target_RVO)