from debug_kernel import kernal
import random
'''
env1 = kernal(car_num = 1, render = True, record=False,robot_id=1)
env2 = kernal(car_num = 1, render = True, record=False,robot_id=2)
'''
env1 = kernal(car_num = 1, render = True, record = False, robot_id = 1)

def demo():
    total_reward = 0
    steps = 0
    rc = random.choice
    state = env1.reset()
    while True:
        if not env1.epoch % 10:
            env1.action = env1.get_order()
        #env1_action[env1.n] = [rc([1,-1]),rc([1,-1]),rc([1,-1]),rc([1,-1]),rc([1,0]),rc([1,0]),rc([1,0])] # random policy
        #env2_action = PPO.act(state)
        state, reward, done, {} = env1.step(action)
        total_reward += reward
        if steps % 20 == 0 or done:
           print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            print(total_reward) 
            return total_reward
        

if __name__ == '__main__':
    demo()  