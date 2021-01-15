from kernel_for_RL import kernal
import random
env = kernal(car_num = 1, robot_id = 1, render = True, record = False)
def demo():
    total_reward = 0
    state = env.reset()
    steps = 0
    while True:
        if env.get_order():
            return
        action = env.orders
        state, reward, done, info = env.step(action)
        steps += 1
        total_reward += reward
        if steps % 20 == 0 or done:
           print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        if done:
           return total_reward
        
if __name__ =="__main__":
    demo()