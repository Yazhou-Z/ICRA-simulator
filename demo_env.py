from kernel import kernal
env = kernal (car_num = 4, render = True, record=False)
env.reset()
a = env.orders
def demo():
    total_reward = 0
    steps = 0
    while True:
        reward, done = env.eachstep()
        total_reward += reward
        if steps % 20 == 0 or done:
           print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: 
            return total_reward
        
if __name__ == '__main__':
    demo()  