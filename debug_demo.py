from debug_kernel import kernal
env= kernal(car_num = 2, render = True, record = False)
env.reset()
env.eachstep()  