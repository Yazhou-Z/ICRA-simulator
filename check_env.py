from kernel_debug import kernal
from stable_baselines3.common.env_checker import check_env
env = kernal(car_num = 1, robot_id = 1, render = True, record = False)
check_env(env, warn=True)