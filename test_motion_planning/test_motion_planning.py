from motion import kernal

test_game = kernal(car_num = 2, render = True)
test_game.reset()
test_game.play()

"""
import ctypes
_mod = ctypes.cdll.LoadLibrary("/home/cogito/ICRA-simulator/test_motion_planning/sp.so")
obj=_mod.create_SPFA(500,800)
#_mod.open_debug_modle()
_mod.set_begin_end(1,1,1,800)
_mod.add_obstables(1,499,400,400)
print(_mod.calc_SPFA())
arr=(ctypes.c_int*2000)()
x=_mod.smooth(arr)
print(x)
for i in range(x):print(arr[i*2+1],arr[i*2+2])
"""