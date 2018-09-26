import numpy as np
import scipy.io as scio


def generate_radio_map(*args):
    if len(args) > 1:
        gridSize = args[0]
    else:
        gridSize = 0.01
    roomX = 20
    roomY = 15
    roomZ = 4
    f = 2400
    apslist = [1, 1, 10, 1, 19, 1, 1, 14, 10, 14, 19, 14]
    apsArray = np.array(apslist).reshape(round(len(apslist)/2), 2)
    print(apsArray.shape[0])
    fingerprint = np.zeros((round(roomX/gridSize) - 1, round(roomY/gridSize) - 1, apsArray.shape[0]))
    print(fingerprint.shape)
    return


def test_var_args(*args):
    if len(args) == 2:
        print(args[0]+args[1])
    else:
        print(args[0])


def test_var_kwargs(**kwargs):
    #定义函数知道key:one two
    if len(kwargs) == 2:
        print(kwargs['one']+kwargs['two'])
    else:
        print(kwargs['one'])



if __name__ == '__main__':
    generate_radio_map()