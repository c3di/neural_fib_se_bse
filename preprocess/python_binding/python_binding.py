import numpy as np
import random
import math

from extended_heightfield import Sphere_Rasterizer_Kernel

print("import successful")

spheres = np.array( [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]] )
kernel = Sphere_Rasterizer_Kernel( spheres, (512,512), 4 )