import numpy as np
import random
import math

from extended_heightfield import Sphere_Rasterizer

print("import successful")

spheres = np.array( [ [8.0,4.0,21.0,2.0], [8.0,4.0,16.0,2.0], [8.0,4.0,0.0,2.0], [8.0,4.0,6.0,2.0], [8.0,4.0,18.5,2.0] ] )
kernel = Sphere_Rasterizer( spheres, (32,32), 4 )

print("rasterizing")
result = kernel.rasterize_spheres(0)
print("done")
print(result.shape)

for z in range(result.shape[2]):
    print(result[8,4,z])