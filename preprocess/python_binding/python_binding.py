import numpy as np
import random
import math
from PIL import Image
import time

print("attempting import")

from extended_heightfield import HeightFieldExtractor
from extended_heightfield import Sphere_Rasterizer
from extended_heightfield import Cylinder_Rasterizer
from extended_heightfield import CSG_Resolver

print("import successful")

# spheres = np.array( [ [8.0,4.0,4.0,2.0], [8.0,4.0,1.0,2.0], [8.0,4.0,21.0,2.0], [8.0,4.0,16.0,2.0], [8.0,4.0,0.0,2.0], [8.0,4.0,6.0,2.0], [8.0,4.0,18.5,2.0], [8.0,4.0,256,2.0], [8.0,4.0,512,2.0] ] )

# filename =  "data/Sphere_Vv03_r10-15_Num1_ITWMconfig.txt"
filename = "data/Cylinder_Vv03_r5-10_h100-150_homogen_ITWMconfig.txt"
file = open(filename)

def read_primitives( file, primitive_size, n_items ):
    primitives = np.empty( (n_items, primitive_size), dtype=np.float32 )
    for count in range( n_items ):
        line = next(file)
        items = line.split("\t")
        i = int(items[0])
        for j in range(1, primitive_size+1):
            primitives[int(i)-1,j-1] = float( items[j] )
    return primitives

n_spheres   = int( next(file) )
n_cylinders = int( next(file) )

spheres   = read_primitives(file, 4, n_spheres);
cylinders = read_primitives(file, 8, n_cylinders);

# cylinders = np.empty( (1, 8), dtype=np.float32 )
# cylinders[0] = [425,425,400,0.0,1.57079632679,0,50,150]

print("performing preprocessing")
start = time.perf_counter()
preprocessor = HeightFieldExtractor( (850,850), 2, 256 )
print(spheres.shape)
if n_spheres > 0:
    preprocessor.add_spheres( spheres )
if n_cylinders > 0:
    preprocessor.add_cylinders( cylinders )
extended_heightfield, normal_map = preprocessor.extract_data_representation( 0.0 )
stop = time.perf_counter()
print(f"done preprocessing in {stop-start:0.3f} sec")

for z in range(extended_heightfield.shape[2]):
    entry_0 = extended_heightfield[:,:,z].astype(np.uint16)
    img = Image.fromarray(entry_0, "I;16")
    img.save("output/integrated_"+str(z)+".tif");

normal_map = ( normal_map + 1.0 ) * 127.5
normal = normal_map.astype(np.uint8)
img = Image.fromarray(normal, "RGB")
img.save("output/norrmal.tif");
