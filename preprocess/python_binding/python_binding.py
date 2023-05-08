import numpy as np
import random
import math
from PIL import Image
import time

from extended_heightfield import HeightFieldExtractor
from extended_heightfield import Sphere_Rasterizer
from extended_heightfield import CSG_Resolver

print("import successful")

# spheres = np.array( [ [8.0,4.0,4.0,2.0], [8.0,4.0,1.0,2.0], [8.0,4.0,21.0,2.0], [8.0,4.0,16.0,2.0], [8.0,4.0,0.0,2.0], [8.0,4.0,6.0,2.0], [8.0,4.0,18.5,2.0], [8.0,4.0,256,2.0], [8.0,4.0,512,2.0] ] )

file = open("data/Sphere_Vv03_r10-15_Num1_ITWMconfig.txt")
nspheres = int( next(file) )
spheres = np.empty( (nspheres, 4), dtype=np.float32 )
_ = next(file) # skip empty line

for line in file:
    i, x, y, z, r = line.split("\t")
    spheres[int(i)-1,:] = [float(x),float(y),float(z),float(r)]
file.close()

print("performing preprocessing")
start = time.perf_counter()
preprocessor = HeightFieldExtractor( spheres, (850,850), 2, 64 );
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
