import numpy as np
import random
import math
from PIL import Image

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

# spheres = np.array( [ [74,45,100,4] ]).astype( np.float32 )

rasterizer = Sphere_Rasterizer( spheres, (850,850), 2, 64);
print("computing intersections")
hit_buffer = rasterizer.rasterize_spheres(0)
print("done")
print( hit_buffer[45,74,:] )

for z in range(hit_buffer.shape[2]):
    entry_0 = hit_buffer[:,:,z].astype(np.uint16)
    img = Image.fromarray(entry_0, "I;16")
    img.save("output/entry_exit_"+str(z)+".tif");
  
print("resolving CSG")
csg_resolver = CSG_Resolver( hit_buffer, 2 );
extended_heightfield = csg_resolver.resolve_csg( 0.0 )
print("done")

print(extended_heightfield[45,74,:])