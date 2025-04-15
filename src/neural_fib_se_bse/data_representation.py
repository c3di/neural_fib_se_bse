import numpy as np
import numpy.lib.recfunctions as rf
import torch

def extract_sphere_data(generator):
    if generator.Particle_Shape != "Sphere":
        return np.zeros( (0, 4), dtype=np.float32 )
    spheres = np.zeros( (generator._Sampled_Particle_Number, 4), dtype=np.float32 )
    spheres[:,0:3]  = generator._Sampled_Centers + 0.5
    spheres[:,3]    = generator._Sampled_Parameters[:,0]
    return spheres

def extract_cylinder_data(generator):
    if generator.Particle_Shape != "Cylinder":
        return np.zeros( (0, 9), dtype=np.float32 )
    cylinders = np.zeros( (generator._Sampled_Particle_Number, 9), dtype=np.float32 )
    cylinders[:,0:3] = generator._Sampled_Centers + 0.5
    cylinders[:,3:7] = generator._Sampled_Rotations
    cylinders[:,7  ] = generator._Sampled_Parameters[:,0]
    cylinders[:,8  ] = generator._Sampled_Parameters[:,1] * 0.5
    return cylinders
   
def extract_cuboid_data(generator):
    if generator.Particle_Shape != "Cuboid":
        return np.zeros( (0, 10), dtype=np.float32 )
    cuboids = np.zeros( (generator._Sampled_Particle_Number, 10), dtype=np.float32 )
    cuboids[:,0:3 ] = generator._Sampled_Centers + 0.5
    cuboids[:,3:7 ] = generator._Sampled_Rotations
    cuboids[:,7:10] = generator._Sampled_Parameters[:,0:4] * 0.5
    return cuboids   
   
def preprocess_to_visualisation(extended_heightfield, normal_map):
    extended_heightfield[extended_heightfield>256.0] = 256.0

    normal_map = normal_map.squeeze(2)
    normal_map = rf.structured_to_unstructured( normal_map );

    normal_map = ( normal_map + 1.0 ) * 127.5
    normal_map = normal_map.astype(np.uint8)
    
    return extended_heightfield,normal_map
    
def preprocess_to_neural(extended_heightfield, normal_map):
    extended_heightfield[extended_heightfield>256.0] = 256.0
    extended_heightfield = np.transpose(extended_heightfield, (2,1,0) )
    extended_heightfield = np.expand_dims(extended_heightfield, 0)
    
    extended_heightfield = extended_heightfield / 256.0
    extended_heightfield = torch.from_numpy(extended_heightfield)

    normal_map = normal_map.squeeze(2)
    normal_map = rf.structured_to_unstructured( normal_map );

    # normal_map = ( normal_map + 1.0 ) * 127.5
    normal_map = ( normal_map + 1.0 ) / 2.0
    
    normal_map = np.transpose(normal_map, (2,1,0) )
    normal_map = np.expand_dims(normal_map, 0)
    normal_map = torch.from_numpy( normal_map )   
    
    return extended_heightfield,normal_map    