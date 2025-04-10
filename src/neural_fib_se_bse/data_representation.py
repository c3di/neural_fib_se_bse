import numpy as np
import numpy.lib.recfunctions as rf
import torch

def extract_sphere_data(generator):
    spheres = np.zeros( (generator._Sampled_Particle_Number, 4), dtype=np.float32 )
    spheres[:,0:3]  = generator._Sampled_Centers + 0.5
    spheres[:,3]    = generator._Sampled_Parameters[:,0]
    return spheres
    
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

    normal_map = ( normal_map + 1.0 ) * 127.5
    
    normal_map = np.transpose(normal_map, (2,1,0) )
    normal_map = np.expand_dims(normal_map, 0)
    normal_map = ( normal_map + 1.0 ) / 2.0    
    normal_map = torch.from_numpy( normal_map )   
    
    return extended_heightfield,normal_map    