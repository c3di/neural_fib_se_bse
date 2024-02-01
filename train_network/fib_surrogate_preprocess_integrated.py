import sys
import os
import pathlib
import argparse

import tifffile
import numpy as np
import numpy.lib.recfunctions as rf
import scipy.spatial.transform.Rotation

import preprocess.extended_heightfield

data_representation = []
config_path = pathlib.Path('config_data/')

file_names = os.listdir(config_path)

def read_noncomment_line( file ):
    found = False
    while not found:
        line = file.readline()
        if not line.startswith("#"):
            found = True
    return line    

def read_int( file ):
    return int( read_noncomment_line( file ) )

def read_sphere( file ):
    tokens = read_noncomment_line( file ).split()    
    return ( int(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]) )

def read_cylinder( file ):
    tokens = read_noncomment_line( file ).split()    
    return ( int(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7]) )

def read_cubes( file ):
    tokens = read_noncomment_line( file ).split()    
    return ( int(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[5]), float(tokens[6]) )

def sphere_data_to_numpy( spheres ):
    spheres_np        = np.zeros( (len(spheres), 4), dtype=np.float32 )
    for i,sphere in enumerate(spheres):
        _,x,y,z,r = sphere
        spheres_np[i,0] = x + 0.5
        spheres_np[i,1] = y + 0.5
        spheres_np[i,2] = z + 0.5
        spheres_np[i,3] = r
    return spheres_np

def cylinder_data_to_numpy( cylinders ):
    cylinders_np        = np.zeros( (len(cylinders), 9), dtype=np.float32 )
    for i,cylinder in enumerate( cylinders ):
        _,x,y,z,euler1,euler2,euler3,r,l = cylinder
        cylinders_np[i,0] = x + 0.5
        cylinders_np[i,1] = y + 0.5
        cylinders_np[i,2] = z + 0.5
        rotation_matrix = Rotation.from_euler("ZXZ", [euler1,euler2,euler3])
        cylinders_np[i,3] = angle0
        cylinders_np[i,4] = angle1
        cylinders_np[i,5] = angle2
        cylinders_np[i,6] = angle3
        cylinders_np[i,7] = r
        cylinders_np[i,8] = l
    return cylinders_np

output_size = 512

for filename in file_names:
    file = open( config_path/filename, mode = 'r', encoding = 'utf-8' )
    
    num_spheres   = read_int(file)
    num_cylinders = read_int(file)
    num_cubes     = read_int(file)

    spheres = []
    cylinders = []
    cubes = []

    for i in range( num_spheres ):
        spheres.append( read_sphere(file) )
    for i in range( num_cylinders ):
        cylinders.append( read_cylinder(file) )
    for i in range( num_cubes ):
        cubes.append( read_cubes(file) )

    preprocessor = preprocess.extended_heightfield.HeightFieldExtractor( (output_size, output_size), 2, 256 )

    if num_spheres > 0:
        spheres_np = sphere_data_to_numpy( spheres )
        preprocessor.add_spheres( spheres_np )
    
    if num_cylinders > 0:
        cylinders_np = cylinder_data_to_numpy( cylinders )
        preprocessor.add_cylinders( cylinders_np )

    extended_heightfield, normal_map = preprocessor.extract_data_representation( 0.0 )
    extended_heightfield = extended_heightfield.reshape( ( output_size, output_size, 4 ) )

    extended_heightfield[ extended_heightfield > 256.0 ] = 256.0
    normal_map = normal_map.squeeze( 2 )
    normal_map = rf.structured_to_unstructured( normal_map );

    normal_map = ( normal_map + 1.0 ) * 127.5

    tifffile.imwrite( filename + "_normal_map.tif", normal_map.astype( np.uint8 ), photometric='rgb')

