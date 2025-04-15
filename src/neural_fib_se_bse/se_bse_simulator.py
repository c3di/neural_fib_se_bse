import numpy as np

from neural_fib_se_bse.preprocess_module import *
from neural_fib_se_bse.boolean_model import BooleanModel
from neural_fib_se_bse import neural_model
from neural_fib_se_bse.data_representation import *

class SE_BSE_Simulator():
    def __init__( self, output_size = (512, 512) ):
        self.output_size = output_size
        self.preprocessor = HeightFieldExtractor( self.output_size, 2, 64 )
        self.neural_model = neural_model.NeuralModel( self.output_size )
        self.statistical_geometry_model = None
        self.extended_heightfield = None
        self.normal_map = None

    def create_image( self, image_plane = 0.0 ):
        if self.statistical_geometry_model is not None:
            self.add_statistical_geometry()
        self.extended_heightfield, self.normal_map = self.preprocessor.extract_data_representation( image_plane )
        exthf_encoded, normal_encoded = self.encode_extfh_and_normal( )
        prediction = self.neural_model.forward(exthf_encoded[:,0:4,:,:], normal_encoded)
        
        return self.prediction_to_cpu( prediction )
        
    def prediction_to_cpu( self, prediction ):
        se,bse = prediction
        se = se.detach().numpy()
        se = se.squeeze(0).squeeze(0)
        bse = bse.detach().numpy()
        bse = bse.squeeze(0).squeeze(0)
        return se, bse
        
    def add_statistical_geometry( self, geometry_model ):
        geometry_model.generate()
        
        spheres = extract_sphere_data( geometry_model )
        if spheres.shape[0] > 0:
            self.preprocessor.add_spheres( spheres )        
        
        cylinders = extract_cylinder_data( geometry_model )
        if cylinders.shape[0] > 0:
            self.preprocessor.add_cylinders( cylinders )           
        
        cuboids = extract_cuboid_data( geometry_model )
        if cuboids.shape[0] > 0:
            self.preprocessor.add_cuboids( cuboids )          
        
    def encode_extfh_and_normal( self ):
        exthf_encoded, normal_encoded = preprocess_to_neural( self.extended_heightfield, self.normal_map )
        exthf_encoded.to('cuda')
        normal_encoded.to('cuda')
        return exthf_encoded, normal_encoded