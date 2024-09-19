# Neural Focussed Ion Beam Simulator
This python package implements a surrogate model to approximately replicate the 
Monte-Carlo simulations performed to simulate scanning 
electron microscopy imaging. Our model accepts threedimensional microstructure representations of porous 
materials in the form of lists of primitives. It converts them to 
a specific data representation suitable for a neural network. 
A convolutional architecture generates two-dimensional 
electron microscopy images in a single forward pass. The 
model performs well on arbitrary microstructures like 
systems of cubes, even though it was trained on structures 
consisting of spheres and cylinders only.
