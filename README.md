# Neural Focussed Ion Beam Simulator
This python package implements a surrogate model to approximately replicate the 
Monte-Carlo simulations performed to simulate scanning 
electron microscopy imaging. Our model accepts three-dimensional microstructure representations of porous 
materials in the form of lists of primitives. It converts them to 
a specific data representation suitable for a neural network. 
A convolutional architecture generates two-dimensional 
electron microscopy images in a single forward pass. The 
model performs well on arbitrary microstructures like 
systems of cubes, even though it was trained on structures 
consisting of spheres and cylinders only.

The method is described in detail in this publication:

[https://openreview.net/attachment?id=SwO84a6yA5&name=pdf]

## Citing
If you use our package, please cite this paper:
```
@inproceedings{pub15015,
    author = { Dahmen, Tim and Rottmayer, Niklas and Kronenberger, Markus and Schladitz, Katja and Redenbach, Claudia },
    title = {A Neural Model for High-Performance Scanning Electron Microscopy Image Simulation of Porous Materials},
    booktitle = {Synthetic Data for Computer Vision Workshop @ CVPR 2024. CVPR Workshop on Synthetic Data for Computer Vision (SynData4CV-2024), befindet sich CVPR 2024, June 18-18, Seattle, OR, United States},
    year = {2024},
    month = {6},
    publisher = {CFV}
}
```
