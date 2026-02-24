# Required PyBind Interfaces

## Transfer Data to Cpp

- Interface to transfer a 3d-Array <- Volume container that holds all volumes

### CPP

Impl for a np 3d-Array with shape information

## Generate Extended Heightfield and NormalMap

- Same Interface as the existing impl -> parameter for the image plane & return the extended heightfield and normal map

## Bug fixes

- Every intersector has its own z-buffer that tries to ensure that we write the normal map with depth information but since every intersector has its own buffer, it is possible to overwrite old normal map values -> Solution: not only share the normal map but also the z-buffer

- `HeightFieldExtractor` returns only the first normal map of the 

## Questions

- How to guarantee that user only uses `VolumeContainer` for adding shapes in python?
- Do we need an intersector class if user can only add shapes with the `VolumeContainer`?
- Why do we only return the normal map from the first intersector in `HeightFieldExtractor::extract_data_representation_py`?

## Python

- [ ] Create `VolumeContainer` class that holds all volumes in one numpy 3d-array, ensure dimensions are at most the image resolution sizes (where?)
- [ ] Load all volumes in `VolumeContainer`
- [ ] Add `VolumeContainer` to `HeightFieldExtractor`

## Cpp

- [ ] Create `VolumeContainer` class with `container` and `shape` member variables
- [ ] Export `VolumeContainer` with pybind
- Do we need an intersector?

## CUDA

- [ ] 
