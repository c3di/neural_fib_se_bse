import create_shapes
from create_volume import create_huge_volume, RESOLUTION_X, RESOLUTION_Y, RESOLUTION_Z
from _preprocess_module import HeightFieldExtractor
from PIL import Image
import numpy.lib.recfunctions as rf
import numpy as np


def run_volume():
    volume = create_huge_volume()
    preprocessor = HeightFieldExtractor((RESOLUTION_Y, RESOLUTION_X), 2, 256)

    preprocessor.add_volume(
        volume.container, (RESOLUTION_X, RESOLUTION_Y, RESOLUTION_Z), 0.0625)

    extended_heightfield, normal_map = preprocessor.extract_data_representation(0.0)

    # Save the extended heightfields
    for z in range(extended_heightfield.shape[2]):
        # entry_0 = rf.structured_to_unstructured(extended_heightfield[:,:,z]);
        entry = extended_heightfield[:, :, z]
        entry = entry.astype(np.uint16)
        img = Image.fromarray(entry, "I;16")
        img.save("output/integrated_" + str(z) + ".tif")

    # Convert the first normal map
    normal_map = normal_map.squeeze(2)
    normal_map = rf.structured_to_unstructured(normal_map)
    normal_map = (normal_map + 1.0) * 127.5
    normal = normal_map.astype(np.uint8)
    # print(normal)
    img = Image.fromarray(normal, "RGB")
    img.save("output/normal.tif")
    del preprocessor



def run_shapes():
    spheres, cuboids, cylinders = create_shapes.get_shapes()
    preprocessor = HeightFieldExtractor((create_shapes.RESOLUTION_X, create_shapes.RESOLUTION_Y), 2, 256)

    if create_shapes.N_SPHERES > 0:
        preprocessor.add_spheres(spheres)

    if create_shapes.N_CYLINDERS > 0:
        preprocessor.add_cylinders(cylinders)

    if create_shapes.N_CUBOIDS > 0:
        preprocessor.add_cuboids(cuboids)

    extended_heightfield, normal_map = preprocessor.extract_data_representation(0.0)


    for z in range(extended_heightfield.shape[2]):
        entry = extended_heightfield[:, :, z]
        entry = entry.astype(np.uint16)
        img = Image.fromarray(entry, "I;16")
        img.save("output/integrated_" + str(z) + ".tif")

    normal_map = normal_map.squeeze(2)
    normal_map = rf.structured_to_unstructured(normal_map)
    normal_map = (normal_map + 1.0) * 127.5
    normal = normal_map.astype(np.uint8)
    img = Image.fromarray(normal, "RGB")
    img.save("output/normal.tif")



def main():
    # run_volume()
    run_shapes()
    

if __name__ == "__main__":
    main()
