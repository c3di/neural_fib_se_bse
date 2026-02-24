from create_volume import create_huge_volume, RESOLUTION_X, RESOLUTION_Y, RESOLUTION_Z
from _preprocess_module import HeightFieldExtractor
from PIL import Image
import numpy.lib.recfunctions as rf
import numpy as np


def main():

    volume = create_huge_volume()
    preprocessor = HeightFieldExtractor((RESOLUTION_Y, RESOLUTION_X), 2, 256)

    preprocessor.add_volume(
        volume.container, (RESOLUTION_X, RESOLUTION_Y, RESOLUTION_Z), 0.0625
    )

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


if __name__ == "__main__":
    main()
