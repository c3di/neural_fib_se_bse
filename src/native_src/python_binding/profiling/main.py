import numpy as np
from PIL import Image
import random
import numpy.lib.recfunctions as rf
from scipy.spatial.transform import Rotation

from _preprocess_module import HeightFieldExtractor

RESOLUTION_X = 1000
RESOLUTION_Y = 1000

N_CYLINDERS = 2000
N_CUBOIDS = 2000
N_SPHERES = 2000


def random_cyclic_idx(values, idx):
    random_idx = idx + random.randint(0, 10)
    return values[random_idx % len(values)]


cylinders = np.empty((N_CYLINDERS, 9), dtype=np.float32)
cuboids = np.empty((N_CUBOIDS, 10), dtype=np.float32)
spheres = np.empty((N_CUBOIDS, 4), dtype=np.float32)

rotationX = Rotation.from_rotvec([45.0, 0.0, 0.0], degrees=True)
rotationY = Rotation.from_rotvec([0.0, 45.0, 0.0], degrees=True)
rotationXZ = Rotation.from_rotvec([45.0, 0.0, 45.0], degrees=True)
rotationYZ = Rotation.from_rotvec([0.0, 45.0, 45.0], degrees=True)
qx = rotationX.as_quat()
qy = rotationY.as_quat()
qxz = rotationXZ.as_quat()
qyz = rotationYZ.as_quat()
possible_rotations = [qx, qy, qxz, qyz]

possible_xs = [50, 25, 70, 90, 15, 10, 35]
possible_ys = [30, 60, 20, 45, 15, 85]
possible_zs = [500, 1000, 750, 250]
possible_diameters = [20, 10, 15]
possible_widths = [10, 15, 20]

for i in range(N_CYLINDERS):
    diameter = random_cyclic_idx(possible_diameters, i)
    width = random_cyclic_idx(possible_widths, i)
    x = random_cyclic_idx(possible_xs, i) * i % RESOLUTION_X
    y = random_cyclic_idx(possible_ys, i) * i % RESOLUTION_Y
    z = random_cyclic_idx(possible_zs, i)
    rotation = random_cyclic_idx(possible_rotations, i)

    cylinders[i] = [
        x,
        y,
        z,
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
        diameter,
        width,
    ]

possible_xs = [10, 60, 25, 15, 40, 25, 75]
possible_ys = [40, 35, 55, 80, 65, 20, 45, 15]
possible_zs = [1250, 1000, 1500]
possible_widths = [10, 15, 5]
possible_heights = [15, 10, 20, 10]
possible_depths = [10, 5, 15]

for i in range(N_CUBOIDS):
    width = random_cyclic_idx(possible_widths, i)
    height = random_cyclic_idx(possible_heights, i)
    depth = random_cyclic_idx(possible_depths, i)
    x = random_cyclic_idx(possible_xs, i) * i % RESOLUTION_X
    y = random_cyclic_idx(possible_ys, i) * i % RESOLUTION_Y
    z = random_cyclic_idx(possible_zs, i)
    rotation = random_cyclic_idx(possible_rotations, i)

    cuboids[i] = [
        x,
        y,
        z,
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
        width,
        height,
        depth,
    ]

possible_xs = [45, 30, 60, 10, 85, 50, 20]
possible_ys = [25, 90, 10, 20, 35, 65, 15, 55]
possible_zs = [750, 500, 1000, 1250, 1500]
possible_diameters = [15, 20, 30, 25]

for i in range(N_SPHERES):
    diameter = random_cyclic_idx(possible_diameters, i)
    x = random_cyclic_idx(possible_xs, i) * i % RESOLUTION_X
    y = random_cyclic_idx(possible_ys, i) * i % RESOLUTION_Y
    z = random_cyclic_idx(possible_zs, i)

    spheres[i] = [x, y, z, diameter]

preprocessor = HeightFieldExtractor((RESOLUTION_X, RESOLUTION_Y), 2, 256)

if N_SPHERES > 0:
    preprocessor.add_spheres(spheres)

if N_CYLINDERS > 0:
    preprocessor.add_cylinders(cylinders)

if N_CUBOIDS > 0:
    preprocessor.add_cuboids(cuboids)

extended_heightfield, normal_map = preprocessor.extract_data_representation(0.0)


for z in range(extended_heightfield.shape[2]):
    # entry_0 = rf.structured_to_unstructured(extended_heightfield[:,:,z]);
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
