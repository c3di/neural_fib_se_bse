#include "ScreenGrid.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

static inline int clamp_i(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

template<class Primitive>
ScreenGrid build_screen_grid_cpu(const std::vector<Primitive>& primitives_cpu,
    int2 output_resolution,
    int  tile_size)
{
    
    ScreenGrid grid;
    
    grid.tile_size = tile_size;

    grid.tile_dim.x = (output_resolution.x + tile_size - 1) / tile_size;
    grid.tile_dim.y = (output_resolution.y + tile_size - 1) / tile_size;
    grid.n_tiles = grid.tile_dim.x * grid.tile_dim.y;

    const int n_primitives = (int)primitives_cpu.size();

    std::vector<int> tile_counts(grid.n_tiles, 0);

    for (int pid = 0; pid < n_primitives; pid++)
    {
        const float4 saabb = get_screen_aabb(primitives_cpu[pid]);
        const float cx = saabb.x, cy = saabb.y, hx = saabb.z, hy = saabb.w;

        int min_tx = clamp_i((int)std::floorf((cx - hx) / tile_size), 0, grid.tile_dim.x - 1);
        int max_tx = clamp_i((int)std::floorf((cx + hx) / tile_size), 0, grid.tile_dim.x - 1);
        int min_ty = clamp_i((int)std::floorf((cy - hy) / tile_size), 0, grid.tile_dim.y - 1);
        int max_ty = clamp_i((int)std::floorf((cy + hy) / tile_size), 0, grid.tile_dim.y - 1);

        for (int ty = min_ty; ty <= max_ty; ty++)
            for (int tx = min_tx; tx <= max_tx; tx++)
                tile_counts[ty * grid.tile_dim.x + tx]++;
    }

    std::vector<int> tile_offsets(grid.n_tiles + 1, 0);
    for (int t = 0; t < grid.n_tiles; t++)
        tile_offsets[t + 1] = tile_offsets[t] + tile_counts[t];

    grid.total_insertions = tile_offsets[grid.n_tiles];

    std::vector<int> tile_prim_list(grid.total_insertions, 0);

    std::fill(tile_counts.begin(), tile_counts.end(), 0);

    for (int pid = 0; pid < n_primitives; pid++)
    {
        const float4 saabb = get_screen_aabb(primitives_cpu[pid]);
        const float cx = saabb.x, cy = saabb.y, hx = saabb.z, hy = saabb.w;

        int min_tx = clamp_i((int)std::floorf((cx - hx) / tile_size), 0, grid.tile_dim.x - 1);
        int max_tx = clamp_i((int)std::floorf((cx + hx) / tile_size), 0, grid.tile_dim.x - 1);
        int min_ty = clamp_i((int)std::floorf((cy - hy) / tile_size), 0, grid.tile_dim.y - 1);
        int max_ty = clamp_i((int)std::floorf((cy + hy) / tile_size), 0, grid.tile_dim.y - 1);

        for (int ty = min_ty; ty <= max_ty; ty++)
        {
            for (int tx = min_tx; tx <= max_tx; tx++)
            {
                int tile_id = ty * grid.tile_dim.x + tx;
                int insert_pos = tile_offsets[tile_id] + tile_counts[tile_id];
                tile_prim_list[insert_pos] = pid;
                tile_counts[tile_id]++;
            }
        }
    }

    cudaError_t error = cudaMalloc((void**)&grid.tile_offsets_gpu,
        sizeof(int) * (grid.n_tiles + 1));
    if (error != cudaSuccess) {
        std::string msg = "Failed to allocate memory on device: ";
        msg += cudaGetErrorString(error);
        throw std::runtime_error(msg);
    }
    error = cudaMemcpy(grid.tile_offsets_gpu, tile_offsets.data(),
        sizeof(int) * (grid.n_tiles + 1), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::string msg = "Failed copy the data to the device: ";
        msg += cudaGetErrorString(error);
        throw std::runtime_error(msg);
    }

    if (grid.total_insertions > 0)
    {
        cudaError_t error = cudaMalloc((void**)&grid.tile_prim_list_gpu,
            sizeof(int) * grid.total_insertions);
        if (error != cudaSuccess) {
        std::string msg = "Failed to allocate memory on device: ";
        msg += cudaGetErrorString(error);
        throw std::runtime_error(msg);
         }

        error = cudaMemcpy(grid.tile_prim_list_gpu, tile_prim_list.data(),
            sizeof(int) * grid.total_insertions, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
        std::string msg = "Failed copy the data to the device: ";
        msg += cudaGetErrorString(error);
        throw std::runtime_error(msg);
    }

    }
    else
    {
        grid.tile_prim_list_gpu = nullptr;
    }
    
    return grid;
    
}

