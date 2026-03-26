#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "sphere.h"
#include "cylinder.h"
#include "cuboid.h"


struct ScreenGrid
{
    int2 tile_dim;           // number of tiles in x and y
    int  tile_size;          // pixels per tile edge (16)
    int  n_tiles;            // tile_dim.x * tile_dim.y
    int  total_insertions;   // total entries in tile_prim_list

    int* tile_offsets_gpu;   // [n_tiles + 1]  prefix-sum (CSR row pointers)
    int* tile_prim_list_gpu; // [total_insertions] primitive indices

    ScreenGrid()
        : tile_dim(make_int2(0, 0)), tile_size(16), n_tiles(0)
        , total_insertions(0)
        , tile_offsets_gpu(nullptr), tile_prim_list_gpu(nullptr)
    {
    }
};

// Free GPU memory owned by a ScreenGrid.
inline void free_screen_grid(ScreenGrid& grid)
{
    if (grid.tile_offsets_gpu) { cudaFree(grid.tile_offsets_gpu);   grid.tile_offsets_gpu = nullptr; }
    if (grid.tile_prim_list_gpu) { cudaFree(grid.tile_prim_list_gpu); grid.tile_prim_list_gpu = nullptr; }
}

inline float4 get_screen_aabb(const Sphere& p)
{
    return make_float4(p.position.x, p.position.y, p.r, p.r);
}

inline float4 get_screen_aabb(const Cylinder& p)
{
    return make_float4(p.position.x, p.position.y, p.aabb.x, p.aabb.y);
}

inline float4 get_screen_aabb(const Cuboid& p)
{
    return make_float4(p.position.x, p.position.y, p.aabb.x, p.aabb.y);
}

template<class Primitive>
ScreenGrid build_screen_grid_cpu(const std::vector<Primitive>& primitives_cpu,
    int2 output_resolution,
    int  tile_size = 16);



