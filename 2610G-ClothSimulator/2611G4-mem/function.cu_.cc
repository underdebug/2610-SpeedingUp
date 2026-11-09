#include "function.h"
#include <cstdio>
#include "kernel.h"
#include "vector"

// std::vector<dim3> makeDim(dim3 grid, dim3 block)
// {
//     dim3 D()
//     std::vector<dim3> 
// }

int main()
{
    // int result[10];

    // blockIdx.x = 0;
    // blockDim = {10, 1, 1};
    // threadIdx.x = 5;
    // modify(result, 5);
    
    // dim3 grid(2, 3, 4);
    // dim3 block(8, 8, 8);
    // std::vector<dim3> threads = makeDim();

    // dim3 dim = grid * block;
    // for(int dz = 0; dz < dim.z / 2; dz++){
    //     double z1 = dim.z / 2 - 
    //     for(int dy = 0; dy < dim.y / 2; dy++){
    //         for(int dx = 0; dx < dim.x / 2; dx++){


    // }

    int result[10];
    add(2, 3, result);
    
    return 1;
}
