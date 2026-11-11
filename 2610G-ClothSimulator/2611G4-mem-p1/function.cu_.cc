
#include "function.h"
#include <cstdio>

// void Modify(int* data, int n){
//     dim3 gridDim(2, 3, 4);
//     dim3 blockDim(8, 8, 8);
//     dim3 o = gridDim / 2;

//     for (unsigned int z = 0; z < gridDim.z; ++z)
//         for (unsigned int y = 0; y < gridDim.y; ++y)
//             for (unsigned int x = 0; x < gridDim.x; ++x)
//                 for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z)
//                     for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y)
//                         for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
//                             blockIdx = dim3((x + o.x) % gridDim.x, (y + o.y) % gridDim.y, (z + o.z) % gridDim.z);
//                             modify(data, 5);
//                         }
// }

int main()
{  
    int data[5] = {0};
    //Modify(data, 5);

    modify<<<dim3(4,3,2), dim3(3,2,1)>>>(data, 5); 
    
    int result[10];
    add(2, 3, result);
    
    return 1;
}
