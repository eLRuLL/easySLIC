#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>

#include <stdio.h>
#include <float.h>

#include "superpixels.h"

cudaGraphicsResource *in_resource;
cudaGraphicsResource *out_resource;
uchar3* in_image;
uchar3* out_image;

__constant__ int img_width;
__constant__ int img_height;

__device__ int2 find_local_minimum( uchar3 *image, int2 center, int index ){
    int i,j;
    float min_grad = FLT_MAX;
    int2 loc_min = center;

    for (i = center.x-1; i < center.x+2; ++i) {
        for (j = center.y-1; j < center.y+2; ++j) {
          /* get L values. */
          unsigned char i1 = image[index+1].x;
          unsigned char i2 = image[index+img_width].x;
          unsigned char i3 = image[index].x;

          /* Compute horizontal and vertical gradients and keep track of the
          minimum. */
          if (sqrtf(powf(i1 - i3, 2)) + sqrtf(powf(i2 - i3,2)) < min_grad) {
            min_grad = fabsf(i1 - i3) + fabsf(i2 - i3);
            loc_min.x = i;
            loc_min.y = j;
          }
        }
      }

      return loc_min;
}

// __global__ void init_data( uchar3* image ){
//     int i,j;
// }

// __global__ void generate_superpixels(){

// }

__global__ void redkernel( uchar3 *in_image, uchar3 *out_image ) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    out_image[offset].x = 255;
    out_image[offset].y = 0;
    out_image[offset].z = 0;

}

__global__ void GPU_invert( uchar3 *in_image, uchar3 *out_image ) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	out_image[offset].x = 255 - in_image[offset].x;
	out_image[offset].y = 255 - in_image[offset].y;
	out_image[offset].z = 255 - in_image[offset].z;

}

void interop_setup(int M, int N) {
	cudaDeviceProp prop;
	int dev;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice( &dev, &prop );
	cudaGLSetGLDevice( dev );	// dev = 0
	cudaMemcpyToSymbol(img_width,&M,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(img_height,&N,1*sizeof(int),0,cudaMemcpyHostToDevice);
}

void interop_register_buffer(GLuint& in_buffer, GLuint& out_buffer){
	cudaGraphicsGLRegisterBuffer( &in_resource,in_buffer,cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer( &out_resource,out_buffer,cudaGraphicsMapFlagsNone);
}

void interop_map() {
	size_t in_size, out_size;

	cudaGraphicsMapResources( 1, &in_resource, NULL );
	cudaGraphicsResourceGetMappedPointer( (void**)&in_image,&in_size,in_resource ) ;

	cudaGraphicsMapResources( 1, &out_resource, NULL );
	cudaGraphicsResourceGetMappedPointer( (void**)&out_image,&out_size,out_resource ) ;
}

void interop_run(int M, int N) {

	dim3 grids(N,M);
	dim3 threads(1,1);

	redkernel<<<grids,threads>>>( in_image, out_image);
	cudaGraphicsUnmapResources( 1, &in_resource, NULL );
	cudaGraphicsUnmapResources( 1, &out_resource, NULL );

}

void interop_cleanup(){
	cudaGraphicsUnregisterResource( in_resource );
	cudaGraphicsUnregisterResource( out_resource );
}