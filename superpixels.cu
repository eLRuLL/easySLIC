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
#include <vector>

#include "superpixels.h"


cudaGraphicsResource *in_resource;
cudaGraphicsResource *out_resource;
uchar3* in_image;
uchar3* out_image;

__constant__ int img_width;
__constant__ int img_height;
__constant__ int step;
__constant__ int ncenters;

uchar3 *centers_colors;
int2 *centers_coords;
int *clusters;
float *distances;

__device__ int2 find_local_minimum( uchar3 *image, int2 center){
    int i,j;
    float min_grad = FLT_MAX;
    int2 loc_min = center;

    for (i = center.x-1; i < center.x+2; ++i) {
        for (j = center.y-1; j < center.y+2; ++j) {
          int index = i + j * img_width;
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

__global__ void init_data( uchar3* image , int* clusters, uchar3 *centers_colors, int2 *centers_coords){
    int x = (threadIdx.x + 1)*step;
    int y = (threadIdx.y + 1)*step;

    int2 newc = find_local_minimum(image, make_int2(x,y));
    uchar3 colour = image[x + y*img_width];

    int offset = threadIdx.x + threadIdx.y*blockDim.x;
    centers_colors[offset] = colour;
    centers_coords[offset] = newc;

    image[newc.x + newc.y*img_width].x = 0;
    image[newc.x + newc.y*img_width].y = 0;
    image[newc.x + newc.y*img_width].z = 255;

}

__device__ float compute_dist(uchar3 center_lab, int2 center_coords, int2 pixel, uchar3 colour){
  float dc = sqrtf(powf(center_lab.x - colour.x, 2) + powf(center_lab.y - colour.y, 2) + powf(center_lab.z - colour.z, 2));
  float ds = sqrtf(powf(center_coords.x - pixel.x, 2) + powf(center_coords.y - pixel.y, 2));
  float m = 40.0; // ESTO DEBE SER GLOBAL O PARAM
  float K = 100.0; // TAMBIEN GLOBAL O PARAM (NUMERO DE SUPERPIXELES)
  float N = 100.0; // ACTUALIZAR CON EL NUMERO DE PIXELES (WIDTH * HEIGHT)
  float S_value = sqrt(N/K);

  return dc + (m/S_value)*ds;
}

// remember to reset distances.
__global__ void update_cluster(uchar3 *image, int *clusters, int2 *centers_coords, uchar3* centers_colors, int n){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int index = x + y * blockDim.x * gridDim.x;

  int i;
  int min_cluster_id;
  float min_cluster_distance = FLT_MAX;

  // solo deberia hacerlo con 9 cercanos
  for(i=0;i<ncenters;++i){
    float _distance = compute_dist(centers_colors[i], centers_coords[i], make_int2(x, y), image[index]);
    if(_distance < min_cluster_distance){
      min_cluster_distance = _distance;
      min_cluster_id = i;
    }
  }
  clusters[index] = min_cluster_id;
}




void interop_setup(int M, int N, int h_ncenters, int h_step, int nc) {
	cudaDeviceProp prop;
	int dev;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice( &dev, &prop );
	cudaGLSetGLDevice( dev );	// dev = 0
	cudaMemcpyToSymbol(img_height,&M,1*sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(img_width,&N,1*sizeof(int),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ncenters,&h_ncenters,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(step,&h_step,1*sizeof(int),0,cudaMemcpyHostToDevice);

    int *hclusters = new int[M*N];
    for(int i=0; i<M*N; i++)
        hclusters[i] = -1;
    cudaMalloc(&clusters, M*N*sizeof(int));
    cudaMemcpy(clusters,hclusters,M*N*sizeof(int),cudaMemcpyHostToDevice);

    float *hdistances = new float[M*N];
    for(int i=0; i<M*N; i++)
        hdistances[i] = FLT_MAX;
    cudaMalloc(&distances, M*N*sizeof(float));
    cudaMemcpy(distances,hdistances,M*N*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&centers_colors, h_ncenters*sizeof(uchar3));
    cudaMalloc(&centers_coords, h_ncenters*sizeof(int2));

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

void interop_run(int M, int N, int hor, int vert) {

    dim3 grid1(1,1);
    dim3 block1(hor,vert);
    init_data<<<grid1,block1>>>(in_image, clusters, centers_colors, centers_coords);

	dim3 grid2(N,M);
	dim3 block2(1,1);
  int i,j,k;
  for(i=0; i<10; ++i){
    update_cluster<<<grid2,block2>>>( in_image, clusters, centers_coords, centers_colors, ncenters);
    // clear center values
    for(j=0;j<h_ncenters;++j){
      centers_colors[i].x = centers_colors[i].y = centers_colors[i].z = 0.0f;
      centers_coords[i].x = centers_coords[i].y = 0.0f;
    }

    // compute new cluster centers
    std::vector<int> center_counts(h_ncenters, 0);
    for(j=0;j<M;++j){
      for(k=0;k<N;++k){
        int c_id = clusters[j + k*N];
        if(c_id != -1){
          uchar3 colour = in_image[j + k*N];
          centers_colors[c_id].x += colour.x;
          centers_colors[c_id].y += colour.y;
          centers_colors[c_id].z += colour.z;

          centers_coords[c_id].x += j;
          centers_coords[c_id].y += k;

          center_counts[c_id] += 1;
        }
      }
    }

    //normalize the clusters
    for(j=0;j<h_ncenters;++j){
      centers_colors[c_id].x /= center_counts[j];
      centers_colors[c_id].y /= center_counts[j];
      centers_colors[c_id].z /= center_counts[j];

      centers_coords[c_id].x /= center_counts[j];
      centers_coords[c_id].y /= center_counts[j];
    }
  }


	cudaGraphicsUnmapResources( 1, &in_resource, NULL );
	cudaGraphicsUnmapResources( 1, &out_resource, NULL );

}

void interop_cleanup(){
	cudaGraphicsUnregisterResource( in_resource );
	cudaGraphicsUnregisterResource( out_resource );
}
