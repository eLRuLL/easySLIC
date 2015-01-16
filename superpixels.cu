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
cudaGraphicsResource *cluster_resource;
cudaGraphicsResource *distance_resource;
cudaGraphicsResource *centercolor_resource;
cudaGraphicsResource *centercoord_resource;
cudaGraphicsResource *centercount_resource;

uchar3 *in_image;
uchar3 *out_image;
int *clusters;
float *distances;
float3 *centers_colors;
int2 *centers_coords;
int *centers_count;

__constant__ int img_width;
__constant__ int img_height;
__constant__ int nr_superpixels;
__constant__ int nc;
__constant__ int step;
__constant__ int ncenters;

__device__ uchar3 Lab2Rgb(float3 lab_color)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;

    float X, Y, Z, fX, fY, fZ;
    int RR, GG, BB;

    int L = lab_color.x * (150.0/255.0);
    int a = lab_color.y - 128;
    int b = lab_color.z - 128;

    uchar3 Rgb;

    fY = pow((L + 16.0) / 116.0, 3.0);
    if (fY < 0.008856)
        fY = L / 903.3;
    Y = fY;

    if (fY > 0.008856)
        fY = powf(fY, 1.0/3.0);
    else
        fY = 7.787 * fY + 16.0/116.0;

    fX = a / 500.0 + fY;      
    if (fX > 0.206893)
        X = powf(fX, 3.0);
    else
        X = (fX - 16.0/116.0) / 7.787;

    fZ = fY - b /200.0;      
    if (fZ > 0.206893)
        Z = powf(fZ, 3.0);
    else
        Z = (fZ - 16.0/116.0) / 7.787;

    X *= (0.950456 * 255);
    Y *=             255;
    Z *= (1.088754 * 255);

    RR =  (int)(3.240479*X - 1.537150*Y - 0.498535*Z + 0.5);
    GG = (int)(-0.969256*X + 1.875992*Y + 0.041556*Z + 0.5);
    BB =  (int)(0.055648*X - 0.204043*Y + 1.057311*Z + 0.5);

    Rgb.x = (unsigned char)(RR < 0 ? 0 : RR > 255 ? 255 : RR);
    Rgb.y = (unsigned char)(GG < 0 ? 0 : GG > 255 ? 255 : GG);
    Rgb.z = (unsigned char)(BB < 0 ? 0 : BB > 255 ? 255 : BB);

    return Rgb;

}

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

__global__ void init_data1(int *clusters, float *distances){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;
    clusters[index] = -1;
    distances[index] = FLT_MAX;

}
__global__ void init_data2( uchar3* image, float3 *centers_colors, int2 *centers_coords, int *centers_count){
    
    // Find x,y coordinates (in image) of initial center

    int x = (threadIdx.x + 1)*step;
    int y = (threadIdx.y + 1)*step;

    // Find coordinates and color of Local Minimum

    int2 newc = find_local_minimum(image, make_int2(x,y));
    uchar3 colour = image[newc.x + newc.y*img_width];

    // Sets coords, colors, count values of center

    int offset = threadIdx.x + threadIdx.y*blockDim.x;

    centers_count[offset] = 0;
    centers_coords[offset] = newc;
    
    centers_colors[offset].x = colour.x;
    centers_colors[offset].y = colour.y;
    centers_colors[offset].z = colour.z;

    // Uncomment to see initial centers
    
    // int2 newcast = centers_coords[offset];
    // image[newcast.x + newcast.y*img_width].x = 0;
    // image[newcast.x + newcast.y*img_width].y = 0;
    // image[newcast.x + newcast.y*img_width].z = 255;

}

__global__ void clear_distances(float *distances){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;
    distances[index] = FLT_MAX;
}

__device__ float compute_dist(int2 center_coords, float3 center_lab, int2 pixel, uchar3 colour){
  float dc = sqrtf(powf(center_lab.x - colour.x, 2) + powf(center_lab.y - colour.y, 2) + powf(center_lab.z - colour.z, 2));
  float ds = sqrtf(powf(center_coords.x - pixel.x, 2) + powf(center_coords.y - pixel.y, 2));
  float m = nc; // ESTO DEBE SER GLOBAL O PARAM
  float K = nr_superpixels; // TAMBIEN GLOBAL O PARAM (NUMERO DE SUPERPIXELES)
  float N = img_width*img_height; // ACTUALIZAR CON EL NUMERO DE PIXELES (WIDTH * HEIGHT)
  float S_value = sqrt(N/K);

  return dc + (m/S_value)*ds;
}

__global__ void generate_superpixels(uchar3 *image, int *clusters, float *distances, int2 *centers_coords, float3 *centers_colors, int *centers_count){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int index = x + y * blockDim.x * gridDim.x;

  int i,j,k,l;
  // int min_cluster_id;

  // for each PIXEL
  distances[index] = FLT_MAX;

  // for each CENTER
  if(index < ncenters)
  {
    for(k = centers_coords[index].x - step/1; k < centers_coords[index].x + step/1; ++k){
      for(l = centers_coords[index].y - step/1; l < centers_coords[index].y + step/1; ++l){

        if (k >= 0 && k < img_width && l >= 0 && l < img_height) {
          float d = compute_dist(centers_coords[index], centers_colors[index], make_int2(k, l), image[k + l*img_width]);
          if(d < distances[k + l*img_width]){
            distances[k + l*img_width] = d;
            clusters[k + l*img_width] = index;
          }
          // clusters[k + l*img_width] = index;

        }

      }
    }

    // clear
    centers_colors[index].x = 0;
    centers_colors[index].y = 0;
    centers_colors[index].z = 0;

    centers_coords[index].x = 0;
    centers_coords[index].y = 0;

    centers_count[index] = 0;
  }

}

__global__ void update_clusters(uchar3 *image, int *clusters, float3 *centers_colors, int2 *centers_coords, int *centers_count){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int index = x + y * blockDim.x * gridDim.x;
  // for each PIXEL
  int c_id = clusters[index];

  if(c_id != -1) {
    uchar3 colour = image[index];
    // centers_colors[c_id].x += colour.x;
    // centers_colors[c_id].y += colour.y;
    // centers_colors[c_id].z += colour.z;

    atomicAdd(&centers_colors[c_id].x, colour.x);
    atomicAdd(&centers_colors[c_id].y, colour.y);
    atomicAdd(&centers_colors[c_id].z, colour.z);
    
    atomicAdd(&centers_coords[c_id].x, x);
    atomicAdd(&centers_coords[c_id].y, y);

    atomicAdd(&centers_count[c_id], 1);
  }
}

__global__ void update_clusters2(uchar3 *image, float3 *centers_colors, int2 *centers_coords, int *centers_count){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int index = x + y * blockDim.x * gridDim.x;

  centers_colors[index].x /= centers_count[index];
  centers_colors[index].y /= centers_count[index];
  centers_colors[index].z /= centers_count[index];
  
  centers_coords[index].x /= centers_count[index];
  centers_coords[index].y /= centers_count[index];
  

  // int2 newcast = centers_coords[index];
  //   image[newcast.x + newcast.y*img_width].x = 0;
  //   image[newcast.x + newcast.y*img_width].y = 0;
  //   image[newcast.x + newcast.y*img_width].z = 255;
}

__global__ void paint_clusters(uchar3 *image, int *clusters, float3* centers_colors){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int index = x + y * blockDim.x * gridDim.x;

  int c_id = clusters[index];
  uchar3 color = Lab2Rgb(centers_colors[c_id]);

  // image[index].x =  (unsigned char)centers_colors[c_id].x;
  // image[index].y =  (unsigned char)centers_colors[c_id].y;
  // image[index].z =  (unsigned char)centers_colors[c_id].z;

  // image[index].x =  100;
  // image[index].y =  0;
  // image[index].z =  c_id + 50;

  image[index].x =  color.z;
  image[index].y =  color.y;
  image[index].z =  color.x;
}  



void interop_setup(int M, int N, int h_nr_superpixels, int h_nc, int h_step, int h_ncenters) {
	cudaDeviceProp prop;
	int dev;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice( &dev, &prop );
	cudaGLSetGLDevice( dev );	// dev = 0
	cudaMemcpyToSymbol(img_height,&M,1*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(img_width,&N,1*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nr_superpixels,&h_nr_superpixels,1*sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nc,&h_nc,1*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(step,&h_step,1*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ncenters,&h_ncenters,1*sizeof(int),0,cudaMemcpyHostToDevice);

}

void interop_register_buffer(GLuint& in_buffer, GLuint& out_buffer, GLuint& cluster_buffer, GLuint& distance_buffer, GLuint& centercolor_buffer, GLuint& centercoord_buffer, GLuint& centercount_buffer){
	cudaGraphicsGLRegisterBuffer( &in_resource,in_buffer,cudaGraphicsMapFlagsNone);
  cudaGraphicsGLRegisterBuffer( &out_resource,out_buffer,cudaGraphicsMapFlagsNone);
  cudaGraphicsGLRegisterBuffer( &cluster_resource,cluster_buffer,cudaGraphicsMapFlagsNone);
  cudaGraphicsGLRegisterBuffer( &distance_resource,distance_buffer,cudaGraphicsMapFlagsNone);
  cudaGraphicsGLRegisterBuffer( &centercolor_resource,centercolor_buffer,cudaGraphicsMapFlagsNone);
  cudaGraphicsGLRegisterBuffer( &centercoord_resource,centercoord_buffer,cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer( &centercount_resource,centercount_buffer,cudaGraphicsMapFlagsNone);
}

void interop_map() {
	size_t in_size, out_size, distance_size, cluster_size, centercolor_size, centercoord_size, centercount_size;

	cudaGraphicsMapResources( 1, &in_resource, NULL );
	cudaGraphicsResourceGetMappedPointer( (void**)&in_image,&in_size,in_resource ) ;

	cudaGraphicsMapResources( 1, &out_resource, NULL );
	cudaGraphicsResourceGetMappedPointer( (void**)&out_image,&out_size,out_resource ) ;

  cudaGraphicsMapResources( 1, &cluster_resource, NULL );
  cudaGraphicsResourceGetMappedPointer( (void**)&clusters,&cluster_size,cluster_resource ) ;

  cudaGraphicsMapResources( 1, &distance_resource, NULL );
  cudaGraphicsResourceGetMappedPointer( (void**)&distances,&distance_size,distance_resource ) ;

  cudaGraphicsMapResources( 1, &centercolor_resource, NULL );
  cudaGraphicsResourceGetMappedPointer( (void**)&centers_colors,&centercolor_size,centercolor_resource ) ;

  cudaGraphicsMapResources( 1, &centercoord_resource, NULL );
  cudaGraphicsResourceGetMappedPointer( (void**)&centers_coords,&centercoord_size,centercoord_resource ) ;

  cudaGraphicsMapResources( 1, &centercount_resource, NULL );
  cudaGraphicsResourceGetMappedPointer( (void**)&centers_count,&centercount_size,centercount_resource ) ;
}

void interop_run(int M, int N, int hor, int vert, int h_ncenters, int nr_iterations, unsigned char* image) {

	  dim3 grid1(N,M);
	  dim3 block1(1,1);

    dim3 grid2(1,1);
    dim3 block2(hor,vert);

      init_data1<<<grid1,block1>>>(clusters,distances);
      init_data2<<<grid2,block2>>>(in_image, centers_colors, centers_coords,centers_count);    
    
    for(int i=0; i<nr_iterations; ++i){
      clear_distances<<<grid1,block1>>>(distances);
      generate_superpixels<<<grid1,block1>>>(in_image, clusters, distances, centers_coords, centers_colors, centers_count);
      update_clusters<<<grid1,block1>>>(in_image, clusters,centers_colors,centers_coords,centers_count);
      update_clusters2<<<grid2,block2>>>(in_image,centers_colors,centers_coords,centers_count);
    }
    paint_clusters<<<grid1,block1>>>(in_image, clusters, centers_colors);

 //  int i,j,k;

  // uchar3 *hcenters_colors = new uchar3[h_ncenters];
  // int2 *hcenters_coords = new int2[h_ncenters];
  // int *hclusters = new int[M*N];


  // // ITERACIONES
  // for(i=0; i<10; ++i){
  //   std::cout<<"iteracion "<<i<<std::endl;
  //   update_cluster<<<grid2,block2>>>( in_image, clusters, centers_coords, centers_colors, h_ncenters);
  //   // cudaMemcpy(hcenters_colors,centers_colors,M*N*sizeof(float),cudaMemcpyDeviceToHost);
  //   // cudaMemcpy(hcenters_coords,centers_coords,M*N*sizeof(float),cudaMemcpyDeviceToHost);
  //   cudaMemcpy(hclusters,clusters,M*N*sizeof(int),cudaMemcpyDeviceToHost);
  //   // cudaMemcpy(out_image,in_image,M*N*sizeof(uchar3),cudaMemcpyDeviceToHost);
  //   // clear center values
  //   std::cout<<"1 todo bien"<<std::endl;
  //   for(j=0;j<h_ncenters;++j){
  //     std::cout<<j<<std::endl;
  //     hcenters_colors[j].x = hcenters_colors[j].y = hcenters_colors[j].z = 0.0f;
  //     hcenters_coords[j].x = hcenters_coords[j].y = 0.0f;
  //   }
  //   std::cout<<"2 todo bien"<<std::endl;

  //   // compute new cluster centers
  //   std::vector<int> center_counts(h_ncenters, 0);
  //   for(j=0;j<M;++j){
  //     for(k=0;k<N;++k){
  //       std::cout<<j<<" "<<k<<std::endl;
  //       int c_id = hclusters[j + k*N];
  //       if(c_id != -1){
  //         std::cout<<"c_id: "<<c_id<<std::endl;
  //         // unsigned char colour = image[j + k*N];
  //         // std::cout<<"color: "<<colour.x<<colour.y<<colour.z<<std::endl;
  //         hcenters_colors[c_id].x += image[(j + k*N)*3];
  //         hcenters_colors[c_id].y += image[j + k*N];
  //         hcenters_colors[c_id].z += image[j + k*N];

  //         hcenters_coords[c_id].x += j;
  //         hcenters_coords[c_id].y += k;

  //         center_counts[c_id] += 1;
  //       }
  //     }
  //   }

  //   //normalize the clusters
  //   for(j=0;j<h_ncenters;++j){
  //     hcenters_colors[j].x /= center_counts[j];
  //     hcenters_colors[j].y /= center_counts[j];
  //     hcenters_colors[j].z /= center_counts[j];

  //     hcenters_coords[j].x /= center_counts[j];
  //     hcenters_coords[j].y /= center_counts[j];
  //   }
  // }


	cudaGraphicsUnmapResources( 1, &in_resource, NULL );
  cudaGraphicsUnmapResources( 1, &out_resource, NULL );
  cudaGraphicsUnmapResources( 1, &cluster_resource, NULL );
  cudaGraphicsUnmapResources( 1, &distance_resource, NULL );
  cudaGraphicsUnmapResources( 1, &centercolor_resource, NULL );
  cudaGraphicsUnmapResources( 1, &centercoord_resource, NULL );
	cudaGraphicsUnmapResources( 1, &centercount_resource, NULL );

}

void interop_cleanup(){
	cudaGraphicsUnregisterResource( in_resource );
  cudaGraphicsUnregisterResource( out_resource );
  cudaGraphicsUnregisterResource( cluster_resource );
  cudaGraphicsUnregisterResource( distance_resource );
  cudaGraphicsUnregisterResource( centercolor_resource );
  cudaGraphicsUnregisterResource( centercoord_resource );
	cudaGraphicsUnregisterResource( centercount_resource );
}
