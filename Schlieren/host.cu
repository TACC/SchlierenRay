/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2012-2013
  Scientific Computing and Imaging Institute, University of Utah

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/





// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <teem/nrrd.h>


// includes, GL
#include "opengl_include.h"
#include <float.h>
#include <assert.h>

// includes
#include <cutil.h>
#include <cutil_math.h>
#include <cuda.h>
#include "cuda_gl_interop.h"
#include <cstdlib>
#include "main.h"

#define THREAD_SAFE 0
#define USE_ATOMICS 0  //doesn't currently work, only works with ints
#define PRECOMPUTE_GRADIENT 1
#define INTERPOLATE_GRADIENT 1  // must also precompute for this
#define TRILINEAR_INTERPOLATION 1 // for data
#define USE_TF 0
#define WEIGHT_VALUES 1 //instead of killing off, weight photons
#define GUASSIAN_FILTER 0  //filter based on location in pixel.  note guassianfilter isn't thread safe
#define INTERFOREMETRY 0

//image filtering methods
  #define GUASSIAN_BLUR 0
  #define PHOTON_MAPPING 0
  #define AVG_FILTER 0

#define PROJECT_TO_FILM 0 // project rays to the film plane, otherwise they are taken as soon as they hit the end 
#define PROJECTION_DISTANCE 0.2f
enum cutoffs {CUTOFF_NONE = 0, CUTOFF_KNIFE_X = 1, CUTOFF_KNIFE_NEGATIVE_X, CUTOFF_KNIFE_Y, CUTOFF_KNIFE_NEGATIVE_Y, CUTOFF_PINHOLE, CUTOFF_INVERSE_PINHOLE};
int cutoff = CUTOFF_NONE;
// of the volume
#define KILL_SCALE 2000.0f
#define OCTREE 0
#define THRESHHOLD 0.0003f //thresh_hold for octree
#define BLUR_GRADIENT 0  //use guassian smoothing kernel over gradient
#define CAMERA_XY_AXIS 0 //full camera rotation or only about y axis
#define CAMERA_X_AXIS 1
#define CAMERA_Y_AXIS 0
#define LINE_OF_SIGHT 1 //trace through with no refraction


#define FILE_DATASET 1
#define SIN_DATASET 0
#define VASE_DATASET 0
#define TSI_DATASET 0 
#define COMBUSTION_DATASET 0
#define PV_DATASET 0
  #define PV_DATASET_MAG 0
  #define PV_DATASET_DENSITY 0
  #define PV_DATASET_PRESSURE 0
#define OCEAN_DATASET 0
#define DATA_SIZE 256.0f
int window_width = 512, window_height = 512;
#define NUM_PER_THREAD 1  //defines how many per thread, otherwise num_per_pixel will be spread out across more blocks
#define NUM_PER_PIXEL 2 //run 5 samples for every pixel per pass of the kernel
#define NUM_RAYS window_width*window_height*NUM_PER_THREAD*NUM_PER_PIXEL
int START_TIMESTEP = 1;
int END_TIMESTEP = 3; // end timestep will be 1 past, ie timestep of 70 will load 0-69
int NUM_TIMESTEPS = (END_TIMESTEP-START_TIMESTEP);
#define SAMPLES_PER_PIXEL 20  //if not clamping, run until this many samples per pixel
#define LOOP 1
#define MOUSE_SPEEDUP 1.0  // how many less rays to cast when moving mouse.  5.0 = 1 fifth of normal

#define CLAMP 1  //if clamped it is used like photography, use a constant normalization factor (over expose/under expose).
// otherwise it normalizes values based on the maximum samples per pixel value.
#define NUM_RUNS 1000.0f
#define NORMALIZATION_CONSTANT  NUM_RUNS*NUM_PER_PIXEL  //in NUM_RUNS I'll cap at...

#define COLOR 0
#define COLOR_DIRECTION 1  //color based on resulting angle offset
#define COLOR_ACCUMULATE 0  //accumulate color based on data value as ray marches
#define COLOR_ON_OTHER_DATA 0
#define COLOR_OPACITY 0 // color is darker if low value... will make most of scene black
#define GLOBAL_MINMAX 0 //use min max values accross all timesteps
typedef struct
{
  char r, g, b;
} color3;

#if COLOR
typedef float4 pixel_bufferT;
typedef color3 normalized_pixel_bufferT;
#else
typedef char normalized_pixel_bufferT;
#if GUASSIAN_FILTER || WEIGHT_VALUES
typedef float pixel_bufferT;
#else 
typedef unsigned int pixel_bufferT;
#endif // GUASSIAN_FILTER
#endif // COLOR

#include "kernel_volume.h"
#include "kernel_post.h"

#define SET_FLOAT3(s, a, b, c) { s.x =  a; s.y = b; s.z = c; }
#define FLOAT3_MINUS(v1, v2, r) { r.x = v1.x - v2.x; r.y = v1.y-v2.y; r.z = v1.z - v2.z; }


// rendering callbacks
void run(int rank, int size, int sizex, int sizey, int sizez, char** files, int num_files, int filemin, int filemax, DataFile** data_files, float step_size, bool record);

extern "C" void loadNRRD(DataFile* datafile, int data_min, int data_max);

void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
bool initGL();
void loadTimestep(int t);
void clear();
void setCamera();
void slowClear();
void writeOutput(int imagenum);

int num_rays = NUM_RAYS;
bool rotating = false;
normalized_pixel_bufferT* normalized_pixel_buffer;
pixel_bufferT* pixel_buffer;
unsigned char* octree, *d_octree;
float3* ray_pos;
float3* ray_dir;
float3* gradient[1];
float3 center = {0,0,0};
float* data[1], *data2[1];
//CUarray* data_array;
//CUtexref* data_ref;
float* rand_x, *rand_y, *rand_t;
float max_spp = 0.0f;
size_t data_depth = DATA_SIZE, data_height = DATA_SIZE, data_width = DATA_SIZE;
//size_t data_depth = 320, data_height = 100, data_width = 320; 
int timestep = 0;
int pass_count = 0;
float3 min_bound = {0,0,0}, max_bound = {1,1,1};
float data_min = 99999;
float data_max = -99999;
//float* file_data = NULL;
DataFile** file_data;
int file_min;  // min and max of slice along z
int file_max;
char** files;
bool clearing = false;
int file_width, file_height, file_depth;
TransferFunction* transferFunction = NULL;
float4* d_TFBuckets = NULL;
TFRenderable* tfRenderable = NULL;
bool dirty_gradient = true; //used for TF changes
float step_size;
float frame_time = 0;
bool orbit = false;
bool recording = false; // because frame capture on linux sucks monkey stuff


int avg_timer_counter_frames = 0;
float avg_timer_total_frames = 0.0f;
float avg_timer_total_frame = 0.0f;
int avg_timer_counter_frame = 0;
//std::vector<float> avg_times_frames;


cudaArray* data_array = NULL;
cudaArray* gradient_array_x = NULL;
cudaArray* gradient_array_y = NULL;
cudaArray* gradient_array_z = NULL;


//CUtexref* texture;

//device memory
  float3* d_ray_pos, *d_ray_dir, *d_gradient;
  pixel_bufferT *d_pixel_buffer;
  float *d_max_spp;
  normalized_pixel_bufferT* d_normalized_pixel_buffer;
  float* d_data, *d_data2, *d_rand_x, *d_rand_y, *d_rand_t;

time_t rngTimer;

float3 set_float3(float x, float y, float z)
{
  float3 r = {x,y,z};
  return r;
}

float magnitude(float3 f)
{
  return sqrt(f.x*f.x + f.y*f.y+f.z*f.z);
}

float dot3h(float3 v1, float3 v2)
{
  return v1.x * v2.x+ v1.y*v2.y+ v1.z*v2.z;
}

float3 norm3h(float3 v1)
{
  float mag = sqrt(dot3h(v1,v1));
  if (mag == 0)
    return v1;
  mag = 1.0/mag;
  float3 result = {v1.x*mag, v1.y*mag, v1.z*mag};
  return result;
}

float3 minus(float3 v1, float3 v2)
{
  float3 result = {v1.x-v2.x, v1.y-v2.y, v1.z-v2.z};
  return result;
}

float3 add(float3 v1, float3 v2)
{
  float3 result = {v1.x+v2.x, v1.y+v2.y, v1.z+v2.z};
  return result;
}

float3 divide(float3 v1, float f)
{
  float3 result = {v1.x/f, v1.y/f, v1.z/f};
  return result;
}

float3 cross3h(float3 v1, float3 v2)
{
  float3 result;
  result.x = (v1.y*v2.z)-(v1.z*v2.y);
  result.y = (v1.z*v2.x) - (v1.x*v2.z);
  result.z = (v1.x*v2.y) - (v1.y*v2.x);
  return result;
}

float3 mul3h(float3 v1, float v2)
{
  float3 result = {v1.x * v2, v1.y*v2, v1.z*v2};
  return result;
}


//#include "MersenneTwister.h"

float rot_x = 0 , rot_y = 0;
float3 camera_pos = {0,0,-5}, camera_x = {1,0,0}, camera_y = {0,1,0}, camera_z = {0,0,1};
int mpi_rank, mpi_size;

float getRandom()
{
  return drand48();
}

void blurGradient()
{
     printf("blurring gradient... ");
    float deviation = 0.2;
    int kernel_size = 1;
    float3 total;
    int sizex = data_width, sizey=data_height, sizez = data_depth;
    //float denom = pow(2.0*M_PI*deviation*deviation, 3.0/2.0);
    float denom = (2.0*M_PI)*(2.0*M_PI)*deviation*deviation;
    float denom2 = 2.0*deviation*deviation;
    float3* filtered_data = new float3[sizex*sizey*sizez];
//2D Guassian Kernel  G(x,y) = (1.0f/((2*pi)*deviation*deviation))*expf(-(x*x+y*y)/(2.0*deviation*deviation));
     for(int p_x = kernel_size; p_x < sizex - kernel_size; p_x++) {
      for(int p_y = kernel_size; p_y < sizey - kernel_size; p_y++) {
        for(int p_z = kernel_size; p_z < sizez - kernel_size; p_z++) {
          total = make_float3(0,0,0);
          int count = 0;
          bool different = false;
          float3 initial_val = gradient[0][(p_x) + (p_y)*sizex + (p_z)*sizex*sizey];
    for(int k_x = -kernel_size; k_x <= kernel_size; k_x++) {
        for(int k_y = -kernel_size; k_y <= kernel_size; k_y++) {
          for(int k_z = -kernel_size; k_z <= kernel_size; k_z++) {
          int b_x = p_x + k_x;
          int b_y = p_y + k_y;
          int b_z = p_z + k_z;
          if (b_x < 0 || b_x >= sizex)
            continue;
          if (b_y < 0 || b_y >= sizey)
            continue;
          if (b_z < 0 || b_z >= sizez)
            continue;
          float3 val = gradient[0][(b_x) + (b_y)*sizex + (b_z)*sizex*sizey];
          if (length(val - initial_val) > 0.0001f)
              different = true;
          float r = k_x*k_x+k_y*k_y+k_z*k_z;
          total += val*(1.0f/denom)*expf(-(r)/(denom2));
          //total += val;
          count++;
          }
        }
    }
    if (!different)
      filtered_data[(p_x) + (p_y)*sizex + (p_z)*sizex*sizey] = initial_val;
    else
      filtered_data[(p_x) + (p_y)*sizex + (p_z)*sizex*sizey] = total;
        }
      }
    }

    for(int p_x = kernel_size; p_x < sizex - kernel_size; p_x++) {
      for(int p_y = kernel_size; p_y < sizey - kernel_size; p_y++) {
        for(int p_z = kernel_size; p_z < sizez - kernel_size; p_z++) {
          size_t index = (p_x) + (p_y)*sizex + (p_z)*sizex*sizey;
          gradient[0][index] = filtered_data[index];
        }
      }
    } 
    delete[] filtered_data;
    filtered_data = NULL;
    printf (" done\n");
}

void display()
{  
  if (orbit) {
   rot_x += 0.03;
   setCamera();
   clear();
  }
  if (step_size <= 0.001f)
  step_size = 0.001f;
  cudaEvent_t start, end, total_start, total_end;
  float seconds;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventCreate(&total_start);
  cudaEventCreate(&total_end);
  cudaEventRecord(total_start, 0);
  cudaEventRecord(start, 0);
  cudaEventRecord(end, 0);
  //cudaThreadSynchronize();
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&seconds, start, end);

  cudaEventRecord(start, 0);  


  float color_scale = 1.0f/(data_max-data_min);
  dim3 block(16,16,1);
  dim3 grid((window_width/block.x)*NUM_PER_PIXEL/NUM_PER_THREAD,(window_height/block.y)*NUM_PER_PIXEL/NUM_PER_THREAD,1);   //positions go 0 to 100, which maps to -1 to 1 on each lightplace axis
  float3 lookup_scale = {1.0f/(max_bound.x-min_bound.x), 1.0f/(max_bound.y - min_bound.y), 1.0f/(max_bound.z-min_bound.z)};
  float3 camera_corner = camera_pos-(camera_x*.5+camera_y*.5);
  float rand_x2 = getRandom();
  float rand_y2 = getRandom();
  float rand_t2 = getRandom();
  if (pass_count == 0) {
        rand_x2 = rand_y2 = 0; rand_t2 = 1.0;
  }
  kernel_volume<<< grid, block>>>(num_rays, data_width, data_height, data_depth, lookup_scale, d_data, 
    window_width, window_height, d_pixel_buffer, d_max_spp, min_bound, max_bound, 
    camera_pos, camera_x, camera_y, camera_z, camera_corner, d_rand_x, d_rand_y, d_rand_t
#if PRECOMPUTE_GRADIENT
    ,d_gradient
#endif
#if COLOR_ON_OTHER_DATA
  ,d_data2
#endif
  , cutoff, data_min, data_max, color_scale, rand_x2, rand_y2, rand_t2, clearing, 
#if OCTREE
d_octree, 
#endif 
#if USE_TF
d_TFBuckets, transferFunction->numBuckets, transferFunction->min, transferFunction->scale, 
#endif 
dirty_gradient, step_size
  );
    
//    cudaThreadSynchronize();
  cudaEventRecord(end, 0);
   cudaEventSynchronize(end);

   cudaThreadSynchronize();
  if (clearing)
        clearing = false;
  cudaEventElapsedTime(&seconds, start, end);

  //copy result from device to host
  float temp;
  cudaMemcpy(&temp, d_max_spp, sizeof(float), cudaMemcpyDeviceToHost);
  dim3 block2(64,1,1);
  int num_pixels_per_thread = NUM_PER_THREAD*10;
  dim3 grid2(window_width*window_height/num_pixels_per_thread/block2.x,1,1);
#if CLAMP
  float clamp = (float(pass_count)/(float(NUM_RUNS)))*float(NORMALIZATION_CONSTANT/NUM_PER_THREAD)*(float(NUM_RAYS)/float(window_width*window_height));
  if (clamp < 0.5)
        clamp = 0.5;
  cudaMemcpy(d_max_spp, &clamp, sizeof(float), cudaMemcpyHostToDevice);
#endif

//  cudaMemcpy(d_normalized_pixel_buffer, &(normalized_pixel_buffer[0]), sizeof(normalized_pixel_bufferT)*window_width*window_height, cudaMemcpyHostToDevice);
  kernel_normalize<<<grid2,block2>>>(d_pixel_buffer, d_normalized_pixel_buffer, d_max_spp, window_width*window_height, window_width, window_height);
   cudaThreadSynchronize();

 cudaMemcpy(normalized_pixel_buffer, &(d_normalized_pixel_buffer[0]), sizeof(normalized_pixel_bufferT)*window_width*window_height, cudaMemcpyDeviceToHost);

#if COLOR
  glDrawPixels(window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, normalized_pixel_buffer);
#else
  glDrawPixels(window_width, window_height, GL_LUMINANCE, GL_UNSIGNED_BYTE, normalized_pixel_buffer);
#endif
#if USE_TF
  tfRenderable->Draw();
#endif
  glFinish();
  glutSwapBuffers();
  glutPostRedisplay();
  
  cudaEventRecord(total_end, 0);
 // cudaThreadSynchronize();
 cudaEventSynchronize(total_end);
  cudaEventElapsedTime(&seconds, total_start, total_end);
  printf("total time: %f \n", seconds/1000.0);
  avg_timer_total_frame += seconds/1000.0;
  avg_timer_counter_frame++;


  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaEventDestroy(total_start);
  cudaEventDestroy(total_end);
  pass_count++;
  if (recording)
  {
    static int count = 0;
    writeOutput(count++);
  }
#if CLAMP
  if (pass_count >= NUM_RUNS) {
    pass_count = 0;
    avg_timer_total_frames += avg_timer_total_frame;
    //avg_times_frames.push_back(avg_timer_total_frame);
    avg_timer_counter_frames++;
    float avg = avg_timer_total_frames/float(avg_timer_counter_frames);
    printf("total time for this frame: %f\n",avg_timer_total_frame/float(avg_timer_counter_frame));
    float stdd = 0.0f;
    //for(int i =0; i < avg_times_frames.size(); i++)
    //  stdd += pow(float(avg_times_frames[i] - avg), 2.0f);
   // stdd = sqrt(1.0/float(avg_times_frames.size())*stdd);
    printf("average time for %d frames: %f, std: %f\n", avg_timer_counter_frames,avg, stdd);
    avg_timer_total_frame = 0;
    avg_timer_counter_frame = 0;
#else
  if (temp >= SAMPLES_PER_PIXEL) { 
#endif //CLAMP
    writeOutput(timestep);
    if (timestep +1 == NUM_TIMESTEPS && !(LOOP)) {
      exit(0);
    }
    int old_timestep = timestep;
    timestep = (timestep+1)%NUM_TIMESTEPS;
    if (old_timestep != timestep)
      loadTimestep(timestep);
    slowClear();
  }
}

void writeOutput(int imagenum)
{
    char filename[255];
    sprintf(filename, "images/render%05d.raw", imagenum);
 FILE* fp = fopen(filename, "wb");
    if(!fp) {
      printf("couldn't write file\n");
      exit(1);
    }
    fwrite(normalized_pixel_buffer, sizeof(normalized_pixel_bufferT),window_width*window_height, fp);
    fclose(fp);
    char call[512];
#if COLOR
     sprintf(call, "convert -size %dx%d -depth 8 rgb:images/render%05d.raw images/render%05d.png", window_width, window_height, imagenum, imagenum);
#else
    sprintf(call, "convert -size %dx%d -depth 8 -colorspace Gray gray:images/render%05d.raw images/render%05d.png", window_width, window_height, imagenum, imagenum);
#endif
    system(call);
}

void clearTimestep()
{
  
}

void loadTimestep(int t)
{
  printf("loading timestep: %f ", float(t));
  int file = t;
  t = 0; //TODO: give option whether to load all timesteps in at once or not
  
#if !GLOBAL_MINMAX
  data_min = 999999;
  data_max = -999999;
#endif

#if FILE_DATASET
  if(data[t])
    delete[] data[t];
  data[t] = NULL;
  loadNRRD(file_data[file], file_min, file_max);

  data[t] = file_data[file]->data;
  data_width = file_data[file]->sizex;
  data_height = file_data[file]->sizey;
  data_depth = file_data[file]->sizez;
  float max_scale = max(max(float(data_width), float(data_height)), float(data_depth));
  if (step_size <= 0.0f)
    step_size = 1.0f/max_scale;
  

#endif

  
#if COMBUSTION_DATASET
    char filename[512];
    sprintf(filename, "/usr/csafe/raid5/brownlee/combustion/jet_%04d/jet_mixfrac_%04d.dat", file+START_TIMESTEP,file+START_TIMESTEP);
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
      printf("error opening file %s\n", filename);
      exit(__LINE__);
    }
    if (data[t])
      delete[] data[t];
    data[t] = (float*)malloc(data_width*data_height*data_depth*sizeof(float));
    fread(data[t], sizeof(float), data_width*data_height*data_depth, fp);
    fclose(fp);
    
    for(size_t j = 0; j < data_depth*data_width*data_height; j++) {
      data[t][j] *= 0.01f; 
    }
#if COLOR_ON_OTHER_DATA
    char filename2[512];
    sprintf(filename2, "/data/scout/combustion/jet_%04d/jet_Y_OH_%04d.dat", file+START_TIMESTEP,file+START_TIMESTEP);
    FILE* fp2 = fopen(filename2, "rb");
    if (!fp2) {
      printf("error opening file %s\n", filename2);
      exit(__LINE__);
    }
    if (data2[t])
      delete[] data2[t];
    data2[t] = (float*)malloc(data_width*data_height*data_depth*sizeof(float));
    fread(data2[t], sizeof(float), data_width*data_height*data_depth, fp2);
    fclose(fp2);
    
//    for(size_t j = 0; j < data_depth*data_width*data_height; j++) {
//      data2[t][j] *= 10.0f; 
//    }
#endif //COLOR_ON_OTHER
#endif  //COMBUSTION

#if PV_DATASET
    char filename[512];
    sprintf(filename, "/data/scout/pv256/rstrt.%03d", file*5+START_TIMESTEP*5 + 250);
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
      printf("error opening file %s\n", filename);
      exit(__LINE__);
    }
    if (data[t])
      delete[] data[t];
    data[t] = (float*)malloc(data_width*data_height*data_depth*sizeof(float));
  #if PV_DATASET_MAG
    double* temp = (double*)malloc((8 * 256 * 256 * 256)*3);
    fseek(fp, (12+8), SEEK_SET);
    fread((void*)temp, sizeof(double), (256 * 256 * 256)*3, fp);
    fclose(fp);
    for(size_t j = 0; j < data_depth*data_width*data_height; j++) {
      data[t][j] = sqrt(temp[j]*temp[j]+temp[j+(256 * 256 * 256)]*temp[j+(256 * 256 * 256)]+
        temp[j+(256 * 256 * 256)*2]*temp[j+(256 * 256 * 256)*2]);
      data[t][j] *= 0.03f; 
    }
        delete[] temp;
  #elif PV_DATASET_DENSITY
    double* temp = (double*)malloc((8 * 256 * 256 * 256));
    fseek(fp, (12+8+(8 * 256 * 256 * 256)*3), SEEK_SET);
    fread((void*)temp, sizeof(double), (256 * 256 * 256), fp);
    fclose(fp);
    for(size_t j = 0; j < data_depth*data_width*data_height; j++) {
      data[t][j] = temp[j];
      data[t][j] *= 0.1f; 
    }
//    char filename2[512];
//    sprintf(filename2, "pv256_float_density/rstrt.%03d", file*5+START_TIMESTEP*5 + 250);
//    FILE* fp2 = fopen(filename2, "wb");
//    fwrite((void*)data[t], sizeof(float), data_depth*data_width*data_height, fp2);
        delete[] temp;
  #elif PV_DATASET_PRESSURE
    double* temp = (double*)malloc((8 * 256 * 256 * 256));
    fseek(fp, (12+8+(8 * 256 * 256 * 256)*4), SEEK_SET);
    fread((void*)temp, sizeof(double), (256 * 256 * 256), fp);
    fclose(fp);
    for(size_t j = 0; j < data_depth*data_width*data_height; j++) {
      data[t][j] = temp[j];
      data[t][j] *= 0.01f; 
    }
    delete[] temp;
  #endif
#endif

#if OCEAN_DATASET
    char filename[512];
    sprintf(filename, "/data/scout/ocean/north_atlantic/little-endian/SALT.0001.raw");
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
      printf("error opening file %s\n", filename);
      exit(__LINE__);
    }
    if (data[t])
      delete[] data[t];
    data[t] = (float*)malloc(data_width*data_height*data_depth*sizeof(float)*3);
    fread(data[t], sizeof(float), data_width*data_height*data_depth*3, fp);
    fclose(fp);
    for(size_t j = 0; j < data_width; j++) {
      for(int k = 0; k < data_height; k++) {
        for(int l = 0; l < data_depth; l++) {
          data[t][j+k*data_width+l*data_width*data_height] = data[t][j+k*data_width+(l+10)*data_width*data_height];
        }
      }
    }
    for(size_t j = 0; j < data_depth*data_width*data_height; j++) {
      data[t][j] *= 0.1f; 
    }
#endif

  
#if PRECOMPUTE_GRADIENT
  //
  //create gradient
  //
   printf("xyz: %f %f %f\n", float(data_width), float(data_height), float(data_depth));
    if (gradient[t])
      delete[] gradient[t];
    gradient[t] = (float3*)malloc(data_width*data_height*data_depth*sizeof(float3));
    for(size_t z = 0; z < data_depth; z++) {
    //          std::cout  << "\033[1A ... [" << int((float(z)/float(this->shapeOf().dimOf(2)))*100) << "]%\n";
      for(size_t y = 0; y < data_height; y++) {
        for(size_t x = 0; x < data_width; x++) {
          size_t DELTA = 1;
          float3 lookup = {x,y,z};
          if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA ||
              lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA) 
            continue;
          float3 sample1, sample2;
          SET_FLOAT3(lookup,x-1,y,z);
          sample1.x = data[t][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
          SET_FLOAT3(lookup,x+1,y,z);
          sample2.x = data[t][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
          
          SET_FLOAT3(lookup,x,y-1,z);
          sample1.y = data[t][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
          SET_FLOAT3(lookup,x,y+1,z);
          sample2.y = data[t][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
          
          SET_FLOAT3(lookup,x,y,z-1);
          sample1.z = data[t][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
          SET_FLOAT3(lookup,x,y,z+1);
          sample2.z = data[t][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
          float3 normal;
          FLOAT3_MINUS(sample1,sample2, normal);
         // normal = norm3h(normal);
         // if (normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0)
          //  SET_FLOAT3(normal,0,0,-1);
          gradient[t][size_t(z*data_width*data_height + y*data_width + x)] = normal;
        }
      }
    }
#if BLUR_GRADIENT
    blurGradient();
    dirty_gradient = false;
#endif
#endif  // PRECOMPUTE_GRADIENT

#if COLOR_ON_OTHER_DATA
  for(size_t i = 0; i < data_width*data_height*data_depth; i++) {
    if (data2[0][i] < data_min)
      data_min = data2[0][i];
    if (data2[0][i] > data_max)
      data_max = data2[0][i];
  }
#else
  for(size_t i = 0; i < data_width*data_height*data_depth; i++) {
    if (data[0][i] < data_min)
      data_min = data[0][i];
    if (data[0][i] > data_max)
      data_max = data[0][i];
  }
#endif

  if (data_array)
    cudaFree(data_array);

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(data_width, data_height, data_depth);
  CUDA_SAFE_CALL( cudaMalloc3DArray(&data_array, &channelDesc, extent) );
//  cudaMemcpyToArray(data_array,0,0,data[0], data_width*data_height*data_depth*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr = make_cudaPitchedPtr((void*)data[0], extent.width*sizeof(float), extent.width, extent.height);
  copyParams.dstArray = data_array;
  copyParams.extent = extent;
  copyParams.kind = cudaMemcpyHostToDevice;
  CUDA_SAFE_CALL(  cudaMemcpy3D(&copyParams) );

  tex_data.addressMode[0] = cudaAddressModeWrap;
  tex_data.addressMode[1] = cudaAddressModeWrap;
  tex_data.filterMode = cudaFilterModeLinear;
  tex_data.normalized = false;

  cudaBindTextureToArray(tex_data, data_array, channelDesc);

  static float* gradient_x = NULL;
  static float* gradient_y = NULL;
  static float* gradient_z = NULL;

  if (gradient_x == NULL)
     gradient_x = (float*)malloc(sizeof(float)*data_width*data_height*data_depth);
  if (gradient_y == NULL)
     gradient_y = (float*)malloc(sizeof(float)*data_width*data_height*data_depth);
  if (gradient_z == NULL)
     gradient_z = (float*)malloc(sizeof(float)*data_width*data_height*data_depth);

  for(size_t i = 0; i < data_width*data_height*data_depth; i++) {
    gradient_x[i] = gradient[0][i].x;
    gradient_y[i] = gradient[0][i].y;
    gradient_z[i] = gradient[0][i].z;
  }

  if (gradient_array_x)
    cudaFree(gradient_array_x);
  if (gradient_array_y)
    cudaFree(gradient_array_y);
  if (gradient_array_z)
    cudaFree(gradient_array_z);
  channelDesc = cudaCreateChannelDesc<float>();
  CUDA_SAFE_CALL( cudaMalloc3DArray(&gradient_array_x, &channelDesc, extent) );
  CUDA_SAFE_CALL( cudaMalloc3DArray(&gradient_array_y, &channelDesc, extent) );
  CUDA_SAFE_CALL( cudaMalloc3DArray(&gradient_array_z, &channelDesc, extent) );
  cudaMemcpy3DParms copyParams2 = {0};

  copyParams2.srcPtr = make_cudaPitchedPtr((void*)gradient_x, extent.width*sizeof(float), extent.width, extent.height);
  copyParams2.dstArray = gradient_array_x;
  copyParams2.extent = extent;
  copyParams2.kind = cudaMemcpyHostToDevice;
  CUDA_SAFE_CALL(  cudaMemcpy3D(&copyParams2) );
  tex_gradient_x.addressMode[0] = cudaAddressModeWrap;
  tex_gradient_x.addressMode[1] = cudaAddressModeWrap;
  tex_gradient_x.filterMode = cudaFilterModeLinear;
  tex_gradient_x.normalized = false;
  cudaBindTextureToArray(tex_gradient_x, gradient_array_x, channelDesc);

  copyParams2.srcPtr = make_cudaPitchedPtr((void*)gradient_y, extent.width*sizeof(float), extent.width, extent.height);
  copyParams2.dstArray = gradient_array_y;
  copyParams2.extent = extent;
  copyParams2.kind = cudaMemcpyHostToDevice;
  CUDA_SAFE_CALL(  cudaMemcpy3D(&copyParams2) );
  tex_gradient_y.addressMode[0] = cudaAddressModeWrap;
  tex_gradient_y.addressMode[1] = cudaAddressModeWrap;
  tex_gradient_y.filterMode = cudaFilterModeLinear;
  tex_gradient_y.normalized = false;
  cudaBindTextureToArray(tex_gradient_y, gradient_array_y, channelDesc);

  copyParams2.srcPtr = make_cudaPitchedPtr((void*)gradient_z, extent.width*sizeof(float), extent.width, extent.height);
  copyParams2.dstArray = gradient_array_z;
  copyParams2.extent = extent;
  copyParams2.kind = cudaMemcpyHostToDevice;
  CUDA_SAFE_CALL(  cudaMemcpy3D(&copyParams2) );
  tex_gradient_z.addressMode[0] = cudaAddressModeWrap;
  tex_gradient_z.addressMode[1] = cudaAddressModeWrap;
  tex_gradient_z.filterMode = cudaFilterModeLinear;
  tex_gradient_z.normalized = false;
  cudaBindTextureToArray(tex_gradient_z, gradient_array_z, channelDesc);


  if (d_rand_x)
     cudaFree(d_rand_x);
  cudaMalloc( (void**)&d_rand_x, num_rays*sizeof(float));
   int window_size = window_width*window_height;
   rand_x  = (float*)malloc(sizeof(float)*window_size);
   for(int i =0;i<window_size;i++){
    rand_x[i] = getRandom();
   }
   cudaMalloc((void**)&d_rand_x, window_size*sizeof(float));
cudaMemcpy( d_rand_x, rand_x,sizeof(float)*window_size, cudaMemcpyHostToDevice  );

  

#if PRECOMPUTE_GRADIENT
 // if (d_gradient)
 //       cudaFree(d_gradient);
 // cudaMalloc( (void**)&d_gradient, data_width*data_height*data_depth*sizeof(float3));
#endif

//TODO: clear out old malloced data...  take out once, won't work if data changes size
static bool once = false;
if (!once) {
once = true;
  
#if COLOR_ON_OTHER_DATA
  if(d_data2)
    cudaFree(d_data2);
  cudaMalloc( (void**)&d_data2, data_width*data_height*data_depth*sizeof(float));
#endif
 // if (d_data)
 //  cudaFree(d_data);
 // cudaMalloc( (void**)&d_data, data_width*data_height*data_depth*sizeof(float));
 // cudaMalloc( (void**)&d_pixel_buffer_i, window_width*window_height*sizeof(float));
  if (d_pixel_buffer)
    void;//cudaFree(d_pixel_buffer);
  cudaMalloc( (void**)&d_pixel_buffer, window_width*window_height*sizeof(pixel_bufferT));
  if (d_normalized_pixel_buffer)
    void;//cudaFree(d_normalized_pixel_buffer);
  cudaMalloc( (void**)&d_normalized_pixel_buffer, window_width*window_height*sizeof(normalized_pixel_bufferT));
  if (d_max_spp)
    void;//cudaFree(d_max_spp);
  cudaMalloc( (void**)&d_max_spp, sizeof(float));
  
  if (d_pixel_buffer)
    void;//cudaFree(d_pixel_buffer);
  cudaMemcpy( d_pixel_buffer, pixel_buffer,window_width*window_height*sizeof(pixel_bufferT), cudaMemcpyHostToDevice  );
  if (d_max_spp)
    void;//cudaFree(d_max_spp);
  cudaMemcpy( d_max_spp, &max_spp,sizeof(float), cudaMemcpyHostToDevice  );
}

#if PRECOMPUTE_GRADIENT
//  cudaMemcpy( d_gradient, gradient[t],data_width*data_height*data_depth*sizeof(float3), cudaMemcpyHostToDevice  );
#endif
#if COLOR_ON_OTHER_DATA
  cudaMemcpy( d_data2, data2[t],data_width*data_height*data_depth*sizeof(float), cudaMemcpyHostToDevice  ) ;
#endif
 // cudaMemcpy( d_data, data[t],data_width*data_height*data_depth*sizeof(float), cudaMemcpyHostToDevice  ) ;


//#if OCTREE
//compute octree
  printf("constructing octree\n");
  if (octree)
     delete[] octree;
  octree = (unsigned char*)malloc(sizeof(unsigned char)*data_width*data_height*data_depth);
  if (d_octree)
    void;//cudaFree(d_octree);
  cudaMalloc( (void**)&d_octree, sizeof(unsigned char)*data_width*data_height*data_depth);
  float threshhold = THRESHHOLD;
  int max_level = 0;
  size_t min_size = data_width;
  size_t max_size = data_width;
  min_size = min(int(min_size), int(data_height));
  min_size = min(int(min_size), int(data_depth));
  max_size = max(int(max_size), int(data_height));
  max_size = max(int(max_size), int(data_depth));
  int num_levels = log(min_size)/log(2);
  for(size_t i = 0; i < data_width*data_height*data_depth; i++) {
    octree[i] = 0;
  }
  for(int level = num_levels; level > 0; level--) {
    int width = pow(2.0, level);
    for(int baseX = 0; baseX < data_width; baseX += width) {
        for(int baseY = 0; baseY < data_height; baseY += width) {
          for(int baseZ = 0; baseZ < data_depth; baseZ += width) {
                float min_val = 999.0f;
                float max_val = -999.0f;
                bool skip = false;
                size_t base_index = baseX + baseY*data_width + baseZ*data_width*data_height;
                if (octree[base_index] > level)
                        continue;
                for(int x = 0 ; x < width && !skip; x++) {
                  for(int y = 0; y < width && !skip; y++) {
                    for(int z = 0; z < width && !skip; z++) {
                        int a_x = baseX + x;
                        int a_y = baseY + y;
                        int a_z = baseZ + z;
                        if (a_x >= data_width || a_y >= data_height || a_z >= data_depth) {
  skip = true;      continue;
}
                        size_t index = a_x + a_y*data_width + a_z*data_width*data_height;
                        float val = data[t][index];
                        min_val = min(min_val, val);
                        max_val = max(max_val, val);
                    }
}
}
                if (skip)
                        continue;
                if (fabs(max_val-min_val) < threshhold) {
                   if (level > max_level)
                        max_level = level;
for(int x = 0 ; x < width; x++) {
                  for(int y = 0; y < width; y++) {
                    for(int z = 0; z < width; z++) {
                        int a_x = baseX + x;
                        int a_y = baseY + y;
                        int a_z = baseZ + z;
                        if (a_x >= data_width || a_y >= data_height || a_z >= data_depth) {
  continue;       
}
                        size_t index = a_x + a_y*data_width + a_z*data_width*data_height;
                        octree[index] = level;
                    }
}
}
}
            
          }
    }
    
  }
}
  cudaMemcpy( d_octree, octree, sizeof(unsigned char)*data_width*data_height*data_depth,cudaMemcpyHostToDevice);
 printf(" octree construction completed, max level %d\n", max_level);
//#endif OCTREE
  
//  cudaBindTexture(0, texRIRef, data[t], data_width*data_height*data_depth*sizeof(float));
 // float* tempData = (float*)malloc(sizeof(float4)*data_width*data_height*data_depth);
//  cudaBindTexture(0, texRef, tempData, sizeof(float4)*data_width*data_height*data_depth);

  //texture gradient...
 // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0,0,0, cudaChannelFormatKindFloat);
  //cudaArray* cu_array;
  //cudaExtent extent = {data_width, data_height, data_depth};
//  cudaMalloc3DArray(&cu_array, &channelDesc, extent);
  //cudaMemcpyToArray(cu_array,0,0,tempData, data_width*data_height*data_depth*sizeof(float), cudaMemcpyHostToDevice);

  //cudaBindTextureToArray(texRIRef, cu_array, channelDesc);
#if USE_TF
  if (transferFunction)
    delete transferFunction;
  transferFunction = new TransferFunction(data_min,data_max,512);
  if (tfRenderable)
    delete tfRenderable;
  tfRenderable = new TFRenderable(transferFunction, 0,0,512,100);
  if (d_TFBuckets)
    void;//cudaFree(d_TFBuckets);
  cudaMalloc((void**)&d_TFBuckets, sizeof(float4)*transferFunction->numBuckets);
  cudaMemcpy(d_TFBuckets, transferFunction->buckets, sizeof(float4)*transferFunction->numBuckets, cudaMemcpyHostToDevice);
#endif USE_TF  

  printf(" ...done\n");
}


bool initGL()
{
  printf("initializing scene\n");
  
  rand_x = (float*)malloc(num_rays*sizeof(float));
  rand_y = (float*)malloc(num_rays*sizeof(float));
  rand_t = (float*)malloc(num_rays*sizeof(float));
  pixel_buffer = (pixel_bufferT*)malloc(window_width*window_height*sizeof(pixel_bufferT));
  normalized_pixel_buffer = (normalized_pixel_bufferT*)malloc(window_width*window_height*sizeof(normalized_pixel_bufferT));
    gradient[0] = NULL;
    data[0] = NULL;
    data2[0] = NULL;
  
  for(int i = 0; i < window_width*window_height; i++) {
#if COLOR
    float4 color = {0,0,0,0};
    pixel_buffer[i] = color;
#else  
    pixel_buffer[i] = 0;
#endif
  }
  
//  //
//  //create 2d sinusodal wave
//  //


#if SIN_DATASET
    data_depth = 32;
    data[0] = (float*)malloc(data_width*data_height*data_depth*sizeof(float));
  for(size_t i = 0; i < data_width*data_height*data_depth; i++) 
    data[0][i] = 0.0f;
  for(size_t z = 0; z < data_depth; z++) {
    for(size_t i = 0; i < data_width; i++) {
      for(size_t j = 0; j < data_height; j++) {
        float phase = 0.25;
        float dst1 = (i - data_width/2.0)/2.0;
        float dst2 = (j - data_height/2.0)/2.0;
        float val = sin(sqrt(dst1*dst1 + dst2*dst2)*phase)*.491 + .4999;
        
        if (val < 0.0)
          val = 0.0;
        if (val > 1.0)
          val = 1.0;
        val *= 1.0f;
        data[0][z*data_width*data_height + j*data_width + i] = val;
      }
    }
  }
#endif

#if COMBUSTION_DATASET
    data_width = 480;
    data_height = 720;
  //  data_depth = 120;
    data_depth = 8;
#endif

#if PV_DATASET
  data_width = data_height = 256;
  data_depth = 32;
#endif

#if OCEAN_DATASET
  data_width = 992;
  data_height = 1280;
  data_depth = 10;  //up to 42
#endif  
  
  //
  //create 3d shape -- vase
  //
#if VASE_DATASET
    data[0] = (float*)malloc(data_width*data_height*data_depth*sizeof(float));
 for(size_t k = 0; k < data_depth; k++) {
    for(size_t i = 0; i < data_width; i++) {
      for(size_t j = 0; j < data_height; j++) {
        float max_r = 0.6;
        float min_r = 0.3; 
        float x = float(i - data_width/2.0)/(data_width/2.0);
        float y = float(j - data_height/2.0)/(data_height/2.0);
        float z = float(k - data_depth/2.0)/(data_depth/2.0);
        float mod = M_PI/0.5;
        float val = (sin(y*mod)*.5 + .5);
        float r = val*(max_r-min_r) + min_r;
        if (sqrt(x*x+z*z) > r || sqrt(x*x+z*z) < r - 0.1) {  //make vase hollow
          data[0][i+j*data_width+k*data_height*data_width] = 0;
         continue;
        }
        val = sqrt(x*x+z*z);
//        else
//          val = 0.01;
        if (val < 0.0)
          val = 0.0;
        if (val > 1.0)
          val = 1.0;
        val *= 0.0001;
        data[0][i+j*data_width+k*data_height*data_width] = val;
      }
    }
  }
#endif

//
// load in tsi data
//
#if TSI_DATASET
  data_width = 320;
  data_height = 64;
  data_depth = 320;
  data[0] = (float*)malloc(320*320*320*sizeof(float));  
  FILE* fp = fopen("/data/scout/tsi/St12dE1193-little.raw", "rb");
  fread(data[0], sizeof(float), 320*320*320, fp);
  fclose(fp);
  
  //shift data
  for(int x = 0; x < data_width; x++) {
    for(int y = 0; y < data_height; y++) {
      for(int z = 0; z < data_depth; z++) {
        data[0][x+y*data_width + z*data_width*data_height] = data[0][x+(y+100)*320+z*320*320];
      }
    }
  }
  
  for(size_t i = 0; i < data_depth*data_width*data_height; i++) {
    data[0][i] *= 0.1; 
  }
#endif
  
  //
  // load in combustion dataset
  //

  
  //loop throught timesteps to find min/max : TODO: faster method of doing this
#if GLOBAL_MINMAX
  for(int i = 0; i < NUM_TIMESTEPS; i++) {
    loadTimestep(i);
  }
#endif
  loadTimestep(timestep);
  
  size_t max = data_width;
  if (data_height > max)
    max = data_height;
  if (data_depth > max)
    max = data_depth;
  
  max_bound = set_float3(float(data_width)/float(max), float(data_height)/float(max), float(data_depth)/float(max));
  
  center = divide(max_bound,2.0);
  
    setCamera();
    printf("camera_z: %f %f %f\n", camera_z.x, camera_z.y, camera_z.z);

  printf("done initializing scene\n");
  return true;
}

int last_x = 0, last_y = 0;

void slowClear() //clears buffer instead of setting flag, slow but no artifacts
{
  cudaMemcpy( d_pixel_buffer, pixel_buffer,window_width*window_height*sizeof(pixel_bufferT), cudaMemcpyHostToDevice  );
  cudaMemcpy( d_max_spp, &max_spp,sizeof(float), cudaMemcpyHostToDevice);
  pass_count = 1;
}

void clear()
{
  cudaMemcpy( d_pixel_buffer, pixel_buffer,window_width*window_height*sizeof(pixel_bufferT), cudaMemcpyHostToDevice  );
  //clearing = true;
  cudaMemcpy( d_max_spp, &max_spp,sizeof(float), cudaMemcpyHostToDevice);
  pass_count = 1;
}

void setCamera()
{
if (CAMERA_XY_AXIS)
{
    camera_pos.z = center.x + cos(rot_y)*cos(rot_x)*5.0;
    camera_pos.y = center.y + sin(rot_y)*5.0;
    
    camera_pos.x = center.z + sin(rot_x)*5.0;

    camera_z = normalize(center-camera_pos);
    
    camera_y = set_float3(0,1,0);
    camera_x = normalize(cross(camera_y, camera_z*-1.0f));
    camera_y = normalize(cross(camera_x, camera_z));
} else if (CAMERA_X_AXIS)
{
    camera_pos.z = center.z + cos(rot_x)*5.0;
    camera_pos.x = center.x + sin(rot_x)*5.0;
    camera_z = normalize(center-camera_pos);
    
    camera_y = set_float3(0,1,0);
    camera_x = normalize(cross(camera_y, camera_z*-1.0f));
    camera_y = normalize(cross(camera_x, camera_z));
} else
{
    camera_pos.z = center.z + cos(rot_y)*5.0;
    camera_pos.y = center.y + sin(rot_y)*5.0;
    camera_z = normalize(center-camera_pos);
    
    camera_y = set_float3(0,1,0);
    camera_x = normalize(cross(camera_y, camera_z*-1.0f));
//    camera_x = set_float3(1,0,0);
    camera_y = normalize(cross(camera_x, camera_z));
}
}

void recomputeGradient()
{
#if PRECOMPUTE_GRADIENT
   for(size_t z = 0; z < data_depth; z++) {
    //          std::cout  << "\033[1A ... [" << int((float(z)/float(this->shapeOf().dimOf(2)))*100) << "]%\n";
      for(size_t y = 0; y < data_height; y++) {
        for(size_t x = 0; x < data_width; x++) {
          size_t DELTA = 1;
          float3 lookup = {x,y,z};
          if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA ||
              lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA) 
            continue;
          float3 sample1, sample2;
          SET_FLOAT3(lookup,x-1,y,z);
          sample1.x = transferFunction->GetValue(data[0][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)]).w;
          SET_FLOAT3(lookup,x+1,y,z);
          sample2.x = transferFunction->GetValue(data[0][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)]).w;
          
          SET_FLOAT3(lookup,x,y-1,z);
          sample1.y = transferFunction->GetValue(data[0][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)]).w;
          SET_FLOAT3(lookup,x,y+1,z);
          sample2.y = transferFunction->GetValue(data[0][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)]).w;
          
          SET_FLOAT3(lookup,x,y,z-1);
          sample1.z = transferFunction->GetValue(data[0][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)]).w;
          SET_FLOAT3(lookup,x,y,z+1);
          sample2.z = transferFunction->GetValue(data[0][size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)]).w;
          float3 normal;
          FLOAT3_MINUS(sample1,sample2, normal);
         // normal = norm3h(normal);
         // if (normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0)
          //  SET_FLOAT3(normal,0,0,-1);
          gradient[0][size_t(z*data_width*data_height + y*data_width + x)] = normal;
        }
      }
    } 
#if BLUR_GRADIENT
        blurGradient();
#endif  
//   cudaMemcpy( d_gradient, gradient[0],data_width*data_height*data_depth*sizeof(float3), cudaMemcpyHostToDevice  );
#endif
}

bool alteringTF = false;

void motion(int x, int y)
{
  if (alteringTF) {
    for(int i =0 ; i < 100; i++) {
        int tx = last_x + (x - last_x)*(i/100.0f);
        int ty = last_y + (y - last_y)*(i/100.0f);
        if (tfRenderable->Contains(tx,window_height - ty))
          tfRenderable->ProcessMouse(tx,window_height -ty);
     }
     clear();
  }
  float scale = 0.001;
  if (rotating) {
    rot_x += (x-last_x)*scale;
    rot_y += (y-last_y)*scale;
    
    setCamera();

//    printf("setting camera: %f %f %f\n", camera_pos.x, camera_pos.y, camera_pos.z);
//    printf("magnitude of camera from center: %f\n", float(magnitude(minus(camera_pos, center))));
//    printf("camera rotx roty: %f %f\n", rot_x, rot_y);
    clear();
  }
  last_x = x;
  last_y = y;
}


void mouse(int button, int state, int x, int y)
{
  if (button == GLUT_RIGHT_BUTTON) {
#if USE_TF
      if (state == GLUT_DOWN)
      {
        alteringTF = true;
        dirty_gradient = true;
        last_x = x;
        last_y = y;
        }
      else {
        cudaMemcpy(d_TFBuckets, transferFunction->buckets, sizeof(float4)*transferFunction->numBuckets, cudaMemcpyHostToDevice);
        alteringTF = false;
        }
#endif
  }
  else if (button == GLUT_LEFT_BUTTON) {
    last_x = x;
    last_y = y;
    static float original_num;
    if (state == GLUT_DOWN) {
      rotating = true;
      original_num = num_rays;
      num_rays = NUM_RAYS/MOUSE_SPEEDUP;
    }
    else {
      rotating = false;
      num_rays = original_num;
      slowClear();
    }
  }
}

void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
  if (key == 'c') {
    cutoff = (cutoff+1)%7;
    slowClear();
    
      switch(cutoff)
  {
    case 1:
      printf("switched cuttoff to: positive x\n");
      break;
    case 2:
      printf("switched cuttoff to: negative x\n");
      break;
    case 3:
      printf("switched cuttoff to: positive y\n");
      break;
    case 4:
     printf("switched cuttoff to: negative y\n");
      break;
    case 5:
      printf("switched cuttoff to: pinhole \n");
      break;
    case 6:
      printf("switched cuttoff to: inverse pinhole \n");
      break;
    default:
      printf("switched cuttoff to: none \n");
  }
  }
  else if (key =='g') {
    dirty_gradient = false;
    recomputeGradient();
    clear();
  }
  else if (key =='r') {
    orbit = !orbit;
  }
  else if (key == 'x') {
    pass_count = 0;
    cudaMemcpy( d_pixel_buffer, pixel_buffer,window_width*window_height*sizeof(pixel_bufferT), cudaMemcpyHostToDevice  );
  }
  else if (key == 'q') {
    exit(0);
  }
}

extern "C"
{
void run(int argc, char** argv, int rank, int size, int sizex, int sizey, int sizez, char** input_files, int num_files, int filemin, int filemax, DataFile** data_files, float stepsize, bool record)
{
  recording = record;
  step_size = stepsize;
  file_width = sizex;
  file_height = sizey;
  file_depth = sizez;
  files = input_files;
  file_data = data_files;
  file_min = filemin;
  file_max = filemax;
  mpi_rank = rank;  mpi_size = size;
  END_TIMESTEP = num_files;
  NUM_TIMESTEPS = num_files;
  timestep = rank;

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "CD CUDA Schlieren");

    // initialize GL
    if( false == initGL()) {
        exit(1);
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);


    // start rendering mainloop
    glutMainLoop();
}
}
