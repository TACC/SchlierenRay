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




#ifndef _KERNEL_H_
#define _KERNEL_H_


#define uint unsigned int

//#include "MersenneTwister_kernel.cu"
#if USE_ATOMIC
#include "sm_11_atomic_functions.h"
#endif

#include "kernel_functions.cu"

__shared__ __device__ float svalues[256];
__shared__ __device__ unsigned int sindices[256];

__global__ void kernel_volume(const int num_rays, const int data_width, const int data_height, const int data_depth, const float3 lookup_scale,
  float* data, const int window_width, const int window_height,pixel_bufferT* pixel_buffer,
  float* max_spp, float3 min_bound,
  float3 max_bound, const float3 camera, float3 camera_x, float3 camera_y, const float3 camera_z, const float3 camera_corner, const float* rand_x, const float* rand_y, const float* rand_t
#if PRECOMPUTE_GRADIENT
  ,const float3* gradient
#endif
#if COLOR_ON_OTHER_DATA
  ,float* data2
#endif
  , const int cutoff, const float data_min, const float data_max, const float data_scale, float rand_x2, float rand_y2, float rand_t2, bool clearing,
#if OCTREE
unsigned char* octree,
#endif //OCTREE
#if USE_TF
float4* TFBuckets, int TFNumBuckets, float TFMin, float TFScale,
#endif
bool dirty_gradient, float step_size)
{

#if THREAD_SAFE
if (threadIdx.x == 0 && threadIdx.y == 0){
	for(int i = 0; i < blockDim.x*blockDim.y; i++) {
	  sindices[i] = 0 ;
	  svalues[i] = 0;
	}
}
__syncthreads();
#endif

float max_scale = max(max(float(data_width), float(data_height)), float(data_depth));
float scaler = 1.0/max_scale;
  int winsize = window_width*window_height;
  float level_scales[] = {scaler,scaler*2, scaler*4, scaler*8,scaler*16,scaler*32,scaler*64,scaler*128};
  float base_step_size = step_size;
#if COLOR_ACCUMULATE
  float4 color = {0,0,0,0};
#endif
  float3 dir = camera_z*-1.0f;
  float3 original_dir = normalize(dir);
  unsigned int win_x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int win_y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int sindex = threadIdx.y*blockDim.x + threadIdx.x;// shared memory index
  unsigned int r_x = win_x + win_y*window_width;
  unsigned int index = r_x%(winsize);
  float shift = float(r_x)/float(winsize);
  int count = 0;
//  if (threadIdx.x == 0)
//	return;

//  if (r_x < 0 || r_x > window_width*window_height)
//	return;
//  pixel_buffer[r_x]++;
  //return;

for(int i = 0; i < NUM_PER_THREAD; i++, count++) {
  float kill_chance = 0;
  float phase_shift = 0;
   float randx = rand_x[int(r_x*rand_t2 + blockIdx.x + blockIdx.y + i + rand_x2*winsize)%winsize];
   float randy = rand_x[int(r_x + blockIdx.x + blockIdx.y + rand_t2*win_x + i + rand_y2*winsize)%winsize];
   float randt = rand_x[int(r_x + blockIdx.x + blockIdx.y + i + rand_t2*winsize)%winsize];
  count++;
  /*  float randx = rand_x2;
  float randy = rand_y2;
  float randt = rand_t2;
  */
  float nextp[] = {min_bound.x,min_bound.y,min_bound.z};
  if (index >= num_rays)
    continue;
  float3 pos;
  dir = original_dir;
  pos = camera+camera_z*9.0f;
  if (clearing) {
#if COLOR
    pixel_buffer[index] = set4(0,0,0,0);
#else
    pixel_buffer[index] = 0;
#endif
        clearing = false;
  }
  float pos_x = float(win_x%window_width)/float(window_width) - .5f;
  float cell_width = 1.0f/float(window_width);
  float cell_height = 1.0f/float(window_height);
  float pos_y = float(win_y%window_height)/float(window_height) - .5f;
  float offx = cos(randx*M_PI*2.0);
  float offy = sin(randx*M_PI*2.0);
  offx = offx*randy*cell_width*0.5;
  offy = offy*randy*cell_height*0.5;
  pos = pos+camera_x*pos_x + camera_x*(randx*1.0f)*cell_width;
  pos = pos+camera_y*pos_y + camera_y*(randy*1.0f)*cell_height;
//  pos = pos+camera_x*pos_x + camera_x*(offx);
//  pos = pos+camera_y*pos_y + camera_y*(offy);

  IntersectWithVolume(pos, dir, min_bound, max_bound);

  pos = pos+dir*randt*step_size;

  float3 previous_normal, real_normal;
  float3 normal = {0.f,0.0f,-1.f};
  float data_depthf = 16.0f;
  int steps = 1.4f/(step_size);
  float old_index = 1.0;
  size_t DELTA = 1;
  int i = 0;
  for(i = 0; i < steps; ++i) {
    pos = pos + dir*step_size/old_index;
    float3 offset = pos-min_bound;
    float3 lookupfn = offset*lookup_scale; // normalized lookup
    float3 lookupf = {lookupfn.x*float(data_width), lookupfn.y*float(data_height), lookupfn.z*float(data_depth)};
    float3 lookup = {static_cast<uint>(lookupfn.x*data_width), static_cast<uint>(lookupfn.y*data_height), static_cast<uint>(lookupfn.z*data_depth) };

    print3(pos,"pos: ");
    if(pos.x <= min_bound.x || pos.y <= min_bound.y || pos.z <= min_bound.z ||
      pos.x >= max_bound.x || pos.y >= max_bound.y || pos.z >= max_bound.z )
      break;

    if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA ||
            lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA)
          continue;

      float val = 0.0f;

#if TRILINEAR_INTERPOLATION
  //val = trilinearInterpolation<float>(data, lookupf, data_width, data_height, data_depth);
    val = tex3D(tex_data, lookup.x, lookup.y, lookup.z);
#else
  //val = DATA_GET(lookup.x, lookup.y, lookup.z);
   val = tex3D(tex_data, lookup.x, lookup.y, lookup.z);
#endif //TRILINEAR_INTERPOLATION
    size_t data_index = lookup.x + lookup.y*data_width + lookup.z*data_width*data_height;
#if OCTREE
   //step_size = base_step_size * exp2f(level);
   //OCTREE Code removed for now
#endif

#if PRECOMPUTE_GRADIENT

#if INTERPOLATE_GRADIENT
#if !USE_TF
  normal = GRADIENT_GET(lookupf.x, lookupf.y, lookupf.z);
#else
  if (!dirty_gradient) {
  normal = GRADIENT_GET(lookupf.x, lookupf.y, lookupf.z);
  } else {
int x = lookup.x;
    int y = lookup.y;
    int z = lookup.z;
    float3 sample1, sample2;
    lookup = set3(x-1,y,z);
      sample1.x = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));
    lookup = set3(x+1,y,z);
      sample2.x = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));

    lookup = set3(x,y-1,z);
     sample1.y = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));
    lookup = set3(x,y+1,z);
      sample2.y = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));

    lookup = set3(x,y,z-1);
       sample1.z = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));
    lookup = set3(x,y,z+1);
      sample2.z = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));
normal = sample1-sample2;
  }
#endif// USE_TF

#else
  normal = GRADIENT_GET(lookupf.x, lookupf.y, lookupf.z);
#endif  //INTERPOLATE_GRADIENT

#else  // PRECOMPUTE_GRADIENT
    int x = lookup.x;
    int y = lookup.y;
    int z = lookup.z;
    float3 sample1, sample2;
    lookup = set3(x-1,y,z);
#if USE_TF
        sample1.x = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));
    lookup = set3(x+1,y,z);
        sample2.x = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));

    lookup = set3(x,y-1,z);
       sample1.y = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));
    lookup = set3(x,y+1,z);
      sample2.y = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));

    lookup = set3(x,y,z-1);
      sample1.z = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));
    lookup = set3(x,y,z+1);
      sample2.z = GetTFRIValue(TFBuckets, TFNumBuckets, TFMin, TFScale, tex3D(tex_data, lookup.x, lookup.y, lookup.z));

#else
       sample1.x = tex3D(tex_data, lookup.x, lookup.y, lookup.z);
    lookup = set3(x+1,y,z);
       sample2.x = tex3D(tex_data, lookup.x, lookup.y, lookup.z);

    lookup = set3(x,y-1,z);
       sample1.y = tex3D(tex_data, lookup.x, lookup.y, lookup.z);
    lookup = set3(x,y+1,z);
      sample2.y = tex3D(tex_data, lookup.x, lookup.y, lookup.z);

    lookup = set3(x,y,z-1);
       sample1.z = tex3D(tex_data, lookup.x, lookup.y, lookup.z);
    lookup = set3(x,y,z+1);
        sample2.z = tex3D(tex_data, lookup.x, lookup.y, lookup.z);
#endif //USE_TF
    normal = sample1-sample2;
#endif //PRECOMPUTE_GRADIENT

//  bool valid = true;
//  float min_val = 0.2*.5, max_val = 0.8*0.5;
//  if (!((val > min_val && val < max_val) && (old_index < 1.0+min_val || old_index > 1.0+max_val) ) )
//    valid = false;  //DEBUG
  old_index = val+1.0f;
// if (!valid)
//        continue; //DEBUG
   if (val < 0.0f)
        continue;
#if !LINE_OF_SIGHT
    dir = dir + step_size*normal;
#else
    float dx = dot(normal, camera_x)*step_size;
    //float dy = dot(normal, camera_y)*step_size;
    kill_chance += (dx);
#endif
    phase_shift += val-1.0;

#if COLOR_ON_OTHER_DATA
  val = trilinearInterpolation<float>(data2, lookupf, data_width, data_height, data_depth);
#endif
#if COLOR_ACCUMULATE
   float norm_val = (val-data_min)*data_scale*1.0;
#if COLOR_OPACITY
   float3 c = hsv(norm_val*260, 1, norm_val*10.0);
#else
   float3 c = hsv(norm_val*260, 1, 1);
#endif
   // float alpha = norm_val*step_size*100.0;
   float alpha = 1.0;
#if COLOR_OPACITY
   if (color.w == 0.0)
        color.w = 1.0;
   color += make_float4(c.x*alpha,c.y*alpha,c.z*alpha,norm_val*10);
#else
        color += make_float4(c.x*alpha,c.y*alpha,c.z*alpha,1);
#endif //COLOR_OPACITY
#endif //COLOR_ACCUMULATE
  }
    if (dot(camera_z*-1.0f, normalize(dir)) < 0.0)
    continue;

#if PROJECT_TO_FILM
    //project to film plane
    float3 film_pos = camera;
    IntersectWithVolume(film_pos, camera_z, min_bound, max_bound);
    film_pos = film_pos+camera_z*-PROJECTION_DISTANCE;
  IntersectWithPlane(pos, dir, camera_z, film_pos);
#endif

#if INTERFOREMTRY
  phase_shift *= 2.0*M_PI/(500e-9);
#endif
  float scale = KILL_SCALE;
  float3 offset = normalize(dir)-original_dir;
  float3 corner_offset = pos-camera_corner;
  float signedx =   dot(offset,camera_x);
  float signedy =   dot(offset,camera_y);
//  float3 dirxoffset = proj3(offset, camera_x);
//  float3 diryoffset = proj3(offset, camera_y);
  float3 xoffset = proj3(corner_offset, camera_x);
  float3 yoffset = proj3(corner_offset, camera_y);
//  float sign = 1.0f;
//  if (dot(offset, camera_x) < 0)
//        sign = -1.0f;
//  float signedx = (length(dirxoffset))*sign;
//  sign = 1.0f;
//  if (dot(offset, camera_y) < 0)
//        sign = -1.0f;
//  float signedy = (length(diryoffset))*sign;
#if !LINE_OF_SIGHT
  switch(cutoff)
  {
    case 1:
      kill_chance = signedx*scale;
      break;
    case 2:
      kill_chance = -signedx*scale;
      break;
    case 3:
      kill_chance = signedy*scale;
      break;
    case 4:
      kill_chance = -signedy*scale;
      break;
    case 5:
      kill_chance = length(offset)*0.1f;
      break;
    case 6:
      kill_chance = 1.0-length(offset)*scale;
      break;
    default:
      kill_chance = 0; //half the light to match cutoffs
  }
#else
  kill_chance *= scale;
#endif
float save_chance = 1.0f;
#if !WEIGHT_VALUES
   if (kill_chance > randt - .5f)
        continue;
#else
  save_chance = 0.5f - kill_chance;
if(save_chance < 0.0f)
        save_chance = 0.0f;
if (save_chance > 1.0f)
        save_chance = 1.0f;
#endif

#if INTERFOREMETRY
  save_chance = sin(phase_shift);
#endif

  unsigned int w_x = length(xoffset)*window_width;
  unsigned int w_y = length(yoffset)*window_height;
  unsigned long long win_index = w_y*window_width + w_x;
  if (w_x < 0 || w_y < 0 || w_x >= window_width || w_y >= window_height)
     continue;

#if COLOR
#if COLOR_ACCUMULATE
  color.x /= color.w;
  color.y /= color.w;
  color.z /= color.w;
  pixel_buffer[win_index].x += color.x*save_chance;
  pixel_buffer[win_index].y += color.y*save_chance;
  pixel_buffer[win_index].z += color.z*save_chance;
  pixel_buffer[win_index].w += 1.0f*save_chance;
#elif COLOR_DIRECTION
  float3 base = camera_y;
  float mag = length(offset)*scale;
  if (mag > 1.0f)
    mag = 1.0f;
  float3 norm_offset = normalize(offset);
  float3 color_dir = hsv(angleBetween(norm_offset, base), 1, mag);
  pixel_buffer[win_index].x += color_dir.x*save_chance;
  pixel_buffer[win_index].y += color_dir.y*save_chance;
  pixel_buffer[win_index].z += color_dir.z*save_chance;
  pixel_buffer[win_index].w += 1.0f*save_chance;
#endif //COLOR_ACCUMULATE or COLOR_DIRECTION
     setMax(max_spp, (float)(pixel_buffer[win_index].w));
#else  // not COLOR
#if GUASSIAN_FILTER
  //guassian weighting
  float val = 1.0f;
  float pixel_width = 1.0f/float(window_width);
  float pixel_height = 1.0f/float(window_height);
  float fx = ((length(proj3(corner_offset,camera_x))*window_width- w_x) - .5f)*2.0f;
  float fy = ((length(proj3(corner_offset,camera_y))*window_height- w_y) - .5f)*2.0f;
  val *= (pow(2.0f, -2.0f*fx*fx) + pow(2.0f, -2.0f*fy*fy))/2.0f;

    pixel_buffer[win_index] = pixel_buffer[win_index] + val*save_chance;
  if (pixel_buffer[win_index] > *max_spp)
    *max_spp = pixel_buffer[win_index];
#elif WEIGHT_VALUES

#if THREAD_SAFE
   sindices[sindex] = win_index;
   svalues[sindex] = save_chance;

#else

 if (win_index == 0)
        continue;
  pixel_buffer[win_index] += save_chance;
  if (pixel_buffer[win_index] > *max_spp)
    *max_spp = pixel_buffer[win_index];

#endif // THREAD_SAFE

#else  //NO GUASSIAN FILTER OR WEIGHTING
//TODO: this is a hacked fix, why are so many values mapping to 0?
  if (win_index == 0)
        continue;
  pixel_buffer[win_index]++;
  if (pixel_buffer[win_index] > *max_spp)
    *max_spp = pixel_buffer[win_index];
#endif  //GUASSIAN_FILTER
#endif // COLOR
}

#if THREAD_SAFE
__syncthreads();
if (threadIdx.x == 0 && threadIdx.y == 0){
    unsigned int num = blockDim.x*blockDim.y;
	for(int i = 0; i < num; i++) {
// int i = sindex;
	float val = svalues[i];
	unsigned int ind = sindices[i];
	if (ind <  winsize) {

	  pixel_buffer[ind] += val;

	  if (val > *max_spp)
		*max_spp = val;
	  }
	}
	}
#endif
}

#endif
