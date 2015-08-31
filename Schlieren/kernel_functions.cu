#ifndef KERNEL_FUNCTIONS_CU
#define KERNEL_FUNCTIONS_CU

#include "cutil.h"
#include "cutil_math.h"
#include <cuda.h>
#include "cuda_gl_interop.h"
#include <cstdlib>
#include <float.h>

texture<float, 3> texRIRef;
texture<float3, 3> texGradientRef;
texture<float4, 3> texRef; //gradient and RI values

texture<float, 3> tex_data;
texture<float, 3> tex_gradient_x;
texture<float, 3> tex_gradient_y;
texture<float, 3> tex_gradient_z;


#define DATA_GET(x,y,z) data[size_t(x + y*data_width + z*data_width*data_height)]
//#define GRADIENT_GET(x,y,z) gradient[size_t(x + y*data_width + z*data_width*data_height)] 
#define GRADIENT_GET(x,y,z) make_float3(tex3D(tex_gradient_x, x,y,z), tex3D(tex_gradient_y, x,y,z), tex3D(tex_gradient_z, x,y,z) )

texture<float, 3, cudaReadModeElementType> tex;


template<typename T>
__device__ T trilinearInterpolation(T* data, float3 lookupf, const size_t data_width, const size_t data_height, const size_t data_depth)
{
  int xf = floor(lookupf.x), yf = floor(lookupf.y), zf = floor(lookupf.z);
  int xc = ceil(lookupf.x), yc = ceil(lookupf.y), zc = ceil(lookupf.z);
  float xd = lookupf.x - floor(lookupf.x);
  float xd1 = 1.0 - xd;
  float yd = lookupf.y - floor(lookupf.y);
  float yd1 = 1.0 - yd;
  float zd = lookupf.z - floor(lookupf.z);
  float zd1 = 1.0-zd;
  
  T i1 = DATA_GET(xf, yf, zf)*zd1 + DATA_GET(xf,yf,zc)*zd;
  T i2 = DATA_GET(xf,yc,zf)*zd1+DATA_GET(xf,yc,zc)*zd;
  T j1 = DATA_GET(xc,yf,zf)*zd1+DATA_GET(xc,yf,zc)*zd;
  T j2 = DATA_GET(xc,yc,zf)*zd1+DATA_GET(xc,yc,zc)*zd;
  
  T w1 = i1*yd1+i2*yd;
  T w2 = j1*yd1+j2*yd;
  
  return w1*xd1+w2*xd;
}

__device__ void print3(float3 v, char* msg)
{
#if DEBUG
  printf(msg);
  printf(": %f %f %f\n", v.x, v.y, v.z);
#endif
}

__device__ float3 set3(float x, float y, float z)
{
  float3 result = {x,y,z};
  return result;
}

__device__ float4 set4(float x, float y, float z, float w)
{
  float4 result = {x,y,z,w};
  return result;
}


__device__ float3 proj3(const float3& v1, const float3& v2) //project v1 onto v2
{
  return v2*dot(v1,v2);
}

__device__ const float toDegrees = 180.0/M_PI;

__device__ float angleBetween(const float3& v1, const float3& v2)
{
  float dp = dot(v1, v2);
  return acos(dp)*toDegrees;
}

// ------------
// --- hsva ---
// ------------
//
__device__ float3 hsv(float h, float s, float v)
{
  int hi = floor(h / 60.0f);
  float f = h/60.0f - (float)hi;
  hi = hi % 6;
  float p = v * (1.0f -s );
  float q = v * ( 1.0f - f * s);
  float t = v * (1.0f - (1.0f -f ) * s);
  if (hi == 0){
    return set3(v,t,p);
  }
  else if( hi == 1){
    return set3(q,v,p);
  }
  else if( hi == 2){
    return set3(p,v,t);
  }
  else if( hi == 3){
    return set3(p,q,v);
  }
  else if( hi == 4){
    return set3(t,p,v);
  }
  else{
    return set3(v,p,q);
  }
}

__device__ bool IntersectWithPlane(float3& ray_pos, float3 ray_dir, float3 normal, float3 pos)
{
  float d = -dot(normal, pos);
  float nDotO = dot(normal,ray_pos);
  float nDotV = dot(normal, ray_dir);
  if (fabs(nDotV) < 0.00001 || nDotV > 0.0)
    return false;
  float t = -(d + nDotO)/(nDotV);
  if (t < 0.00001)
    return false;
  ray_pos = ray_pos+ray_dir*t;
  return true;
}

 __device__ bool IntersectWithVolume(float3& ray_pos, float3 ray_dir, float3 p1, float3 p2)
{
  float d1 = p1.x;
  float d2 = p2.x;
  float tnear =  - (FLT_MAX - 1);
  float tfar = FLT_MAX;
  float t1= (d1 - ray_pos.x)/(ray_dir.x);
  float t2 = (d2 - ray_pos.x)/(ray_dir.x);
  if (t1 >t2)
  {
    float temp = t1;
    t1 = t2;
    t2 = temp;
  }
  if (t1 > tnear)
    tnear = t1;
  if (t2 < tfar)
    tfar = t2;
  if (tnear > tfar) //miss
    return false;
  if (tfar < 0.0) // box behind ray
    return false;
  
  t1= (p1.y - ray_pos.y)/(ray_dir.y);
  t2 = (p2.y - ray_pos.y)/(ray_dir.y);
  if (t1 >t2)
  {
    float temp = t1;
    t1 = t2;
    t2 = temp;
  }
  if (t1 > tnear)
    tnear = t1;
  if (t2 < tfar)
    tfar = t2;
  if (tnear > tfar) //miss
    return false;
  if (tfar < 0.0) // box behind ray
    return false;
  
  t1= (p1.z - ray_pos.z)/(ray_dir.z);
  t2 = (p2.z - ray_pos.z)/(ray_dir.z);
  if (t1 >t2)
  {
    float temp = t1;
    t1 = t2;
    t2 = temp;
  }
  float t;
  if (t1 >= 0 && t1 > tnear)
    tnear = t1;
  if (t2 < tfar)
    tfar = t2;
  if (tnear > tfar) //miss
    return false;
  if (tfar < 0.0) // box behind ray
    return false;
  else if (tnear < 0)  //I dunnno if this is right... put this in for rays starting in box
    t = tfar;
  else
    t = tnear;
  ray_pos = ray_pos+ray_dir*t;
  return true;
}  


__device__ void setInc(float* old)
{
#if USE_ATOMICS
      atomicInc(old, 9999999);  
#else
        *old+=1;
#endif
}

__device__ void setInc(int* old)
{
#if USE_ATOMICS
      atomicInc(old, 9999999);  
#else
        *old+=1;
#endif
}

__device__ void setInc(unsigned int* old)
{
#if USE_ATOMICS
      atomicInc(old, 9999999);  
#else
        *old+=1;
#endif
}

__device__ void setMax(unsigned int* i1, unsigned int i2)
{
#if USE_ATOMICS       
     atomicMax(i1, i2);
#else
     if (*i1 < i2)
        *i1 = i2;
#endif
}     

__device__ void setMax(float* i1, unsigned int i2)
{
#if USE_ATOMICS       
     atomicMax(i1, i2);
#else
     if (*i1 < i2)
        *i1 = i2;
#endif
}     

__device__ float4 GetTFValue(float4* TFBuckets, int TFNumBuckets, float TFMin, float TFScale, float v)
{
  int index = (v-TFMin)*TFScale*TFNumBuckets;
  if (index >= TFNumBuckets)
        index = TFNumBuckets-1;
  if (index < 0)
        index = 0;
  return TFBuckets[index];
}

__device__ float GetInterpolatedTFRIValue(float4* TFBuckets, int TFNumBuckets, float TFMin, float TFScale, float v)
{
  float iv = (v-TFMin)*TFScale*TFNumBuckets;
  int index = int(iv);
  float rv = iv - index;
  float lv = 1.0f - rv;
  if (index >= TFNumBuckets - 1)
        index = TFNumBuckets- 2;
  if (index < 0)
        index = 0;
  return (TFBuckets[index]*lv + TFBuckets[index+1]*rv).w;
}

__device__ float GetTFRIValue(float4* TFBuckets, int TFNumBuckets, float TFMin, float TFScale, float v)
{
  float iv = (v-TFMin)*TFScale*TFNumBuckets;
  size_t index = int(iv);
  if (index >= TFNumBuckets)
        index = TFNumBuckets- 1;
  if (index < 0)
        index = 0;
  return (TFBuckets[index]).w;
}

#endif
