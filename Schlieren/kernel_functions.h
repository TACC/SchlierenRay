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





#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H

__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__device__ void print3(float3 v, char* msg)
{
#if __DEVICE_EMULATION__
  printf(msg);
  printf(": %f %f %f\n", v.x, v.y, v.z);
#endif
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


 __device__ float3 proj3(const float3& v1, const float3& v2) //project v1 onto v2
 {
   return v2*dot(v1,v2);
 }

 //__device__ const float toDegrees = 180.0/M_PI;

 __device__ float angleBetween(const float3& v1, const float3& v2)
 {
   float dp = dot(v1, v2);
   return acos(dp)*180.0/M_PI;
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
     return make_float3(v,t,p);
   }
   else if( hi == 1){
     return make_float3(q,v,p);
   }
   else if( hi == 2){
     return make_float3(p,v,t);
   }
   else if( hi == 3){
     return make_float3(p,q,v);
   }
   else if( hi == 4){
     return make_float3(t,p,v);
   }
   else{
     return make_float3(v,p,q);
   }
 }


#endif // KERNEL_FUNCTIONS_H
