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





#ifndef MAIN_H
#define MAIN_H

#include <vector_types.h>
#include <vector_functions.h>
#include <cuda.h>
#include "opengl_include.h"
#include <float.h>
#include <assert.h>

typedef struct {
  float* data;
  char* filename;
  int sizex, sizey, sizez;
} DataFile;

class TransferFunction
{
public:
  TransferFunction(float min_, float max_, int numBuckets_)
   : min(min_), max(max_), numBuckets(numBuckets_)
  {
    scale = 1.0f/(max-min);
    buckets = new float4[numBuckets_];
    for(int i = 0; i < numBuckets_; i++)
      buckets[i] = make_float4(0.5f, 0.5f, 0.5f, min + float(i)/float(numBuckets)/scale);
  }
  float4 GetValue(float v)
  {
    float vi = (v-min)*scale*numBuckets;
    size_t index = vi;
    if (index >= numBuckets-1)
      index = numBuckets-2;
    if (index < 0)
      index = 0;
    float rv = vi-index;
    float lv = 1.0f - rv;
    float4 v1 = buckets[index];
    float4 v2 = buckets[index+1];
    float4 result = v1;
    result.x = v1.x*lv + v2.x*rv;
    result.y = v1.y*lv + v2.y*rv;
    result.z = v1.z*lv + v2.z*rv;
    result.w = v1.w*lv + v2.w*rv;
    return (result);
  };
  float4* buckets;
  float scale;
  float min, max;
  int numBuckets;
};

class TFRenderable
{
 public:
  TFRenderable(TransferFunction* tf_, int x_, int y_, int w_, int h_)
   : tf(tf_), x(x_), y(y_), w(w_), h(h_)
    {
      pixels = new unsigned char[w_*h_*3];
    }

  void Draw()
    {
      for(int i = 0; i < w; i++) {
          float4 val = tf->buckets[i];
          int yval = (val.w-tf->min)*tf->scale*h;
        for(int j = 0; j < h; j++) {
          size_t index = (i+j*w)*3;
          pixels[index] = val.x*255;
          pixels[index+1] = val.y*255;
          pixels[index+2] = val.z*255;
          if (j == yval) {
            pixels[index] = pixels[index+1] = pixels[index+2] = 255;
          }
        }
      }
      //      glRasterPos2i(1,0);
      glDrawPixels(w,h,GL_RGB, GL_UNSIGNED_BYTE, pixels);
      // glRasterPos2i(0,0);
    }

  bool Contains(int x_, int y_)
    {
      if (x_ < x || x_ > x + w)
        return false;
      if (y_ < y || y_ > y + h)
        return false;
      return true;
    }

  void ProcessMouse(int x_, int y_)
    {
      if (Contains(x_,y_)) {
        float ypos = (y_ - y)/float(h);
        int bucket = (x_-x);
        tf->buckets[bucket].w = ypos/tf->scale+tf->min;
      }
    }

 protected:
  int w, h,x,y;
  TransferFunction* tf;
  unsigned char* pixels;
};

#endif
