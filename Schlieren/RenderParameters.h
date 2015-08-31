#ifndef RENDERPARAMETERS_H
#define RENDERPARAMETERS_H

//#include "cutil.h"
//#include "cutil_math.h"
//#include <cuda.h>
////#include "cuda_gl_interop.h"
//#include <cstdlib>
//#include <float.h>
#include <vector_types.h>

enum cutoffs {CUTOFF_NONE = 0, CUTOFF_KNIFE_X = 1, CUTOFF_KNIFE_NEGATIVE_X,
              CUTOFF_KNIFE_Y, CUTOFF_KNIFE_NEGATIVE_Y, CUTOFF_PINHOLE, CUTOFF_INVERSE_PINHOLE,
              CUTOFF_INTERFOREMETRY, CUTOFF_IMAGE, CUTOFF_TRACE};

 struct RenderParameters
{
    //texture<float4, 3> tex_data;  //gradient and refractive index
    unsigned int data_width, data_height, data_depth;
    float data_min;
    float4* data;
  float4* color_data;  //used for color accululation of volume
    float* data2;
    float4* inout_rgb;  //color information on film plane
    unsigned int* out_rgb; //filtered output texture
    unsigned int width, height;
    float3 min_bound, max_bound;
    float3 camera_corner, camera_pos, camera_x, camera_y, camera_z;
    float* random_array;
    float rand1, rand2, rand3, rand4;
    float stepSize, projectionDistance;
    unsigned int raysPerPixel, numRenderPasses;
    float dataScalar;
    float cutoffScalar;

    bool threadSafe, useOctree, useRefraction;
    int cutoff;
    bool cutoff_dirty;  //needs to be re-uploaded to card

float4* cutoff_rgb;  //image of filter
uint2 cutoffSize;
unsigned int passes;
};


#endif // RENDERPARAMETERS_H
