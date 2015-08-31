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



#ifndef KRENEL_CUTOFF_H
#define KRENEL_CUTOFF_H

__device__ void kernel_cutoff(RenderParameters& params, float2 offset, float phase_shift, float3& color)
{
//    float m = length(offset);
//    color = make_float3(m,m,m)*100.0;

    float kill_chance = 0;
    float scale = params.cutoffScalar;
    offset *= scale;
      switch(params.cutoff)
      {
        case 1:
          kill_chance = offset.x;
          break;
        case 2:
          kill_chance = -offset.x;
          break;
        case 3:
          kill_chance = offset.y;
          break;
        case 4:
          kill_chance = -offset.y;
          break;
        case 5:
          kill_chance = length(offset);
          break;
        case 6:
          kill_chance = 1.0-length(offset);
          break;
      case CUTOFF_INTERFOREMETRY:
          kill_chance = sin(phase_shift);
          break;
      case CUTOFF_IMAGE:
          {
              float4 c = tex2D(tex_cutoff, offset.x + .5, offset.y + .5);
              color = make_float3(c.x,c.y,c.z);
              return;

          }
      case CUTOFF_TRACE:
          {
              color = make_float3(offset.x+.5,offset.y+.5,0);
              return;
          }
      default:
          kill_chance = 0; //half the light to match cutoffs
      }

      float save_chance = 1.0f;
    //   #if !WEIGHT_VALUES
    //      if (kill_chance > randt - .5f)
    //           continue;
    //   #else
         save_chance = 0.5f - kill_chance;
//         save_chance = 0.5f - sin(phase_shift);
       if(save_chance < 0.0f)
               save_chance = 0.0f;
       if (save_chance > 1.0f)
               save_chance = 1.0f;
    //   #endif
    //
    //   #if INTERFOREMETRY
       if (params.cutoff == CUTOFF_INTERFOREMETRY)
 save_chance = sin(phase_shift);

    //   #endif

color = make_float3(save_chance,save_chance,save_chance);
//
//    float3 base = camera_y;
//    float mag = length(offset)*scale;
//    if (mag > 1.0f)
//      mag = 1.0f;
//    float3 norm_offset = normalize(offset);
//    float3 color_dir = hsv(angleBetween(norm_offset, base), 1, mag);

}

#endif // KRENEL_CUTOFF_H
