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





#ifndef KERNEL_FILTER_H
#define KERNEL_FILTER_H

__global__ void kernel_filter(RenderParameters* paramsp, float4* rgb_in, unsigned int* rgb_out)
{
  float max_spp = paramsp->passes+paramsp->raysPerPixel*paramsp->numRenderPasses;//(*max_spp_real); // increase brightness
  int image_width = paramsp->width;
  int image_height = paramsp->height;
//printf("normalization called\n");
/*  unsigned int r_x = blockIdx.x*blockDim.x + threadIdx.x;
  int num_pixels_per_thread = NUM_PER_THREAD*10; //doing one per thread will die
  for (int x = r_x*num_pixels_per_thread; x < (r_x+1)*num_pixels_per_thread; x++) {
    if (x >= num_pixels)
       return;*/

    int p_x = blockIdx.x*blockDim.x + threadIdx.x;
    int p_y = blockIdx.y*blockDim.y + threadIdx.y;


    unsigned int index = p_x + p_y*paramsp->width;
 //   float4 c = rgb_in[index]*255.0/max_spp;
//    rgb_out[index] = rgbToInt(c.x,c.y,c.z);

   //figure out radius of filter
    float radius = 0.0f;
    int max_level = 6;
    float k = 1.15; //weighting scale >= 1.0
    float nscale = (64 - max_spp)*(paramsp->raysPerPixel*paramsp->numRenderPasses/max_spp);
    if (nscale < 1.0)
      nscale = 0.5;
    float N = (max_spp)*nscale; //number of samples to look for when filtering
    float total_count = 0;
    bool done = false;
    for(int level = 0;!done;level++) {
      radius += 1.0f;
      for(int s_y = -level; s_y <= level; s_y++) {
        int b_y = p_y + s_y;
        int increment = level*2;
        if (abs(level) == s_y)
          increment = 1;
        for(int s_x = -level; s_x <= level; s_x+=increment) {
             int b_x = p_x+s_x;
            if (b_x < 0 || b_y < 0 || b_x >= image_width || b_y >= image_height){
                done = true;
                continue;
             }
//float distance = sqrtf(s_x*s_x + s_y*s_y);
//if (distance > level)
  //              continue;
        float num = 0;
        float4& color = rgb_in[b_x + b_y*image_width];
        num += color.w;

                //if (s_x == 0 && s_y == 0 && num == 0)
                //  done = true;
             total_count += num;
        }
      }
        if (total_count >= N || level >= max_level) {
          done = true;
        }
     //   N += (*max_spp_real)*2.0;
    }
        radius += 1;  //DEBUG
    //sum up each weighted values
    float weight_scale = 1.0f/(k*radius);
    //float denom = 1.0f/((1.0f-2.0f*k/3.0f)*2.0*radius*radius); //CD square version
    float denom = 1.0f/((1.0f-2.0f*k/3.0f)*M_PI*radius*radius);
    float total = 0.0; //total radiance value
    float4 total_color = make_float4(0.0,0.0,0.0,0.0);
    //total_count = 0;
    done = false;
    int level = radius;
    //for(int level = 0;level < radius;level++) {
      float level_total = 0.0;
      for(int s_y = -level; s_y <= level; s_y++) {
        int b_y = p_y + s_y;
        //int increment = level*2;
        //if (abs(level) == s_y)
        //  increment = 1;
        for(int s_x = -level; s_x <= level; s_x++) {
            int b_x = p_x+s_x;
            if (b_x < 0 || b_y < 0 || b_x >= image_width || b_y >= image_height){
                done = true;
                continue;
             }
             float distance = sqrtf(s_x*s_x + s_y*s_y);
             if (distance > radius)
                continue;
             float weight = 1.0- distance*weight_scale;
        float4& color = rgb_in[b_x + b_y*image_width];
        total_color += color*weight;
        }
      }

  total_color *= denom;

  float scale =  1.0f/float(float(max_spp))*255.0f/1.8f;
  float r = total_color.x*scale; //((float(pixel_buffer[x].x))*scale);
  float g = total_color.y*scale; //((float(pixel_buffer[x].y))*scale);
  float b = total_color.z*scale; //((float(pixel_buffer[x].z))*scale);
if (r > 255.0f)
    r = 255.0f;
  if (g > 255.0f)
    g = 255.0f;
  if (b > 255.0f)
    b = 255.0f;
    if (r < 0.0f)
    r = 0;
  if (g < 0.0f)
    g = 0;
  if (b < 0.0f)
    b = 0;
rgb_out[index] = rgbToInt(r,g,b);

//printf("normalizing\n");
 // }
}


#endif // KERNEL_FILTER_H
