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



#include "kernel_functions.cu"

__global__ void kernel_normalize(pixel_bufferT* pixel_buffer, normalized_pixel_bufferT* normalized_pixel_buffer, float* max_spp_real, int num_pixels, int image_width, int image_height)
{
  float max_spp = (*max_spp_real); // increase brightness
//printf("normalization called\n");
  unsigned int r_x = blockIdx.x*blockDim.x + threadIdx.x;
  int num_pixels_per_thread = NUM_PER_THREAD*10; //doing one per thread will die
  for (int x = r_x*num_pixels_per_thread; x < (r_x+1)*num_pixels_per_thread; x++) {
    if (x >= num_pixels)
       return;

    int p_x = x%image_width;
    int p_y = x/image_width;
#if GUASSIAN_BLUR
#if !COLOR
    //GUASSIAN SMOOTHING KERNEL
    float deviation = 1.0;
    int kernel_size = 5;
    float total = 0.0f;
    int counter = 0;
    int level = 0;
    while (counter < (max_spp)*(level+1) && level < 21) {
        counter = 0;
        for(int k_x = -level; k_x <= level; k_x++) {
        for(int k_y = -level; k_y <= level; k_y++) {
          int b_x = p_x + k_x;
          int b_y = p_y + k_y;
          counter += pixel_buffer[(b_x) + (b_y)*image_width];//*(1.0f/(2*M_PI*deviation*deviation))*expf(-(k_x*k_x+k_y*k_y)/(2.0*deviation*deviation));
        }
        }
        level ++;
        deviation = level;
    }
    kernel_size = 1+(level-1)*2.0;
    deviation *= 0.8;
    if (kernel_size < 3) {
        total = pixel_buffer[(p_x) + (p_y)*image_width];
    } else {
    //Guassian Kernel  G(x,y) = (1.0f/(sqrtf(2*pi)*deviation))*expf(-(x*x+y*y)/(2.0*deviation*deviation));
    for(int k_x = -kernel_size/2; k_x <= kernel_size/2; k_x++) {
        for(int k_y = -kernel_size/2; k_y <= kernel_size/2; k_y++) {
          int b_x = p_x + k_x;
          int b_y = p_y + k_y;
          if (b_x < 0 || b_x >= image_width)
            continue;
          if (b_y < 0 || b_y >= image_height)
            continue;
          total += pixel_buffer[(b_x) + (b_y)*image_width]*(1.0f/(2*M_PI*deviation*deviation))*expf(-(k_x*k_x+k_y*k_y)/(2.0*deviation*deviation));
        }
    }
    }
  float val = ((float(total)/float(float(max_spp)))*255.0f)*2.0f;
  if (val > 255.0f)
    val = 255.0f;
  normalized_pixel_buffer[x] = (normalized_pixel_bufferT)val;
#endif
#elif PHOTON_MAPPING
   //FOR NOW FILTERING ASSUMES NO COLOR
   //figure out radius of filter
    float radius = 0.0f;
    int max_level = 6;
    float k = 1.15; //weighting scale >= 1.0
    float nscale = 14 - *max_spp_real;
    if (nscale < 1.0)
      nscale = 0.5;
    float N = (*max_spp_real)*nscale; //number of samples to look for when filtering
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
#if COLOR
        float num = 0;
        float4& color = pixel_buffer[b_x + b_y*image_width];
        num += color.w;
#else
        float num = pixel_buffer[b_x + b_y*image_width];
#endif
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
#if COLOR
//DEBUG
//    radius = 3-max_spp/10.0;
//    if (radius < 1)
//      radius = 1;
        radius = 2;
#endif
        radius += 1;  //DEBUG
    //sum up each weighted values
    float weight_scale = 1.0f/(k*radius);
    float denom = 1.0f/((1.0f-2.0f*k/3.0f)*M_PI*radius*radius);
    float total = 0.0; //total radiance value
    float4 total_color = set4(0,0,0,0);
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
#if COLOR
        float4& color = pixel_buffer[b_x + b_y*image_width];
        total_color += color*weight;
#else
             float num = pixel_buffer[b_x + b_y*image_width];
             total += num*weight;
#endif

        }
      }

  total *= denom;

#if COLOR
  float scale =  1.0f/float(float(max_spp))*255.0f;
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
  normalized_pixel_buffer[x].r = r;
  normalized_pixel_buffer[x].g = g;
  normalized_pixel_buffer[x].b = b;
#else
  float val = ((float(total)/float(float(max_spp)))*255.0f)/2.0f;
  if (val > 255.0f)
    val = 255.0f;
  normalized_pixel_buffer[x] = (normalized_pixel_bufferT)val;
#endif

#elif AVG_FILTER

    float4 total_color;
    float total = 0;
    int level = 2;
    int area = 0;
    for(int s_y = -level; s_y <= level; s_y++) {
      int b_y = p_y + s_y;
      for(int s_x = -level; s_x <= level; s_x++) {
          int b_x = p_x+s_x;
          if (b_x < 0 || b_y < 0 || b_x >= image_width || b_y >= image_height){
            continue;
          }
#if COLOR
        float num = 0;
        float4& color = pixel_buffer[b_x + b_y*image_width];
        total_color += color;
#else
         float num = pixel_buffer[b_x + b_y*image_width];
         total += num;
#endif
         area++;

      }
    }

  total /= float(area);

#if COLOR
  float scale =  1.0f/float(float(max_spp))*255.0f;;
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
  normalized_pixel_buffer[x].r = (char)r;
  normalized_pixel_buffer[x].g = (char)g;
  normalized_pixel_buffer[x].b = (char)b;
#else
  float val = ((float(total)/float(float(max_spp)))*255.0f);
  if (val > 255.0f)
    val = 255.0f;
  normalized_pixel_buffer[x] = (normalized_pixel_bufferT)val;
#endif

#else  // no filtering

#if COLOR
  float scale = 1.0f/float(float(max_spp))*255.0f;
  float r = ((float(pixel_buffer[x].x))*scale);
  float g = ((float(pixel_buffer[x].y))*scale);
  float b = ((float(pixel_buffer[x].z))*scale);
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
  normalized_pixel_buffer[x].r = (char)r;
  normalized_pixel_buffer[x].g = (char)g;
  normalized_pixel_buffer[x].b = (char)b;
#else
  float val = ((float(pixel_buffer[x])/(float(max_spp)))*255.0f);
  if (val > 255.0f)
    val = 255.0f;
  normalized_pixel_buffer[x] = (normalized_pixel_bufferT)val;
#endif
#endif // GUASSIAN_BLUR
//printf("normalizing\n");
  }
}
