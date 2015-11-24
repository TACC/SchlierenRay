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



#include "schlierenrenderer.h"

#include "opengl_include.h"
//#include "cutil.h"
//#include "cutil_math.h"
#include "cudaIncludes.h"
#include <cuda.h>
#include "cuda_gl_interop.h"

#include <float.h>
#include <cmath>

extern "C" void Host_Render( RenderParameters* paramsp);
extern "C" void Host_Init(RenderParameters* paramsp);
extern "C" void Host_Clear(RenderParameters* paramsp);
extern "C" void Host_Kill();

// pbo and fbo variables
GLuint pbo_out, pbo_dest;
GLuint fbo_source;
GLuint tex_source;

// (offscreen) render target
// fbo variables
GLuint framebuffer;
GLuint tex_screen;
GLuint depth_buffer;

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
  void
createPBO(GLuint* pbo, int width, int height)
{
  //    // set up vertex data parameter
  //    int num_texels = width * height;
  //    int num_values = num_texels * 4;
  //    int size_tex_data = sizeof(GLubyte) * num_values;
  //    void *data = malloc(size_tex_data);
  //
  //    // create buffer object
  //    glGenBuffers(1, pbo);
  //    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
  //    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
  //    free(data);
  //
  //    glBindBuffer(GL_ARRAY_BUFFER, 0);
  //
  //    // register this buffer object with CUDA
  //    cudaGLRegisterBufferObject(*pbo);
  //
  //    //CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete PBO
////////////////////////////////////////////////////////////////////////////////
  void
deletePBO(GLuint* pbo)
{
  //    // unregister this buffer object with CUDA
  //    cudaGLUnregisterBufferObject(*pbo);
  //
  //    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
  //    glDeleteBuffers(1, pbo);
  //
  //    *pbo = 0;
}

// display image to the screen as textured quad
void SchlierenRenderer::displayImage()
{
  // render a screen sized quad
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glMatrixMode( GL_MODELVIEW);
  glLoadIdentity();

  glViewport(0, 0, _params.width, _params.height);

  //#if USE_FLOAT_SHADER
  //    // fragment program is required to display floating point texture
  //    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
  //    glEnable(GL_FRAGMENT_PROGRAM_ARB);
  //    glDisable(GL_DEPTH_TEST);
  //#endif

  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
  glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
  glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
  glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glDisable(GL_TEXTURE_2D);

}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
createFramebuffer(GLuint* fbo, GLuint color, GLuint depth)
{
  //    // create and bind a framebuffer
  //    glGenFramebuffersEXT(1, fbo);
  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *fbo);
  //
  //    // attach images
  //    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, color, 0);
  //    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth);
  //
  //    // clean up
  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y)
{
  // create a texture
  glGenTextures(1, tex_name);
  glBindTexture(GL_TEXTURE_2D, *tex_name);

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // buffer data
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);


}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
deleteTexture(GLuint* tex)
{
  glDeleteTextures(1, tex);


  *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
deleteFramebuffer( GLuint* fbo)
{
  //    glDeleteFramebuffersEXT(1, fbo);
  //
  //    *fbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
createDepthBuffer(GLuint* depth, unsigned int size_x, unsigned int size_y)
{
  //    // create a renderbuffer
  //    glGenRenderbuffersEXT(1, depth);
  //    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *depth);
  //
  //    // allocate storage
  //    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, size_x, size_y);
  //
  //    // clean up
  //    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
deleteDepthBuffer(GLuint* depth)
{
  //    glDeleteRenderbuffersEXT(1, depth);
  //
  //    *depth = 0;
}

  SchlierenRenderer::SchlierenRenderer()
: _initialized(false), _rot_x(0), _rot_y(0)
{
  _params.passes = 0;
  //_params.tex_data = NULL;
  _params.width = 512;
  _params.height = 512;
  _params.inout_rgb = NULL;
  _params.out_rgb = NULL;
  //        _params.camera_pos = make_float3(0,0,-5);
  //        _params.camera_x = make_float3(1,0,0);
  //        _params.camera_y = make_float3(0,1,0);
  //        _params.camera_z = make_float3(0,0,1);
  updateCamera();
  _params.raysPerPixel = 1;
  _params.numRenderPasses = 1;
  _params.cutoff = CUTOFF_NONE;
  _params.cutoff_dirty = true;
  _params.data = NULL;
  _params.projectionDistance = 0.4;
  _params.stepSize = 0.1;
  _params.threadSafe = false;
  _params.dataScalar = 1.0f;
  _params.cutoffScalar = 5000.0f;
  _params.useOctree = false;
  _params.useRefraction = false;
  _params.inout_rgb = new float4[_params.width*_params.height];
  _params.min_bound = make_float3(-.5,-.5,-.5);
  _params.max_bound = make_float3(.5,.5,.5);
  float3 camera_pos, camera_x, camera_y, camera_z, center = _params.max_bound/2.0;
  float rot_x = 0, rot_y = 0;
  //        camera_pos.z = center.z + cos(rot_x)*5.0;
  //        camera_pos.x = center.x + sin(rot_x)*5.0;
  //        camera_z = normalize(center-camera_pos);
  //        camera_z = make_float3(0,0,1);

  //        camera_y = make_float3(0,1,0);
  //        camera_x = normalize(cross(camera_y, camera_z*-1.0f));
  //        camera_y = normalize(cross(camera_x, camera_z));

  //        _params.camera_corner = _params.camera_pos-(_params.camera_x*.5+_params.camera_y*.5);

  //        _params.camera_x = camera_x;
  //        _params.camera_y = camera_y;
  //        _params.camera_z = camera_z;

  _center = make_float3(0,0,0);
}

SchlierenRenderer::~SchlierenRenderer()
{
  //    cudaGLUnregisterBufferObject(pbo_dest);
  //
  //    deletePBO(&pbo_dest);
  //
  //    deleteTexture(&tex_screen);
  //    deleteFramebuffer(&framebuffer);
  //    deleteFramebuffer(&depth_buffer);
  Host_Kill();
}

void SchlierenRenderer::init()
{
  _initialized=true;
  // create pbo
  //    createPBO(&pbo_dest,_params.width, _params.height);
  //
  //    // create texture for blitting onto the screen
  //    createTexture(&tex_screen, _params.width, _params.height);
  //
  //    // create a depth buffer for offscreen rendering
  //    createDepthBuffer(&depth_buffer, _params.width, _params.height);
  //
  //    // create a framebuffer for offscreen rendering
  //    createFramebuffer(&framebuffer, tex_screen, depth_buffer);

  Host_Init(&_params);

}

void SchlierenRenderer::render()
{
  if (!_initialized)
    init();
  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);
  //    cudaGLMapBufferObject( (void**)&_params.out_data, pbo_out);

  Host_Render(&_params);

  //    cudaGLUnmapBufferObject( pbo_out);
  //
  //    // download texture from destination PBO
  //    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);
  //    glBindTexture(GL_TEXTURE_2D, tex_screen);
  //    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
  //                    _params.width, _params.height,
  //                    GL_BGRA, GL_UNSIGNED_BYTE, NULL);
  //
  //    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
  //    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  //
  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  //    displayImage();
}

void SchlierenRenderer::rotate(float x, float y)
{
  _rot_x += x;
  _rot_y += y;
  updateCamera();
}

void SchlierenRenderer::updateCamera()
{

  float3 center = _params.max_bound/2.0f;

  _params.camera_pos.z = center.x + cos(_rot_y)*cos(_rot_x)*5.0;
  _params.camera_pos.y = center.y + sin(_rot_y)*5.0;

  _params.camera_pos.x = center.z + sin(_rot_x)*5.0;

  _params.camera_z = normalize(center-_params.camera_pos);

  _params.camera_y = make_float3(0,1,0);
  _params.camera_x = normalize(cross(_params.camera_y, _params.camera_z*-1.0f));
  _params.camera_y = normalize(cross(_params.camera_x, _params.camera_z));


  _params.camera_pos.z = center.z + cos(_rot_x)*5.0;
  _params.camera_pos.x = center.x + sin(_rot_x)*5.0;
  _params.camera_z = normalize(center-_params.camera_pos);

  _params.camera_y = make_float3(0,1,0);
  _params.camera_x = normalize(cross(_params.camera_y, _params.camera_z*-1.0f));
  _params.camera_y = normalize(cross(_params.camera_x, _params.camera_z));



  _params.camera_pos.z = _center.z + cos(_rot_y)*5.0;
  _params.camera_pos.y = _center.y + sin(_rot_y)*5.0;
  _params.camera_z = normalize(_center-_params.camera_pos);

  _params.camera_y = make_float3(0,1,0);
  _params.camera_x = normalize(cross(_params.camera_y, _params.camera_z*-1.0f));
  //    camera_x = set_float3(1,0,0);
  _params.camera_y = normalize(cross(_params.camera_x, _params.camera_z));


  // alim: fixed rotation logic

  static int counter = 0;
  if(counter > 72) counter = 0;

  const double angle_step = M_PI / 72.0;
  const double start_angle = angle_step * 0; // 73.0;
  double curr_angle = angle_step * counter + start_angle;
  float y = cos(curr_angle) * 5.0;
  float z = -sin(curr_angle) * 5.0;

  _params.camera_pos = normalize(make_float3(0, y, z));
  _params.camera_pos = make_float3(0, y, z);


  if(counter == 0) {
    _params.camera_x = make_float3(-1, 0, 0);
    _params.camera_y = make_float3(0, 0, 1);
    _params.camera_z = make_float3(0, -1, 0);

  } else {
    _params.camera_z = normalize(_center-_params.camera_pos);
    _params.camera_y = make_float3(0,1,0);
    _params.camera_x = normalize(cross(_params.camera_y, _params.camera_z*-1.0f));
    _params.camera_y = normalize(cross(_params.camera_x, _params.camera_z));
  }

  std::cout << "center: [" << counter << "] pos: [" << center.x << ", " << center.y << ", " << center.z << "]" << std::endl;
  std::cout << "cam: [" << counter << "] pos: [" << _params.camera_pos.x << ", " << _params.camera_pos.y << ", " << _params.camera_pos.z << "]" << std::endl;
  std::cout << "cam-x: [" << _params.camera_x.x << ", " << _params.camera_x.y << ", " << _params.camera_x.z << "]" << std::endl;
  std::cout << "cam-y: [" << _params.camera_y.x << ", " << _params.camera_y.y << ", " << _params.camera_y.z << "]" << std::endl;
  std::cout << "cam-z: [" << _params.camera_z.x << ", " << _params.camera_z.y << ", " << _params.camera_z.z << "]" << std::endl;

  counter++;



  _params.camera_corner = _params.camera_pos-(_params.camera_x*.5+_params.camera_y*.5);
}

void SchlierenRenderer::setData(float* data, int data_width, int data_height, int data_depth)
{
  _params.data_min = FLT_MAX;
  _params.data2 = data;
  //compute gradient
  _params.data = new float4[data_width*data_height*data_depth];
  for(size_t z = 0; z < data_depth; z++) {
    for(size_t y = 0; y < data_height; y++) {
      for(size_t x = 0; x < data_width; x++) {
        size_t DELTA = 1;
        float3 lookup = {x,y,z};
        if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA ||
            lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA)
          continue;
        float3 sample1, sample2;
        lookup = make_float3(x-1,y,z);
        sample1.x = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        lookup = make_float3(x+1,y,z);
        sample2.x = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];

        lookup = make_float3(x,y-1,z);
        sample1.y = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        lookup = make_float3(x,y+1,z);
        sample2.y = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];

        lookup = make_float3(x,y,z-1);
        sample1.z = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        lookup = make_float3(x,y,z+1);
        sample2.z = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        float3 normal;
        normal = sample1 - sample2;
        float4& datap = _params.data[size_t(z*data_width*data_height + y*data_width + x)];
        datap.x = normal.x;
        datap.y = normal.y;
        datap.z = normal.z;
        datap.w = data[size_t(z*data_width*data_height + y*data_width + x)];
        if (datap.w < _params.data_min)
          _params.data_min = datap.w;
      }
    }
  }
  float m = max(float(data_width), float(data_height));
  m = max(m, float(data_depth));
  float3 dataScale = make_float3(float(data_width)/m, float(data_height)/m, float(data_depth)/m);
  _params.min_bound = -dataScale/2.0f;
  _params.max_bound = dataScale/2.0f;

  _params.data_width = data_width;
  _params.data_height = data_height;
  _params.data_depth = data_depth;
}

void SchlierenRenderer::setFilter(SchlierenCutoff* filter)
{
  filter->Setup(_params);
}

void SchlierenRenderer::clear()
{
  Host_Clear(&_params);
  _params.passes = 0;
}
