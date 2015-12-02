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



#ifndef SCHLIERENRENDERER_H
#define SCHLIERENRENDERER_H

#include "RenderParameters.h"
#include "schlierenfilter.h"
#include "schlierenimagefilter.h"

class SchlierenRenderer
{
public:
    SchlierenRenderer();
    ~SchlierenRenderer();
    void init();
    void setRenderSize(int x, int y) { _params.width = x; _params.height = y; }
    int getWidth() { return _params.width; }
    int getHeight() { return _params.height; }
    void setWidth(int width) { _params.width=width; }
    void setHeight(int height) { _params.height=height; }
    void render();
    void copyInoutBuffer();
    void setFilter(SchlierenCutoff* filter);
    void setThreadSafe(bool st) { _params.threadSafe = st; }
    void setImageFilter(ImageFilter* imageFilter) { imageFilter->Setup(_params); }
    void setProjectionDistance(float d) { _params.projectionDistance = d; }
    float getProjectionDistance() { return _params.projectionDistance; }
    void useOctree(bool st) { _params.useOctree = st; }
    void setOctreeThreshold(float t) {  }
    void rotate(float x, float y);
    void updateCamera();
    void setCameraIndex(int idx);
    void clear();
    void setRaysPerPixel(int rpp) { _params.raysPerPixel = rpp; }
    int getRaysPerPixel() { return _params.raysPerPixel; }
    void setNumRenderPasses(int p) { _params.numRenderPasses = p; }
    int getNumRenderPasses() { return _params.numRenderPasses; }
    void setStepSize(float s) { _params.stepSize = s; }
    void useRefraction(bool st) { _params.useRefraction = st; }
    void setData(float* data, int width, int height, int depth);
    void setDataScale(float data_scalar) { _params.dataScalar = data_scalar; }
    float getDataScale() { return _params.dataScalar; }
    void setCutoffScale(float cutoff_scalar) { _params.cutoffScalar = cutoff_scalar; }

    //protected:
    void displayImage();
        RenderParameters _params;
        bool _initialized;
        float _rot_x,_rot_y;
        float3 _center;
};

#endif // SCHLIERENRENDERER_H
