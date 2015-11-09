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



#ifndef SCHLIERENFILTER_H
#define SCHLIERENFILTER_H

#include "RenderParameters.h"

class SchlierenCutoff
{
public:
    SchlierenCutoff(){}
    virtual void Setup(RenderParameters& params) = 0;
};

class SchlierenBOSCutoff : public SchlierenCutoff
{
public:
    SchlierenBOSCutoff() {}
    virtual void Setup(RenderParameters& params) { params.cutoff = CUTOFF_BOS; }
};

class SchlierenShadowgraphCutoff : public SchlierenCutoff
{
public:
    SchlierenShadowgraphCutoff() {}
    virtual void Setup(RenderParameters& params);

};

class SchlierenPositiveHorizontalKnifeEdgeCutoff : public SchlierenCutoff
{
    public:
    SchlierenPositiveHorizontalKnifeEdgeCutoff() {}
            virtual void Setup(RenderParameters& params);
};

class SchlierenImageCutoff : public SchlierenCutoff
{
public:
    //data must be rgba_float
    SchlierenImageCutoff(float* data, int width = 256, int height = 256);
    void setData(float* data);
    virtual void Setup(RenderParameters& params);

protected:
    int _width, _height;
    float* _data;
};

class SchlierenInterforemetryCutoff : public SchlierenCutoff
{
    public:
    SchlierenInterforemetryCutoff() {}
            virtual void Setup(RenderParameters& params);
};

class SchlierenTraceCutoff : public SchlierenCutoff
{
    public:
    SchlierenTraceCutoff() {}
    virtual void Setup(RenderParameters& params);
};

#endif // SCHLIERENFILTER_H
