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



#include "schlierenfilter.h"
#include <cassert>
//#include <cutil.h>
//#include <cutil_inline.h>
//#include <cutil_math.h>
//#include <helper_cuda.h>
//#include <helper_math.h>
#include "cudaIncludes.h"

void SchlierenShadowgraphCutoff::Setup(RenderParameters& params)
{
    params.cutoff = CUTOFF_NONE;
}

void SchlierenPositiveHorizontalKnifeEdgeCutoff::Setup(RenderParameters& params)
{
    params.cutoff = CUTOFF_KNIFE_X;
}


SchlierenImageCutoff::SchlierenImageCutoff(float* data, int width, int height)
    : _width(width), _height(height)
{
    _data = data;
}

void SchlierenImageCutoff::setData(float* data)
{
//assert(data);
_data = data;
}

void SchlierenImageCutoff::Setup(RenderParameters& params)

{
    params.cutoff_rgb = (float4*)_data;
    params.cutoff = CUTOFF_IMAGE;
    params.cutoffSize = make_uint2(_width,_height);
    params.cutoff_dirty = true;
}


void SchlierenInterforemetryCutoff::Setup(RenderParameters& params)
{
    params.cutoff = CUTOFF_INTERFOREMETRY;
}

void SchlierenTraceCutoff ::Setup(RenderParameters& params)
{
    params.cutoff = CUTOFF_TRACE;
}
