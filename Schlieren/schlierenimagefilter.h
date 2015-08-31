#ifndef SCHLIERENIMAGEFILTER_H
#define SCHLIERENIMAGEFILTER_H

#include "RenderParameters.h"

class ImageFilter
{
public:
    ImageFilter() {}
    virtual void Setup(RenderParameters& params);
};

#endif // SCHLIERENIMAGEFILTER_H
