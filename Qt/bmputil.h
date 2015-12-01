#ifndef _bmputil_h
#define _bmputil_h
#include <string>
#include <cstdlib>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>

// bmp utils
// assumes 32bit colors are being used
// the struct formats are basically copied from windows documentation
namespace bmp {

struct __attribute__((__packed__)) bmpfileheader {
    uint16_t type; // fixed to 'BM'
    uint32_t size; // fileheader size + bmpheader size + data size
    uint16_t res0; // ignore
    uint16_t res1; // ignore
    uint32_t offset; // fixed to sizeof(bmpfileheader) + sizeof(bmpinfoheader)

    bmpfileheader() :
        type(0x4d42), size(0), res0(0), res1(0), offset(0)
    {}
};

struct __attribute__((__packed__)) bmpinfoheader {
    uint32_t size; // size of infoheader
    int32_t width; // image width
    int32_t height; // image height
    uint16_t planes; // 1 plane
    uint16_t bitcount; // 32bit (rgba)
    uint32_t compression; // ignore - 0
    uint32_t sizeimage; // width * height * 4 bytes (rgba)
    int32_t hres; // ignore - just putting 1000
    int32_t vres; // ignore - just putting 1000
    uint32_t colors; // ignore - 0
    uint32_t colorsimportant; // ignore - 0

    bmpinfoheader() :
        size(sizeof(bmpinfoheader)), width(0), height(0), planes(1), bitcount(32),
        compression(0), sizeimage(0), hres(1000), vres(1000),
        colors(0), colorsimportant(0)
    {}
};

struct __attribute__((__packed__)) bmpheader {
    struct bmpfileheader fh;
    struct bmpinfoheader ih;

    // modify both header parts to reflect the size of the image
    void set_size(const int32_t width, const int32_t height) {
        const uint32_t fhsize = sizeof(bmpfileheader);
        const uint32_t ihsize = sizeof(bmpinfoheader);
        const uint32_t datasize = width * height * 4;

        fh.size = fhsize + ihsize + datasize;
        fh.offset = fhsize + ihsize;
        ih.width = width;
        ih.height = height;
        ih.sizeimage = datasize;
    }
};

// Does an in-memory replacement of r and b channels
// Uses const_cast and reinterpret_cast to open the pointer for writing
void rgba_to_bgra(const void* rgba, const int32_t width, const int32_t height) {
    void *bgra_v = const_cast<void*>(rgba);
    char *bgra = reinterpret_cast<char*>(bgra_v);

    const int32_t size = width * height * 4;
    for(int32_t i=0; i<size; i+=4) {
        const char t = bgra[i];
        bgra[i] = bgra[i+2];
        bgra[i+2] = t;
    }
}

} // end namespace bmp


#endif // _bmputil_h
