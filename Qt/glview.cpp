#include <QDebug>

#include "glview.h"
//#if __APPLE__
//#include <GLUT/glut.h>
//#else
//#include <GL/glut.h>
//#endif
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <stringstream>
#include "bmputil.h"
using namespace std;


typedef struct {
  float* data;
  string filename;
  int sizex, sizey, sizez;
} DataFile;

void loadNRRD(DataFile* datafile, int data_min, int data_max)
{

  printf("loading file %s : ", datafile->filename.c_str());
  Nrrd* nrrd = nrrdNew();
  if(nrrdLoad(nrrd, datafile->filename.c_str(), 0)) {
    char* err=biffGetDone(NRRD);
    cerr << "Failed to open \"" + string(datafile->filename) + "\":  " + string(err) << endl;
    exit(__LINE__);
  }
  int sizex, sizey, sizez;
  sizex = nrrd->axis[0].size;
  sizey = nrrd->axis[1].size;
  sizez = nrrd->axis[2].size;
  printf(" size: %f %f %f ", float(sizex), float(sizey), float(sizez));
  if (sizez > (data_max-data_min))
    sizez = data_max-data_min;
  float* data = new float[sizex*sizey*sizez];
  float min = FLT_MAX;
  float max = -FLT_MAX;
  float* dataNrrd = (float*)nrrd->data;
  float* datai = data;
  for(int i = 0; i < sizex; i++) {
    for(int j = 0; j < sizey; j++) {
      for( int k = 0; k < sizez; k++) {
        *datai = (*dataNrrd)*0.1;

        if (*datai > max)
          max = *datai;
        if (*datai < min)
          min = *datai;
        datai++;
        dataNrrd++;

      }
    }
  }


  datafile->data = data;
  datafile->sizex = sizex;
  datafile->sizey = sizey;
  datafile->sizez = sizez;
  nrrdNuke(nrrd);
  printf("  ...done\n");
}

GLView::GLView(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers|QGL::AlphaChannel), parent), pass(0), trace_buffer(0), data_loaded(false)
{
    schlieren = new SchlierenRenderer();
    float* cutoffData = new float[256*256*4];
    float* datap = cutoffData;
    for(int i = 0; i < 256; i++)
    {
        for(int j = 0; j < 256; j++)
        {
              datap[0] = 0.0f;
              datap[1] = 0.0f;
              datap[2] = 0.0f;
              datap[3] = 0.0f;
              datap+=4;
        }
    }
#if USE_IMAGE_CUTOFF
    filter = new SchlierenImageCutoff(cutoffData);
#else
    filter = new SchlierenPositiveHorizontalKnifeEdgeCutoff();
    //filter = new SchlierenBOSCutoff();
#endif
    schlieren->setFilter(filter);
    schlieren->setImageFilter(new ImageFilter());
    schlieren->setStepSize(0.02);
   // loadData("/home/carson/data/coal.nrrd");
    schlieren->setRaysPerPixel(3);
    schlieren->setRenderSize(700,700);
    schlieren->setNumRenderPasses(1);

    makeCurrent();
    //fbo = new QGLFramebufferObject(1024, 1024);
    rot_x = rot_y = rot_z = 0.0f;
    scale = 1.0f;

    tile_list = glGenLists(1);
    glNewList(tile_list, GL_COMPILE);
    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);

        glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);

        glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);

        glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);

        glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);

        glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
    }
    glEnd();
    glEndList();

}

GLView::~GLView()
{
    glDeleteLists(tile_list, 1);
    //delete fbo;
    delete schlieren;
}

void GLView::paintGL()
{
    draw();
}

//void GLView::paintEvent(QPaintEvent *)
//{
//    draw();
//}

void GLView::draw()
{

    glViewport(0, 0, width(), height());
    glClearColor(.8,.8,.8,1);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (!data_loaded)
    return;
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    //glFrustum(-1, 1, -1, 1, 10, 100);
//    //glTranslatef(0.0f, 0.0f, -15.0f);
//    gluPerspective(45,width()/height(),0.1, 100.0);
//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();
//    glViewport(0, 0, width(), height());
//    //glEnable(GL_BLEND);
//    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//
//    /*glBindTexture(GL_TEXTURE_2D, fbo->texture());
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    glEnable(GL_TEXTURE_2D);
//    glEnable(GL_MULTISAMPLE);
//    glEnable(GL_CULL_FACE);*/
//
//    // draw background
//    glPushMatrix();
//    glScalef(1.7f, 1.7f, 1.7f);
//    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
//    glCallList(tile_list);
//    glPopMatrix();
//
//    glTranslatef(0,0,-10.0);
//
//    glRotatef(rot_x, 1.0f, 0.0f, 0.0f);
//    glRotatef(rot_y, 0.0f, 1.0f, 0.0f);
//    glRotatef(rot_z, 0.0f, 0.0f, 1.0f);
//    glScalef(scale, scale, scale);
//
//    //glDepthFunc(GL_LESS);
//    //glEnable(GL_DEPTH_TEST);
//    // draw the Qt icon
//
//    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
//    glutSolidTeapot(1.0);
    // qDebug() << "rendering schlieren image";

      schlieren->render();

//    static char buffer[1024*1024*3];
//    for(int i=0;i<1024*1024*3;i++)
//        buffer[i]=255;
//    glDrawPixels(width(),height(),GL_RGBA,GL_UNSIGNED_BYTE,buffer);

//      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glFinish();
//
//    // restore the GL state that QPainter expects
//    restoreGLState();
//
//    // draw the overlayed text using QPainter
//    p.setPen(QColor(197, 197, 197, 157));
//    p.setBrush(QColor(197, 197, 197, 127));
//    p.drawRect(QRect(0, height()-50, width(), height()));
//    p.setPen(Qt::black);
//    p.setBrush(Qt::NoBrush);
//    const QString str1(tr("FPS:  10."));
//    const QString str2(tr("Use the mouse wheel to zoom, press buttons and move mouse to rotate, double-click to flip."));
//    QFontMetrics fm(p.font());
//    p.drawText(width()/2 - fm.width(str1)/2, height() - 20 - fm.lineSpacing() , str1);
//    p.drawText(width()/2 - fm.width(str2)/2, height() - fm.lineSpacing(), str2);
   if (pass++ < 200)
       update();
}

void GLView::setImageCutoff(float* img, int width, int height)
{
    qDebug() << "setting image cutoff";

//  float* cutoffData = new float[256*256*4];
//    float* datap = cutoffData;
//    for(int i = 0; i < 256; i++)
//    {
//        for(int j = 0; j < 256; j++)
//        {
//              datap[0] = float(i)/256.0;
//              datap[1] = float(i)/256.0;
//              datap[2] = float(i)/256.0;
//              datap[3] = 1.0f;
//            datap+=4;
//        }
//    }
    filter = new SchlierenImageCutoff(img,256,256);
    schlieren->setFilter(filter);
    schlieren->clear();
    pass = 0;
    update();
}

void GLView::resizeGL(int width, int height)
{
    schlieren->setRenderSize(width, height);
    if (trace_buffer)
        delete trace_buffer;
    trace_buffer = new unsigned char[width*height*3];
    schlieren->clear();
    pass = 0;
    update();
}

void GLView::mousePressEvent(QMouseEvent *e)
{
    anchor = e->pos();
	if (e->buttons() & Qt::RightButton)
	{
            trace();
                 float x = trace_buffer[(anchor.x() + anchor.y()*width())*3];
                float y = trace_buffer[(anchor.x() + anchor.y()*width())*3+1];
                emit drawFilterAtPoint(x/255.0f,y/255.0f);
              //  qDebug() << float(anchor.x())/float(width()) << " " << float(anchor.y())/float(height());
	}
        else if (e->buttons() & Qt::LeftButton)
        {
              last_position = e->pos();
              //  qDebug() << float(anchor.x())/float(width()) << " " << float(anchor.y())/float(height());
        }
}

void GLView::mouseMoveEvent(QMouseEvent *e)
{
    QPoint diff = e->pos() - anchor;
    if (e->buttons() & Qt::LeftButton) {
        rot_x += diff.y()/5.0f;
        rot_y += diff.x()/5.0f;
    } else if (e->buttons() & Qt::RightButton) {
        rot_z += diff.x()/5.0f;
    }

    anchor = e->pos();

        if (e->buttons() & Qt::RightButton)
        {
            float x = trace_buffer[(anchor.x() + anchor.y()*width())*3];
                float y = trace_buffer[(anchor.x() + anchor.y()*width())*3+1];
                emit drawFilterAtPoint(x/255.0f,y/255.0f);
                //emit drawFilterAtPoint(float(anchor.x())/float(width()),float(anchor.y())/float(height()));
              //  qDebug() << float(anchor.x())/float(width()) << " " << float(anchor.y())/float(height());
        }
        else if (e->buttons() & Qt::LeftButton)
        {
            QPoint offset = e->pos() - last_position;
//            schlieren->rotate(float(offset.x()*.005f), float(offset.y())*.005f);

            schlieren->rotate(float(offset.y()*.05f), 0);

              last_position = e->pos();
              //  qDebug() << float(anchor.x())/float(width()) << " " << float(anchor.y())/float(height());
              schlieren->clear();
              pass = 0;
        }

    update();
}

void GLView::wheelEvent(QWheelEvent *e)
{
    e->delta() > 0 ? scale += scale*0.1f : scale -= scale*0.1f;
    draw();
}


void write_file(std::string fn, unsigned int *out_rgb, int width, int height) {
    bmp::bmpheader bmph;
    bmph.set_size(width, height);

    char* rgba = (char*)out_rgb;
    bmp::rgba_to_bgra(rgba, width, height);

    FILE *f = fopen(fn.c_str(), "w");
    fwrite((void*)&bmph, sizeof(bmph), 1, f);
    size_t size = width * height * 4;
    fwrite((void*)rgba, size, 1, f);
    fclose(f);
}


void GLView::mouseDoubleClickEvent(QMouseEvent *)
{
    RenderParameters &p = schlieren->_params;

    // set resolution

    for(int i=0; i<72; i++) {
        schlieren->setCameraIndex(i);

        float dataScalar = p.dataScalar;
        // render background, no data
        {
            p.dataScalar = 0.0;
            schlieren->clear();
            pass = 0;

            for(pass=0; pass<10; pass++) {
                schlieren->render();
            }

            stringstream ss;
            ss << i << "_base.bmp";
            write_file(ss.str(), p.out_rgb, p.width, p.height);
        }

        // render with data
        {
            p.dataScalar = dataScalar;
            schlieren->clear();
            pass = 0;

            for(pass=0; pass<10; pass++) {
                schlieren->render();
            }

            stringstream ss;
            ss << i << "_data.bmp";
            write_file(ss.str(), p.out_rgb, p.width, p.height);
        }
    }
}

void GLView::saveGLState()
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
}

void GLView::restoreGLState()
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
}

void GLView::trace()
{
    schlieren->clear();
  schlieren->setFilter(new SchlierenTraceCutoff());
  int rpp = schlieren->getRaysPerPixel();
  schlieren->setRaysPerPixel(1);
  schlieren->render();

  glReadPixels(0,0,width(),height(),GL_RGB, GL_UNSIGNED_BYTE, trace_buffer);

  schlieren->setFilter(filter);
  schlieren->setRaysPerPixel(rpp);
  schlieren->clear();
  update();
}

void GLView::loadData(string filename)
{
    data_loaded = false;
    DataFile dataFiles[1];
    dataFiles[0].filename = filename;
    loadNRRD(&dataFiles[0],0, 1000);
    schlieren->setData(dataFiles[0].data, dataFiles[0].sizex,dataFiles[0].sizey, dataFiles[0].sizez);
    data_loaded = true;
    pass = 0;
    update();
}

void GLView::setDataScale(float scale)
{
  schlieren->setDataScale(scale);
  schlieren->clear();
  pass = 0;
  update();
}

void GLView::setProjectionDistance(float d)
{
  schlieren->setProjectionDistance(d);
  schlieren->clear();
  pass = 0;
  update();
}


void GLView::setCutoffScale(float c)
{
  schlieren->setCutoffScale(c);
  schlieren->clear();
  pass = 0;
  update();
}
