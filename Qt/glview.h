#ifndef GLVIEW_H
#define GLVIEW_H

#include <QtOpenGL>
#include <QImage>
#include <QTimeLine>
#include <QSvgRenderer>
#include <string>
#include "schlierenrenderer.h"

#define USE_IMAGE_CUTOFF 0

class GLView : public QGLWidget
{
    Q_OBJECT
public:
    GLView(QWidget* parent);
    ~GLView();

    void saveGLState();
    void restoreGLState();

    void paintGL();

    void resizeGL(int width, int height);
    //void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseDoubleClickEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void wheelEvent(QWheelEvent *);
        void trace();
        

public slots:
    void draw();
    void setImageCutoff(float* img, int width, int height);
    void loadData(std::string filename);
    void setDataScale(float scale);
    void setProjectionDistance(float d);
    void setCutoffScale(float c);
	
 signals:
        void drawFilterAtPoint(float, float);

private:
        bool data_loaded;
        QPoint anchor, last_position;
        GLuint tile_list;
        float scale, rot_x,rot_y,rot_z;
        QGLFramebufferObject * fbo;
        SchlierenRenderer* schlieren;
        int pass;
        SchlierenCutoff* filter;
        unsigned char* trace_buffer;

};

#endif // GLVIEW_H
