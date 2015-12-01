#include <QApplication>
#include <QtOpenGL>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    if (!QGLFormat::hasOpenGL() || !QGLFramebufferObject::hasOpenGLFramebufferObjects()) {
        QMessageBox::information(0, "OpenGL framebuffer objects",
                                 "This system does not support OpenGL/framebuffer objects.");
        return -1;
    }

    MainWindow w;
    w.resize(700,700);
    w.show();
    return a.exec();
}
