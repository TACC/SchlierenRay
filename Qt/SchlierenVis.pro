# -------------------------------------------------
# Project created by QtCreator 2010-04-13T12:31:52
# -------------------------------------------------
QT += opengl \
    xml \
 #   multimedia \
    svg
TARGET = SchlierenVis
TEMPLATE = app
SOURCES += main.cpp \
    mainwindow.cpp \
    glview.cpp \
    filter.cpp \
    painterwidget.cpp
HEADERS += mainwindow.h \
    glview.h \
    filter.h \
    painterwidget.h
FORMS += mainwindow.ui \
    filter.ui
INCLUDEPATH += /Users/carson/svn/Schlieren
INCLUDEPATH += /Developer/NVIDIA/CUDA-7.0/samples/common/inc/
INCLUDEPATH += /usr/local/cuda/include/
INCLUDEPATH += /Users/carson/opt/teem-1.11.0-src/include
#LIBS += -LD:/home/carson/svn/Schlieren/build -lSchlieren
#unix:LIBS += -lglut
LIBPATH += /Users/carson/svn/Schlieren/
LIBPATH += /Users/carson/opt/teem-1.11.0-src/bin
LIBS += -L/Users/carson/svn/Schlieren/ -lSchlieren -lteem
