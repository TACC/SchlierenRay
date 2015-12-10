# -------------------------------------------------
# Project created by QtCreator 2010-04-13T12:31:52
# -------------------------------------------------
TEMPLATE += app
QT += opengl \
    xml \
    svg \
    core \
    gui \
    declarative
TARGET = SchlierenVis
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

#lappy
INCLUDEPATH += /Users/carson/git/SchlierenRay/Schlieren
INCLUDEPATH += /Developer/NVIDIA/CUDA-7.0/samples/common/inc/
INCLUDEPATH += /usr/local/cuda/include/
INCLUDEPATH += /Users/carson/opt/teem-1.11.0-src/include
LIBPATH += /Users/carson/git/SchlierenRay/Schlieren/build
LIBPATH += /Users/carson/opt/teem-1.11.0-src/bin

#Maverick
#INCLUDEPATH += /work/01336/carson/git/SchlierenRay/Schlieren
#INCLUDEPATH += /opt/apps/cuda/7.0/include
#INCLUDEPATH += /opt/apps/cuda/7.0/samples/common/inc/
#INCLUDEPATH += /work/01336/carson/opt/include
#LIBPATH += /work/01336/carson/git/SchlierenRay/Schlieren/build
#LIBPATH += /work/01336/carson/opt/lib

#INCLUDEPATH += /home/01652/alim/w/schlieren/SchlierenRay/Schlieren
#INCLUDEPATH += /opt/apps/cuda/7.0/include
#INCLUDEPATH += /opt/apps/cuda/7.0/samples/common/inc/
#INCLUDEPATH += /home/01652/alim/w/schlieren/local/include

#LIBS += -LD:/home/carson/svn/Schlieren/build -lSchlieren
#unix:LIBS += -lglut

#LIBPATH += /home/01652/alim/w/schlieren/SchlierenRay/Schlieren/build
#LIBPATH += /home/01652/alim/w/schlieren/local/bin
#LIBPATH += /home/01652/alim/w/schlieren/local/lib
LIBS +=  -lSchlieren -lteem
