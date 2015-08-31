# -------------------------------------------------
# Project created by QtCreator 2010-04-19T22:10:23
# -------------------------------------------------
QT -= core \
    gui
TARGET = Schlieren
TEMPLATE = lib
DEFINES += SCHLIEREN_LIBRARY
SOURCES += main.cpp \
    schlierenrenderer.cpp \
    schlierenimagefilter.cpp \
    schlierenfilter.cpp
HEADERS += opengl_include.h \
    main.h \
    kernel_volume.h \
    kernel_post.h \
    kernel_functions.cu \
    cutil_math.h \
    cutil.h \
    schlierenrenderer.h \
    schlierenimagefilter.h \
    schlierenfilter.h \
    RenderParameters.h \
    kernel_render.h \
    kernel_functions.h \
    kernel_cutoff.h \
    kernel_filter.h
OTHER_FILES += host.cu \
    host_render.cu \
    CMakeLists.txt
