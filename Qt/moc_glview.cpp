/****************************************************************************
** Meta object code from reading C++ file 'glview.h'
**
** Created: Mon Apr 26 10:28:25 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "glview.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'glview.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GLView[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // signals: signature, parameters, type, tag, flags
      10,    8,    7,    7, 0x05,

 // slots: signature, parameters, type, tag, flags
      41,    7,    7,    7, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_GLView[] = {
    "GLView\0\0,\0drawFilterAtPoint(float,float)\0"
    "draw()\0"
};

const QMetaObject GLView::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_GLView,
      qt_meta_data_GLView, 0 }
};

const QMetaObject *GLView::metaObject() const
{
    return &staticMetaObject;
}

void *GLView::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GLView))
        return static_cast<void*>(const_cast< GLView*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int GLView::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: drawFilterAtPoint((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 1: draw(); break;
        default: ;
        }
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void GLView::drawFilterAtPoint(float _t1, float _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
