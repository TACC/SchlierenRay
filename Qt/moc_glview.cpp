/****************************************************************************
** Meta object code from reading C++ file 'glview.h'
**
** Created: Wed Dec 2 16:33:05 2015
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "glview.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'glview.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GLView[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      10,    8,    7,    7, 0x05,

 // slots: signature, parameters, type, tag, flags
      41,    7,    7,    7, 0x0a,
      65,   48,    7,    7, 0x0a,
     105,   96,    7,    7, 0x0a,
     133,  127,    7,    7, 0x0a,
     155,  153,    7,    7, 0x0a,
     186,  184,    7,    7, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_GLView[] = {
    "GLView\0\0,\0drawFilterAtPoint(float,float)\0"
    "draw()\0img,width,height\0"
    "setImageCutoff(float*,int,int)\0filename\0"
    "loadData(std::string)\0scale\0"
    "setDataScale(float)\0d\0"
    "setProjectionDistance(float)\0c\0"
    "setCutoffScale(float)\0"
};

void GLView::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GLView *_t = static_cast<GLView *>(_o);
        switch (_id) {
        case 0: _t->drawFilterAtPoint((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 1: _t->draw(); break;
        case 2: _t->setImageCutoff((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 3: _t->loadData((*reinterpret_cast< std::string(*)>(_a[1]))); break;
        case 4: _t->setDataScale((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 5: _t->setProjectionDistance((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 6: _t->setCutoffScale((*reinterpret_cast< float(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GLView::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GLView::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_GLView,
      qt_meta_data_GLView, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GLView::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GLView::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
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
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
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
