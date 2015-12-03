/****************************************************************************
** Meta object code from reading C++ file 'painterwidget.h'
**
** Created: Wed Dec 2 16:33:07 2015
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "painterwidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'painterwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_PainterWidget[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      32,   15,   14,   14, 0x05,

 // slots: signature, parameters, type, tag, flags
      63,   61,   14,   14, 0x0a,
      87,   81,   14,   14, 0x0a,
     113,  109,   14,   14, 0x0a,
     138,  136,   14,   14, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_PainterWidget[] = {
    "PainterWidget\0\0img,width,height\0"
    "imageChanged(float*,int,int)\0s\0"
    "setBrushSize(int)\0color\0setBrushColor(QColor)\0"
    "x,y\0drawBrush(float,float)\0p\0"
    "drawBrushAbsolute(QPoint)\0"
};

void PainterWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        PainterWidget *_t = static_cast<PainterWidget *>(_o);
        switch (_id) {
        case 0: _t->imageChanged((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 1: _t->setBrushSize((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->setBrushColor((*reinterpret_cast< QColor(*)>(_a[1]))); break;
        case 3: _t->drawBrush((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 4: _t->drawBrushAbsolute((*reinterpret_cast< QPoint(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData PainterWidget::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject PainterWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_PainterWidget,
      qt_meta_data_PainterWidget, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &PainterWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *PainterWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *PainterWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_PainterWidget))
        return static_cast<void*>(const_cast< PainterWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int PainterWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void PainterWidget::imageChanged(float * _t1, int _t2, int _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
