/****************************************************************************
** Meta object code from reading C++ file 'filter.h'
**
** Created: Wed Dec 2 16:33:06 2015
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "filter.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'filter.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ColorFilterWidget[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      36,   19,   18,   18, 0x05,

 // slots: signature, parameters, type, tag, flags
      69,   65,   18,   18, 0x0a,
      94,   92,   18,   18, 0x0a,
     114,   19,   18,   18, 0x0a,
     149,  147,   18,   18, 0x0a,
     173,  147,   18,   18, 0x0a,
     197,  147,   18,   18, 0x0a,
     222,  220,   18,   18, 0x0a,
     241,  220,   18,   18, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_ColorFilterWidget[] = {
    "ColorFilterWidget\0\0img,width,height\0"
    "imageChanged(float*,int,int)\0x,y\0"
    "drawBrush(float,float)\0c\0updateColor(QColor)\0"
    "imageChangedSlot(float*,int,int)\0v\0"
    "onDataSliderChange(int)\0onProjSliderChange(int)\0"
    "onCutSliderChange(int)\0i\0openImage(QString)\0"
    "saveImage(QString)\0"
};

void ColorFilterWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ColorFilterWidget *_t = static_cast<ColorFilterWidget *>(_o);
        switch (_id) {
        case 0: _t->imageChanged((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 1: _t->drawBrush((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 2: _t->updateColor((*reinterpret_cast< const QColor(*)>(_a[1]))); break;
        case 3: _t->imageChangedSlot((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 4: _t->onDataSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->onProjSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->onCutSliderChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->openImage((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 8: _t->saveImage((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData ColorFilterWidget::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ColorFilterWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_ColorFilterWidget,
      qt_meta_data_ColorFilterWidget, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ColorFilterWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ColorFilterWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ColorFilterWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ColorFilterWidget))
        return static_cast<void*>(const_cast< ColorFilterWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int ColorFilterWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void ColorFilterWidget::imageChanged(float * _t1, int _t2, int _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
