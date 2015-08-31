/****************************************************************************
** Meta object code from reading C++ file 'filter.h'
**
** Created: Mon Apr 26 09:43:08 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "filter.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'filter.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ColorFilterWidget[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      23,   19,   18,   18, 0x0a,
      48,   46,   18,   18, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_ColorFilterWidget[] = {
    "ColorFilterWidget\0\0x,y\0drawBrush(float,float)\0"
    "c\0updateColor(QColor)\0"
};

const QMetaObject ColorFilterWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_ColorFilterWidget,
      qt_meta_data_ColorFilterWidget, 0 }
};

const QMetaObject *ColorFilterWidget::metaObject() const
{
    return &staticMetaObject;
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
        switch (_id) {
        case 0: drawBrush((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 1: updateColor((*reinterpret_cast< const QColor(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
