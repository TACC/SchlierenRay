/****************************************************************************
** Meta object code from reading C++ file 'painterwidget.h'
**
** Created: Mon Apr 26 09:43:09 2010
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "painterwidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'painterwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_PainterWidget[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      17,   15,   14,   14, 0x0a,
      41,   35,   14,   14, 0x0a,
      67,   63,   14,   14, 0x0a,
      92,   90,   14,   14, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_PainterWidget[] = {
    "PainterWidget\0\0s\0setBrushSize(int)\0"
    "color\0setBrushColor(QColor)\0x,y\0"
    "drawBrush(float,float)\0p\0"
    "drawBrushAbsolute(QPoint)\0"
};

const QMetaObject PainterWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_PainterWidget,
      qt_meta_data_PainterWidget, 0 }
};

const QMetaObject *PainterWidget::metaObject() const
{
    return &staticMetaObject;
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
        switch (_id) {
        case 0: setBrushSize((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: setBrushColor((*reinterpret_cast< QColor(*)>(_a[1]))); break;
        case 2: drawBrush((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 3: drawBrushAbsolute((*reinterpret_cast< QPoint(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
