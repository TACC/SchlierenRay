/********************************************************************************
** Form generated from reading UI file 'filter.ui'
**
** Created: Mon Sep 10 15:34:10 2012
**      by: Qt User Interface Compiler version 4.8.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FILTER_H
#define UI_FILTER_H

#include <QtCore/QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QHeaderView>
#include <QWidget>

QT_BEGIN_NAMESPACE

class Ui_ColorFilter
{
public:

    void setupUi(QWidget *ColorFilter)
    {
        if (ColorFilter->objectName().isEmpty())
            ColorFilter->setObjectName(QString::fromUtf8("ColorFilter"));
        ColorFilter->resize(290, 283);

        retranslateUi(ColorFilter);

        QMetaObject::connectSlotsByName(ColorFilter);
    } // setupUi

    void retranslateUi(QWidget *ColorFilter)
    {
//        ColorFilter->setWindowTitle(QApplication::translate("ColorFilter", "Form", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ColorFilter: public Ui_ColorFilter {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FILTER_H
