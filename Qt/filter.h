#ifndef COLOR_FILTER_WIDGET_H
#define COLOR_FILTER_WIDGET_H

#include <QColor>
#include <QImage>
#include <QPainterPath>
#include <QWidget>
#include <QSpinBox>
#include <QSlider>
#include <QPushButton>
#include <QLabel>
#include <QColorDialog>

#include "painterwidget.h"
#include "glview.h"

namespace Ui {
    class ColorFilter;
}

class ColorFilterWidget : public QWidget {
    Q_OBJECT
public:
    ColorFilterWidget(GLView* gl, QWidget *parent = 0);
    ~ColorFilterWidget();

public slots:
    //draw brush at point p, coordinates are normalized [0,1]
    void drawBrush(float x, float y);
    void updateColor(const QColor &c);
    void imageChangedSlot(float* img, int width, int height);
    void onDataSliderChange(int v);
    void onProjSliderChange(int v);
    void onCutSliderChange(int v);
    void openImage(QString i){ painter_widget->openImage(i); }
    void saveImage(QString i) {painter_widget->saveImage(i,"png"); }

signals:
    void imageChanged(float* img, int width, int height);

protected:
    void changeEvent(QEvent *e);
    void colorButtonPressed();

private:
    Ui::ColorFilter *ui;
    PainterWidget *painter_widget;
    QSpinBox    *spin_box;
    QSlider     *slider,*dataSlider,*projSlider, *cutSlider;
    QColorDialog* color_dialog;
    QPushButton* color_button;
    QLabel *dataLabel,*projLabel, *cutLabel;
    GLView* glview;
};

#endif // FILTER_H
