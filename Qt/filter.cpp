#include <QPainter>
#include <QMouseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <sstream>

#include "filter.h"
#include "ui_filter.h"
#include "painterwidget.h"
using namespace std;

ColorFilterWidget::ColorFilterWidget(GLView* gl, QWidget *parent) :
        QWidget(parent), painter_widget(new PainterWidget()), glview(gl)
{
    ui->setupUi(this);

   // setAttribute(Qt::WA_StaticContents);
   // setAttribute(Qt::WA_NoBackground);

    QVBoxLayout *layout = new QVBoxLayout();
    {
        layout->addWidget(painter_widget);
        QGroupBox* hbox = new QGroupBox();
        {
        QHBoxLayout* hlayout = new QHBoxLayout();
        spin_box = new QSpinBox();
        spin_box->setValue(20);
        slider = new QSlider(Qt::Horizontal);
        slider->setRange(1,99);
        slider->setValue(20);

        hlayout->addWidget(spin_box);
        hlayout->addWidget(slider);
        hbox->setLayout(hlayout);
    }

        QGroupBox* hbox2 = new QGroupBox();
        {
        QHBoxLayout* hlayout2 = new QHBoxLayout();
        dataLabel = new QLabel();
        dataLabel->setText("1.0");
        dataSlider = new QSlider(Qt::Horizontal);
        dataSlider->setRange(1,10000);
        dataSlider->setValue(100);
//        dataSlider->setSingleStep(0.01);

        QLabel* label = new QLabel("Scalar:");
        hlayout2->addWidget(label);
        hlayout2->addWidget(dataSlider);
        hlayout2->addWidget(dataLabel);
        hbox2->setLayout(hlayout2);
    }


        QGroupBox* hbox3 = new QGroupBox();
        {
        QHBoxLayout* hlayout3 = new QHBoxLayout();
        projLabel = new QLabel();
        projLabel->setText("0.0");
        projSlider = new QSlider(Qt::Horizontal);
        projSlider->setRange(0,10000);
        projSlider->setValue(0);
//        projSlider>setSingleStep(0.01);

        QLabel* label2 = new QLabel("Projection:");
        hlayout3->addWidget(label2);
        hlayout3->addWidget(projSlider);
        hlayout3->addWidget(projLabel);
        hbox3->setLayout(hlayout3);
    }

        QGroupBox* hbox4 = new QGroupBox();
        {
        QHBoxLayout* hlayout4 = new QHBoxLayout();
        cutLabel = new QLabel();
        cutLabel->setText("1.0");
        cutSlider = new QSlider(Qt::Horizontal);
        cutSlider->setRange(100,1000000);
        cutSlider->setSingleStep(100);
        cutSlider->setValue(500000);
//        projSlider>setSingleStep(0.01);

        QLabel* label3 = new QLabel("Cutoff:");
        hlayout4->addWidget(label3);
        hlayout4->addWidget(cutSlider);
        hlayout4->addWidget(cutLabel);
        hbox4->setLayout(hlayout4);
    }

        layout->addWidget(hbox);
        color_dialog = new QColorDialog();
        color_button = new QPushButton(tr("Color..."));
        layout->addWidget(color_button);
        layout->addWidget(hbox2);
        layout->addWidget(hbox3);
        layout->addWidget(hbox4);
      }
       setLayout(layout);

       connect(slider, SIGNAL(valueChanged(int)), painter_widget, SLOT(setBrushSize(int)));
       connect(slider, SIGNAL(valueChanged(int)), spin_box, SLOT(setValue(int)));

       connect(spin_box, SIGNAL(valueChanged(int)), painter_widget, SLOT(setBrushSize(int)));
       connect(spin_box, SIGNAL(valueChanged(int)), slider, SLOT(setValue(int)));

       connect(color_button, SIGNAL(pressed()), color_dialog, SLOT(show()));
       connect(color_dialog, SIGNAL(currentColorChanged(QColor)), painter_widget, SLOT(setBrushColor(QColor)));
       connect(color_dialog, SIGNAL(currentColorChanged(QColor)), this, SLOT(updateColor(QColor)));

       connect(painter_widget, SIGNAL(imageChanged(float*,int,int)), this, SLOT(imageChangedSlot(float*,int,int)));

       connect(dataSlider, SIGNAL(valueChanged(int)), this, SLOT(onDataSliderChange(int)));
       connect(projSlider, SIGNAL(valueChanged(int)), this, SLOT(onProjSliderChange(int)));
       connect(cutSlider, SIGNAL(valueChanged(int)), this, SLOT(onCutSliderChange(int)));
}

ColorFilterWidget::~ColorFilterWidget()
{
    delete ui;
}

void ColorFilterWidget::changeEvent(QEvent *e)
{
    QWidget::changeEvent(e);
    switch (e->type()) {
    case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        break;
    }
}

void ColorFilterWidget::drawBrush(float x, float y)
{
    painter_widget->drawBrush(x,y);
}

void ColorFilterWidget::colorButtonPressed()
{
    color_dialog->show();
}

void ColorFilterWidget::updateColor(const QColor &c)
{
    const QString COLOR_STYLE("QPushButton { background-color : %1; color : %2; }");

    QColor ChosenColor = c; // Color chosen by the user with QColorDialog
    QColor IdealTextColor = Qt::black;//getIdealTextColor(ChosenColor);
    color_button->setStyleSheet(COLOR_STYLE.arg(ChosenColor.name()).arg(IdealTextColor.name()));
}

void ColorFilterWidget::imageChangedSlot(float* img, int width, int height)
 {
     emit imageChanged(img,width,height);
 }

void ColorFilterWidget::onDataSliderChange(int v)
 {
  float val = float(v)/100.0f;
  stringstream ss;
  ss << val;
  dataLabel->setText(QString(ss.str().c_str()));
  glview->setDataScale(val);
 }

void ColorFilterWidget::onProjSliderChange(int v)
 {
     float val = float(v)/100.0f;
     stringstream ss;
     ss << val;
     projLabel->setText(QString(ss.str().c_str()));
     glview->setProjectionDistance(val);
 }

void ColorFilterWidget::onCutSliderChange(int v)
 {
     float val = float(v)/100.0f;
     stringstream ss;
     ss << val;
     cutLabel->setText(QString(ss.str().c_str()));
     glview->setCutoffScale(val);
 }
