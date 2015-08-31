#include <QDebug>
#include <QPainter>
#include <QMouseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QRadialGradient>
#include <iostream>

#include "painterwidget.h"

PainterWidget::PainterWidget(QWidget * parent)
    : QWidget(parent), brush_size(10), brush_color(Qt::black),
    last_pos(-1,-1), image(256,256,QImage::Format_RGB32),soft_brush(32, 32,QImage::Format_RGB32)
{
    clear();
    updateBrush();
    cutoffData = new float[256*256*4];
}

bool PainterWidget::openImage(const QString &fileName)
{
    QImage imaget;
    if (!imaget.load(fileName))
        return false;

    setImage(imaget);
    return true;
}

void PainterWidget::setImage(const QImage &imaget)
{
    image = imaget.convertToFormat(QImage::Format_RGB32);
    update();
    updateGeometry();
}

bool PainterWidget::saveImage(const QString &fileName, const char *fileFormat)
{
    return image.save(fileName, fileFormat);
}

void PainterWidget::setBrushSize(int s)
{
    brush_size = s;
    updateBrush();
    qDebug() << "set brush size:" << s;
}

void PainterWidget::setBrushColor(QColor color)
{
    brush_color = color;
    updateBrush();
}

void PainterWidget::paintEvent(QPaintEvent * /* event */)
{
//QPainter painter(&image);
//            setupPainter(painter);
//			painter.drawPoint(QPoint(20,20));
			
    QPainter painter2(this);
    painter2.drawImage(QPoint(0, 0), image);
	//painter2.drawPoint(QPoint(100,100));

    float* datap = cutoffData;
    for(int i = 0; i < 256; i++)
    {
        for(int j = 0; j < 256; j++)
        {
            QColor c = QColor(image.pixel(j,255-i));
              datap[0] = float(c.red())/255.0f;
              datap[1] = float(c.green())/255.f;
              datap[2] = float(c.blue())/255.f;
              datap[3] = 1.0f;
            datap+=4;
        }
    }
    emit imageChanged(cutoffData, 256, 256);
}

void PainterWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        drawBrushAbsolute(event->pos());
        last_pos = event->pos();
    }
}

void PainterWidget::mouseMoveEvent(QMouseEvent *event)
{
    if ((event->buttons() & Qt::LeftButton) && last_pos != QPoint(-1, -1)) {

            drawBrushAbsolute(event->pos());

        last_pos = event->pos();
    }
}
//! [1]

void PainterWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && last_pos != QPoint(-1, -1)) {
            drawBrushAbsolute(event->pos());

        last_pos = QPoint(-1, -1);
    }
}

void PainterWidget::clear()
{
  //  image.fill(qRgb(255,255,255));
    for(int i = 0; i < 256; i++)
    {
        for(int j = 0; j < 256; j++)
        {
            float v = float(j);
            QColor c = QColor(v,v,v);
            image.setPixel(i,j,c.rgb());
        }
    }
}

void PainterWidget::drawBrush(float x, float y)
{
        x = std::min(1.0f, std::max(0.0f, x));
        y = std::min(1.0f, std::max(0.0f, y));
            QPainter painter(&image);
            setupPainter(painter);
            painter.drawImage(x*image.width()-soft_brush.width()/2,y*image.height()-soft_brush.height()/2,soft_brush);
            update();
}

void PainterWidget::drawBrushAbsolute(QPoint p)
{
        p.setX(std::min(width(), std::max(0, p.x())));
        p.setY(std::min(height(), std::max(0, p.y())));
            QPainter painter(&image);
            setupPainter(painter);
            painter.drawImage(p.x()-soft_brush.width()/2,p.y()-soft_brush.height()/2,soft_brush);
            update();
}

void PainterWidget::setupPainter(QPainter &painter)
{
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor(Qt::black), 12.0));
}

void PainterWidget::updateBrush()
{
    QRadialGradient g;
    g.setCenter(brush_size/2, brush_size/2);
    g.setFocalPoint(brush_size/2, brush_size/2);
    g.setRadius(brush_size/2);
    g.setColorAt(1.0, Qt::black);
    g.setColorAt(0, QColor(100,100,100));

    QImage mask(brush_size, brush_size,QImage::Format_RGB32);
    mask.fill(qRgb(0,0,0));
    QPainter painter2(&mask);
    painter2.fillRect(mask.rect(), g);
    painter2.end();
    soft_brush = QImage(brush_size, brush_size,QImage::Format_RGB32);
    soft_brush.fill(brush_color.rgb());
    soft_brush.setAlphaChannel(mask);
}
