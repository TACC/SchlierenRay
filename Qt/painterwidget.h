#ifndef PAINTERWIDGET_H
#define PAINTERWIDGET_H

#include <QWidget>
#include <QImage>
#include <QColor>
#include <QPoint>

class PainterWidget : public QWidget
{
    Q_OBJECT

public:
    PainterWidget(QWidget *parent = 0);

    bool openImage(const QString &fileName);
    bool saveImage(const QString &fileName, const char *fileFormat);
    void setImage(const QImage &image);

public slots:
    void setBrushSize(int s);
    void setBrushColor(QColor color);
    
    //draw brush at point p, coordinates are normalized [0,1]
    void drawBrush(float x, float y);
    void drawBrushAbsolute(QPoint p);

signals:
    void imageChanged(float* img, int width, int height);

protected:
    void paintEvent(QPaintEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

private:
    void setupPainter(QPainter &painter);
    void updateBrush();
        void clear();

    QImage image, soft_brush;
    QColor brush_color;
    QPoint last_pos;
    int brush_size;
    float* cutoffData;
};

#endif // PAINTERWIDGET_H
