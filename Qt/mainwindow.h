#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QAction>
#include <QMenu>
#include "glview.h"
#include "filter.h"


namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void onLoadData();
    void onLoadImage();
    void onSaveImage();

protected:
    void changeEvent(QEvent *e);

private:
    Ui::MainWindow *ui;
    GLView *view;
    ColorFilterWidget* color_widget;
    QAction* loadDataAct,*loadImageAct,*saveImageAct;
    QMenu *fileMenu;
};

#endif // MAINWINDOW_H
