#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    view = new GLView(this);
    view->resize(512,512);
    setWindowTitle(tr("SchlierenVis"));
   // view->show();
    setCentralWidget(view);

   /* QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(view);
    setLayout(layout);

    layout->setSizeConstraint(QLayout::SetMinimumSize);*/
    color_widget = new ColorFilterWidget(view);
    color_widget->resize(300,1024);
    color_widget->show();


    loadDataAct = new QAction(tr("&Load Data"), this);
    loadDataAct->setShortcuts(QKeySequence::Open);
    loadDataAct->setStatusTip(tr("load data file"));
    connect(loadDataAct, SIGNAL(triggered()), this, SLOT(onLoadData()));

    loadImageAct = new QAction(tr("&Load Image"), this);
    //loadDataAct->setShortcuts(QKeySequence::New);
    loadImageAct->setStatusTip(tr("load image file"));
    connect(loadImageAct, SIGNAL(triggered()), this, SLOT(onLoadImage()));

    saveImageAct = new QAction(tr("&Save Image"), this);
    //loadDataAct->setShortcuts(QKeySequence::New);
    saveImageAct->setStatusTip(tr("save image file"));
    connect(saveImageAct, SIGNAL(triggered()), this, SLOT(onSaveImage()));

    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(loadDataAct);
    fileMenu->addAction(loadImageAct);
    fileMenu->addAction(saveImageAct);
	
    connect(view, SIGNAL(drawFilterAtPoint(float,float)), color_widget, SLOT(drawBrush(float,float)));

#if USE_IMAGE_CUTOFF
    connect(color_widget, SIGNAL(imageChanged(float*,int,int)), view, SLOT(setImageCutoff(float*,int,int)));
#endif
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::changeEvent(QEvent *e)
{
    QMainWindow::changeEvent(e);
    switch (e->type()) {
    case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        break;
    }
}

void MainWindow::onLoadData()
{
    QFileDialog::Options options;
     QString selectedFilter;
QString fileName = QFileDialog::getOpenFileName(this,
                                tr("QFileDialog::getOpenFileName()"),
                                "select data",
                                tr("All Files (*);;Text Files (*.txt)"),
                                &selectedFilter,
                                options);
    if (fileName != "")
      view->loadData(fileName.toStdString());
}

void MainWindow::onLoadImage()
{
        QFileDialog::Options options;
     QString selectedFilter;
   QString fileName = QFileDialog::getOpenFileName(this,
                                tr("QFileDialog::getOpenFileName()"),
                                "select data",
                                tr("All Files (*);;Text Files (*.txt)"),
                                &selectedFilter,
                                options);
   if (fileName != "")
   color_widget->openImage(QString(fileName));
}


void MainWindow::onSaveImage()
{
        QFileDialog::Options options;
     QString selectedFilter;
//QString fileName = QFileDialog::getSaveFileName(this,
//                                tr("QFileDialog::getOpenFileName()"),
//                                "select data",
//                                tr("All Files (*);;Text Files (*.txt)"),
//                                &selectedFilter,
//                                options);

     QFileDialog dialog(this);
 dialog.setFileMode(QFileDialog::AnyFile);

 QStringList fileNames;
 if (dialog.exec())
     fileNames = dialog.selectedFiles();
 else
     return;

    color_widget->saveImage(QString(fileNames.at(0)));
}
