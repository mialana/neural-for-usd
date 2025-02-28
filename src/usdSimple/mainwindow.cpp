#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "camera.h"

#include <QFile>
#include <pxr/usdImaging/usdAppUtils/frameRecorder.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/base/tf/token.h>
#include <iostream>

#include <QSurfaceFormat>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QtOpenGL/QOpenGLFramebufferObjectFormat>
#include <QTextStream>

#include "mycpp/mydefines.h"

void MainWindow::setUpOpenGLFormat()
{
    QSurfaceFormat glFormat = QSurfaceFormat::defaultFormat();
    glFormat.setSamples(4);

    // Create an off-screen surface and bind a gl context to it.
    QOffscreenSurface glWidget = QOffscreenSurface();
    glWidget.setFormat(glFormat);
    glWidget.create();

    QOpenGLContext c = QOpenGLContext();
    c.setFormat(glFormat);
    c.create();
    c.makeCurrent(&glWidget);

    QOpenGLFramebufferObjectFormat glFBOFormat = QOpenGLFramebufferObjectFormat();
    QOpenGLFramebufferObject f = QOpenGLFramebufferObject(QSize(1, 1), glFBOFormat);
    f.bind();

    return;
}

std::string buildFrameString(int f)
{
    std::ostringstream last;
    last << "/Users/liu.amy05/Documents/Neural-for-USD/data/japanesePlaneToy/internalVal/r";
    if (f / 10 >= 1) {
        last << "0" << f;
    } else {
        last << "00" << f;
    }

    last << ".png";

    return last.str();
}

void MainWindow::record()
{
    if (frame >= 100) {
        timer->stop();
        frame = 0;
        return;
    }
    frame += 1;

    pxr::UsdAppUtilsFrameRecorder frameRecorder = pxr::UsdAppUtilsFrameRecorder(pxr::TfToken(),
                                                                                true);

    frameRecorder.SetColorCorrectionMode(pxr::TfToken::Find("sRGB"));
    frameRecorder.SetComplexity(4.0);
    frameRecorder.SetDomeLightVisibility(true);

    std::string outputImagePath = buildFrameString(frame);

    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(
        "/Users/liu.amy05/Documents/Neural-for-USD/japanesePlaneToy.usda");
    pxr::UsdGeomCamera camera = pxr::UsdGeomCamera::Define(stage,
                                                           pxr::SdfPath("/Xform_MyCam/MyCam"));

    qDebug() << frameRecorder.Record(stage, camera, frame, outputImagePath);

    pm->load(QString::fromStdString(buildFrameString(frame)));

    ui->label->setPixmap(*pm);

    QString s;
    QTextStream t(&s);
    t << "Recording frame" << frame + 1 << QString("...");

    ui->pushButton->setText(s);

    qDebug().Nq() << "Frame" << frame << "recorded!";

    return;
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QPushButton* b = ui->pushButton;

    connect(b, &QPushButton::clicked, this, &MainWindow::slot_beginDataCollection);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::slot_beginDataCollection()
{
    // setUpOpenGLFormat();

    Camera c = Camera();

    QLabel* l = ui->label;
    l->setScaledContents(true);
    pm = new QPixmap();

    timer = new QTimer(this);

    connect(timer, &QTimer::timeout, this, QOverload<>::of(&MainWindow::record));
    timer->start(1000);
    return;
}
