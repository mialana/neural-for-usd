#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
    , m_camera(nullptr)
{
    m_ui->setupUi(this);

    connect(m_ui->pushButton, &QPushButton::clicked, this, &MainWindow::slot_beginDataCollection);
    connect(m_ui->pushButton_2, &QPushButton::clicked, this, &MainWindow::slot_findUsdFilePath);
}

MainWindow::~MainWindow()
{
    delete m_ui;
    delete m_camera;
    close();
}

void MainWindow::slot_beginDataCollection()
{
    QString outputPrefix = "/r";
    m_camera->record(outputPrefix, m_ui->progressBar, 106);

    m_camera->toJson();

    return;
}

void MainWindow::slot_findUsdFilePath()
{
    // Open file dialog with USD file filter and default directory
    QString sfp = QFileDialog::getOpenFileName(nullptr,
                                               "Select USD File",
                                               PROJECT_SOURCE_DIR + QString("/assets/"),
                                               "USD Files (*.usd *.usda *.usdc)");

    if (!QFile(sfp).exists()) {
        return;
    }
    m_ui->lineEdit->setText(sfp);

    QString assetDir = QFileInfo(sfp).dir().absolutePath();

    QString hfp = QString(PROJECT_SOURCE_DIR) + "/assets/domelight/HDR_029_Sky_Cloudy_Ref.hdr";
    QString osfp = assetDir + "/data/mystage.usda";
    QString odfp = assetDir + "/data/data.json";
    QString ordp = assetDir + "/data/internalVal";

    m_camera = new Camera(sfp, hfp, osfp, odfp, ordp);
}
