#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QThreadPool>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
    , m_camera(nullptr)
    , m_timer()
{
    m_ui->setupUi(this);

    this->defaultInit();

    // Tab 1
    connect(m_ui->pushButton_usdStage, &QPushButton::clicked, this, &MainWindow::slot_findUsdStagePath);
    connect(m_ui->pushButton_domeLight, &QPushButton::clicked, this, &MainWindow::slot_findDomeLightPath);
    connect(m_ui->pushButton_renderPreview, &QPushButton::clicked, this, &MainWindow::slot_renderPreview);

    // Tab 2
    connect(m_ui->pushButton_dataCollect, &QPushButton::clicked, this, &MainWindow::slot_beginDataCollection);
    connect(&m_timer, &QTimer::timeout, this, &MainWindow::slot_handleUpdateProgressBar);

    // this->setStyleSheet("QProgressBar {\nbackground-color: #C0C6CA;\nborder: 0px;\npadding: 0px;\nheight: 100px;\n}\nQProgressBar::chunk {\nbackground: #7D94B0;\nwidth:5px\n}");
}

MainWindow::~MainWindow()
{
    delete m_ui;
    delete m_camera;
    close();
}

void MainWindow::slot_findUsdStagePath()
{
    // Open file dialog with USD file filter and default directory
    QString stageFilePath = QFileDialog::getOpenFileName(nullptr,
                                               "Select USD File",
                                               PROJECT_SOURCE_DIR
                                                   + QString("/assets/"),
                                               "USD Files (*.usd *.usda *.usdc)");

    if (!QFile(stageFilePath).exists()) {
        return;
    }
    m_ui->lineEdit_usdStage->setText(stageFilePath);
}

void MainWindow::slot_findDomeLightPath()
{
    // Open file dialog with USD file filter and default directory
    QString domeLightPath = QFileDialog::getOpenFileName(nullptr,
                                                         "Select Dome Light File Path",
                                                         PROJECT_SOURCE_DIR
                                                             + QString("/assets/domelights"),
                                                         "USD Files (*.usd *.usda *.usdc)");

    if (!QFile(domeLightPath).exists()) {
        return;
    }
    m_ui->lineEdit_domeLight->setText(domeLightPath);
}

void MainWindow::slot_renderPreview()
{
    QString stageFilePath = m_ui->lineEdit_usdStage->text();
    QString domeLightPath = m_ui->lineEdit_domeLight->text();

    m_camera = new Camera(stageFilePath, domeLightPath);
}

void MainWindow::slot_beginDataCollection()
{
    m_timer.start(16);

    QThreadPool::globalInstance()->start([this]() {
        m_camera->record();
        QMetaObject::invokeMethod(this, [this]() {
            m_timer.stop();
            m_camera->toJson();
            m_ui->progressBar->setValue(100);
        }, Qt::QueuedConnection);
    });

    return;
}

void MainWindow::defaultInit()
{
    this->move(100, -995);
    this->setFixedSize(1024, 768);

    m_ui->lineEdit_usdStage->setText(PROJECT_SOURCE_DIR + QString("/assets/simpleCube/simpleCube.usda"));
    m_ui->lineEdit_domeLight->setText(PROJECT_SOURCE_DIR + QString("/assets/domelights/squash_court_4k.hdr"));

    this->slot_renderPreview();
}

void MainWindow::slot_handleUpdateProgressBar()
{
    double progress = m_camera->getCurrProgress();

    m_ui->progressBar->setValue(progress * 100);
    qDebug() << "Progress bar set to" << progress;
}
