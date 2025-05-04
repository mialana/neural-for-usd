#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QThreadPool>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
    , m_timer()
{
    m_ui->setupUi(this);

    this->initDefaults();

    // Tab 1
    connect(m_ui->pushButton_usdStage, &QPushButton::clicked, this, &MainWindow::slot_findUsdStagePath);
    connect(m_ui->pushButton_domeLight, &QPushButton::clicked, this, &MainWindow::slot_findDomeLightPath);
    connect(m_ui->pushButton_renderPreview, &QPushButton::clicked, this, &MainWindow::slot_renderPreview);
    connect(m_ui->horizontalSlider, &QSlider::sliderMoved, this, &MainWindow::slot_handleUpdateSlider);

    // Tab 2
    connect(m_ui->pushButton_dataCollect, &QPushButton::clicked, this, &MainWindow::slot_beginDataCollection);
    connect(&m_timer, &QTimer::timeout, this, &MainWindow::slot_handleUpdateProgressBar);
}

MainWindow::~MainWindow()
{
    delete m_ui;
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

    // m_camera = new Camera(stageFilePath, domeLightPath);
}

void MainWindow::slot_beginDataCollection()
{
    m_timer.start(16);

    // QThreadPool::globalInstance()->start([this]() {
    //     m_camera->record();
    //     QMetaObject::invokeMethod(this, [this]() {
    //         m_timer.stop();
    //         m_camera->toJson();
    //         m_ui->progressBar->setValue(100);
    //     }, Qt::QueuedConnection);
    // });

    return;
}

void MainWindow::slot_handleUpdateProgressBar()
{
    // double progress = m_camera->getCurrProgress();

    // m_ui->progressBar->setValue(progress * 100);
    // qDebug() << "Progress bar set to" << progress;
}

void MainWindow::slot_handleUpdateSlider(int value)
{
    m_ui->label_frameNum->setText(QString("Frame %1").arg(value));
    this->m_ui->myGl->slot_setStageManagerCurrentFrame(value);

    this->m_ui->myGl->slot_changeRenderEngineMode("fixed");
}

void MainWindow::initDefaults()
{
    QFile file(":/style/style.qss");
    if (file.open(QFile::ReadOnly | QFile::Text)) {
        QString styleSheet = QString::fromUtf8(file.readAll());
        this->setStyleSheet(styleSheet);
        qDebug() << "Style sheet successfully read.";
    } else {
        qDebug() << "Style sheet couldn't be read.";
    }

    this->move(100, -995);
    this->setFixedSize(1024, 768);

    const QString defaultUsdStagePath = PROJECT_SOURCE_DIR + QString("/assets/campfire/campfire.usd");
    const QString defaultLuxDomeLightPath = PROJECT_SOURCE_DIR + QString("/assets/domelights/squash_court_4k.hdr");

    m_ui->lineEdit_usdStage->setText(defaultUsdStagePath);
    m_ui->lineEdit_domeLight->setText(defaultLuxDomeLightPath);

    m_ui->myGl->loadStageManager(defaultUsdStagePath, defaultLuxDomeLightPath);
}
