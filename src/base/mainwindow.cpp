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
    connect(m_ui->horizontalSlider, &QSlider::sliderReleased, this, &MainWindow::slot_handleUpdateSlider);

    connect(m_ui->myGl, &MyGL::engineModeChanged, this, &MainWindow::slot_handleEngineModeChanged);

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
                                                         "HDR Files (*.hdr *.hdri)");

    if (!QFile(domeLightPath).exists()) {
        return;
    }
    m_ui->lineEdit_domeLight->setText(domeLightPath);
}

void MainWindow::slot_renderPreview()
{
    QString stageFilePath = m_ui->lineEdit_usdStage->text();
    QString domeLightPath = m_ui->lineEdit_domeLight->text();

    m_ui->myGl->loadStageManager(stageFilePath, domeLightPath);

    QFileInfo info(stageFilePath);
    QString assetName = info.baseName();
    assetName[0] = assetName[0].toUpper();
    m_ui->label_stageName->setText(assetName);

    this->slot_handleUpdateSlider();
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

void MainWindow::slot_handleUpdateSlider()
{
    int position = m_ui->horizontalSlider->sliderPosition();
    this->m_ui->myGl->slot_setStageManagerCurrentFrame(position);

    this->m_ui->myGl->slot_changeRenderEngineMode("fixed");

    m_ui->label_frameNum->setText(QString("Frame %1").arg(position));

    QFont font = m_ui->label_frameNum->font();

    font.setWeight(QFont::Bold);
    m_ui->label_frameNum->setFont(font);
}

void MainWindow::slot_handleEngineModeChanged(QString mode)
{
    if (mode == "free") {
        QFont font = m_ui->label_frameNum->font();

        font.setWeight(QFont::Thin);
        m_ui->label_frameNum->setFont(font);
    }

    m_ui->label_cameraMode->setText(QString("Camera Mode: %1").arg(mode.toUpper()));
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

    this->slot_renderPreview();
}
