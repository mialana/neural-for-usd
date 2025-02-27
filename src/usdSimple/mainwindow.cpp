#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "camera.h"

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
    Camera c = Camera();

    QLabel* l = ui->label;
}
