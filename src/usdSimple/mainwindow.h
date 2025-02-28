#pragma once

#include <QMainWindow>
#include <QPixmap>
#include <QTimer>

namespace Ui
{
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

public slots:
    void slot_beginDataCollection();

private:
    Ui::MainWindow* ui;
    QPixmap* pm;
    QTimer* timer;

    int frame = 0;

    void record();

    void setUpOpenGLFormat();
};
