#pragma once

#include "camera.h"

#include <QMainWindow>

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
    void slot_findUsdFilePath();

private:
    Ui::MainWindow* m_ui;
    Camera* m_camera;
};
