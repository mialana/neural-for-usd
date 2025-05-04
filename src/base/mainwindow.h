#pragma once

#include <QMainWindow>
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

public Q_SLOTS:
    void slot_renderPreview();
    void slot_findUsdStagePath();
    void slot_findDomeLightPath();

    void slot_beginDataCollection();

private:
    Ui::MainWindow* m_ui;
    QTimer m_timer;

    void initDefaults();

private Q_SLOTS:
    void slot_handleUpdateProgressBar();

    void slot_handleUpdateSlider();

    void slot_handleEngineModeChanged(QString mode);
};
