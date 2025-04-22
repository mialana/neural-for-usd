#include "usdwindow.h"
#include "ui_usdwindow.h"

UsdWindow::UsdWindow(QWidget* parent)
    : QMainWindow{parent}
    , m_ui(new Ui::UsdWindow)
{
    m_ui->setupUi(this);

    this->move(QPoint(0, 0));

    connect(m_ui->pushButton, &QPushButton::clicked, m_ui->MyUsdGL, &UsdGL::slot_saveEngineRenderToFile);
}

