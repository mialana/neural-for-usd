#pragma once

#include <QMainWindow>

namespace Ui
{
class UsdWindow;
}

class UsdWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit UsdWindow(QWidget* parent = nullptr);
    Ui::UsdWindow* m_ui;

Q_SIGNALS:
};
