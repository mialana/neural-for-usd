#include "mycpp/mystartup.h"
#include "usdwindow.h"

#include <QDebug>
#include <QApplication>

int main(int argc, char* argv[])
{
    startup::doSimpleSetup();

    QApplication a(argc, argv);

    UsdWindow w = UsdWindow();
    w.show();

    return a.exec();
}
