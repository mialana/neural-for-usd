#include "mycpp/myutils.h"
#include "mainwindow.h"

#include <QDebug>
#include <QApplication>

int main(int argc, char* argv[])
{
    startup::doSimpleSetup();

    QApplication a(argc, argv);

    MainWindow w;
    w.show();

    return a.exec();
}
