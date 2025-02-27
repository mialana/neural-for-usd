#include "mycpp/myutils.h"
#include "mainwindow.h"

#include <QDebug>
#include <QApplication>
#include <QSurfaceFormat>

int main(int argc, char* argv[])
{
    startup::doSimpleSetup();

    QApplication a(argc, argv);

    qDebug() << "HELLO!";

    MainWindow w;
    w.show();

    return a.exec();
}
