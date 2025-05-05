#include <mycpp/mystartup.h>
#include "mainwindow.h"

#include <QDebug>
#include <QApplication>
#include <QGuiApplication>
#include <QScreen>

#include <QStyleFactory>

int main(int argc, char* argv[])
{
    startup::doSimpleSetup();

    QApplication a(argc, argv);
    a.setStyle(QStyleFactory::create("fusion"));

    MainWindow w;

    w.show();

    return a.exec();
}
