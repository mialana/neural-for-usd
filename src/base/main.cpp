#include <mycpp/mystartup.h>
#include "mainwindow.h"

#include <QDebug>
#include <QApplication>
#include <QGuiApplication>
#include <QScreen>

int main(int argc, char* argv[])
{
    startup::doSimpleSetup();

    QApplication a(argc, argv);

    MainWindow w;

    w.show();

    return a.exec();
}
