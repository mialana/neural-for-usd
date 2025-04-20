#include "mycpp/mystartup.h"

#include <QDebug>
#include <QApplication>

int main(int argc, char* argv[])
{
    startup::doSimpleSetup();

    QApplication a(argc, argv);

    return a.exec();
}
