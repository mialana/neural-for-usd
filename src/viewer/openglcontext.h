#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_0_Core>

class OpenGLContext : public QOpenGLWidget, public QOpenGLFunctions_4_0_Core
{
    Q_OBJECT

public:
    OpenGLContext(QWidget* parent);
    ~OpenGLContext();

    void debugContextVersion();
    void printGLErrorLog();
    void printLinkInfoLog(int prog);
    void printShaderInfoLog(int shader);
};
