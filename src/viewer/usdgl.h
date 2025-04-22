#pragma once

#include "openglcontext.h"
#include "framebuffer.h"
#include "scene.h"
#include "myengine.h"
#include "engine.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/camera.h>

#include <QTimer>
#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_0_Core>

PXR_NAMESPACE_USING_DIRECTIVE

class UsdGL : public OpenGLContext
{
    Q_OBJECT

private:
    std::unique_ptr<Scene> mp_scene;
    std::unique_ptr<MyEngine> mp_engine;
    std::unique_ptr<Engine> _engine;
    FrameBuffer2D m_frameBuffer;

    QTimer m_timer;

public:
    explicit UsdGL(QWidget* parent = nullptr);
    ~UsdGL();

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

public Q_SLOTS:
    void tick();

    void slot_saveEngineRenderToFile(bool signaled = true);
};
