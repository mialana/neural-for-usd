#pragma once

#include <openglcontext.h>

#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/camera.h>
#include <pxr/usdImaging/usdImagingGL/engine.h>
#include <pxr/usdImaging/usdImagingGL/renderParams.h>

#include <QElapsedTimer>

#include "scene.h"

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

PXR_NAMESPACE_USING_DIRECTIVE

class UsdGL : public OpenGLContext
{
    Q_OBJECT

private:
    QElapsedTimer m_timer;

    UsdImagingGLEngine m_engine;
    UsdImagingGLRenderParams m_engineParams;

    UsdStageRefPtr m_stage;
    GfCamera m_camera;

    Scene m_scene;

public:
    explicit UsdGL(QWidget* parent = nullptr);
    ~UsdGL();

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

public Q_SLOTS:
    void tick();
};
