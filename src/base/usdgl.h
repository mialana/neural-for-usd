#pragma once

#include "openglcontext.h"
#include "myframerecorder.h"
#include "mycamera.h"

#include <pxr/usd/usd/stage.h>

#include <QTimer>
#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_0_Core>

#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgiInterop/hgiInterop.h>

PXR_NAMESPACE_USING_DIRECTIVE

class UsdGL : public OpenGLContext
{
    Q_OBJECT

private:
    QTimer m_timer;
    UsdStageRefPtr stage;
    UsdGeomCamera camera;
    MyFrameRecorder frameRecorder;
    HgiUniquePtr _hgi;
    HgiInterop _interop;
    GfMatrix4d camView;
    GfMatrix4d camProj;

    MyCamera myCam;

    GfVec2d m_mousePosPrev;

public:
    explicit UsdGL(QWidget* parent = nullptr);
    ~UsdGL();

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

protected:
    void keyPressEvent(QKeyEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

public Q_SLOTS:
    void tick();

    void slot_saveEngineRenderToFile(bool signaled = true);
};
