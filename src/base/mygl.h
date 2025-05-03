#pragma once

#include <mycpp/mydefines.h>

#include "stagemanager.h"

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

class MyGL : public OpenGLContext
{
    Q_OBJECT

public:
    explicit MyGL(QWidget* parent = nullptr);
    ~MyGL();

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

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

    uPtr<StageManager> m_stage;

    void initDefaultStage();

protected:
    void keyPressEvent(QKeyEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

public Q_SLOTS:
    void tick();

    void slot_triggerRenderPreview(bool signaled = true);
};
