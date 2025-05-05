#pragma once

#include <mycpp/mydefines.h>

#include "stagemanager.h"
#include "renderengine.h"

#include "openglcontext.h"

#include <pxr/usd/usd/stage.h>

#include <QTimer>

PXR_NAMESPACE_USING_DIRECTIVE

class MyGL : public OpenGLContext
{
    Q_OBJECT

public:
    uPtr<StageManager> m_manager;

    explicit MyGL(QWidget* parent = nullptr);
    ~MyGL();

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void loadStageManager(const QString& stagePath, const QString& domeLightPath);
    void handleEngineRecordProcess();

private:
    void resetEngine();

    QTimer m_timer;
    GfVec2d m_mousePosPrev;
    bool isRecording;

    uPtr<RenderEngine> m_engine;

protected:
    void keyPressEvent(QKeyEvent* e) override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

public Q_SLOTS:
    void tick();

    void slot_setStageManagerCurrentFrame(int frame);

    /**
     * @brief slot_changeRenderEngineMode
     * @param mode: "fixed" or "free"
     */
    void slot_changeRenderEngineMode(QString mode);

Q_SIGNALS:
    void engineModeChanged(QString mode);
};
