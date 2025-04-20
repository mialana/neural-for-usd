#pragma once

#include "openglcontext.h"
#include "models/model.h"
#include "engine.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/camera.h>

#include "sceneindices/gridsceneindex.h"

#include <QTimer>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include <QOpenGLShader>

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

PXR_NAMESPACE_USING_DIRECTIVE

class UsdGL : public OpenGLContext
{
    Q_OBJECT

private:
    std::unique_ptr<Model> m_model;
    std::unique_ptr<Engine> m_engine;

    UsdStageRefPtr m_stage;
    GfCamera m_camera;
    GridSceneIndexRefPtr m_gridSceneIndex;

    // QOpenGLShader m_screenShader;

    QTimer m_timer;
    int m_width;
    int m_height;

    GLuint m_fbo;
    GLuint m_vao;
    GLuint m_vbo;

public:
    explicit UsdGL(QWidget* parent = nullptr);
    ~UsdGL();

    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

public Q_SLOTS:
    void tick();
};
