#include "usdgl.h"
#include <mycpp/myglm.h>

#include <QApplication>
#include <QKeyEvent>
#include <QDir>
#include <QFileDialog>

#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/prim.h>

#include <pxr/usd/usd/stageCache.h>

UsdGL::UsdGL(QWidget* parent)
    : OpenGLContext(parent)
    , m_engine()
    , m_engineParams()
    , m_stage()
    , m_camera()
    , m_scene()
{
    setFocusPolicy(Qt::StrongFocus);
}

UsdGL::~UsdGL()
{
    makeCurrent();
}

void UsdGL::initializeGL()
{
    // Create an OpenGL context using Qt's QOpenGLFunctions_3_2_Core class
    // If you were programming in a non-Qt context you might use GLEW (GL Extension Wrangler)instead
    initializeOpenGLFunctions();
    // Print out some information about the current OpenGL context
    debugContextVersion();

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // Set a few settings/modes in OpenGL rendering
    // Set the color with which the screen is filled at the start of each render call.
    glClearColor(0.5, 0.5, 0.5, 1);

    printGLErrorLog();

    // load usd
    m_stage = pxr::UsdStage::Open(
        "/Users/Dev/Desktop/cis7000-assets/Assets/campfire/campfire.usda");

    // // camera
    m_camera = GfCamera();
    pxr::GfMatrix4d t;
    // t.SetTranslate(pxr::GfVec3d::ZAxis() * 1.f);
    m_camera.SetTransform(t);
    m_camera.SetFocusDistance(1.f);

    // renderer
    m_engineParams.enableLighting = true;

    m_timer.start();
}

void UsdGL::resizeGL(int w, int h)
{
    printGLErrorLog();
}

//This function is called by Qt any time your GL window is supposed to update
//For example, when the function update() is called, paintGL is called implicitly.
void UsdGL::paintGL()
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glEnable(GL_BLEND);

    m_scene.prepare(m_timer.elapsed() / 100.0f);
    m_scene.draw(width(), height());
    m_timer.restart();

    // m_engine.SetRenderViewport(pxr::GfVec4d(0,
    //                                         0,
    //                                         this->width() * this->devicePixelRatio(),
    //                                         this->width() * this->devicePixelRatio()));
    // m_engine.Render(m_stage->GetPseudoRoot(), m_engineParams);
    // m_engine.SetCameraPath(SdfPath("/Xform_MyCam/MyCam"));

    // m_engine.SetCameraState(m_camera.GetFrustum().ComputeViewMatrix(),
    //                         m_camera.GetFrustum().ComputeProjectionMatrix());

    update();
}

void UsdGL::tick()
{
    update();
}
