#include "usdgl.h"

#include "xformfiltersceneindex.h"

#include <pxr/usd/usdGeom/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/rendererPlugin.h>

UsdGL::UsdGL(QWidget* parent)
    : OpenGLContext(parent)
    , m_model(std::make_unique<Model>())
    , m_engine(nullptr)
    , m_stage(nullptr)
    , m_camera()
    , m_timer()
    , m_width(400)
    , m_height(400)
{
    connect(&m_timer, &QTimer::timeout, this, &UsdGL::tick);
}

UsdGL::~UsdGL()
{
    makeCurrent();
}

void UsdGL::initializeGL()
{
    initializeOpenGLFunctions();  // Qt function to load OpenGL symbols

    // Set default clear color (background)
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Enable depth testing (Hydra uses depth by default)
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // enable blending if you ever have transparent AOVs
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Load USD stage and get camera
    m_stage = pxr::UsdStage::Open(
        "/Users/Dev/Projects/Neural-for-USD/assets/japanesePlaneToy/japanesePlaneToy.usda");
    if (!m_stage) {
        qWarning("Failed to open USD stage.");
        return;
    }
    m_camera = UsdGeomCamera::Define(m_stage, SdfPath("/Xform_MyCam/MyCam")).GetCamera(0);

    m_model->SetStage(m_stage);

    UsdImagingCreateSceneIndicesInfo info;
    info.displayUnloadedPrimsWithBounds = false;

    UsdImagingSceneIndices sceneIndices = UsdImagingCreateSceneIndices(info);
    sceneIndices.stageSceneIndex->SetStage(m_stage);
    sceneIndices.stageSceneIndex->SetTime(UsdTimeCode::Default());

    HdSceneIndexBaseRefPtr filteredSceneIndex = XformFilterSceneIndex::New(
        sceneIndices.finalSceneIndex);

    TfToken defaultRenderer = Engine::GetDefaultRendererPlugin();
    m_engine = std::make_unique<Engine>(filteredSceneIndex, defaultRenderer);

    m_timer.start(16);
}

void UsdGL::resizeGL(int w, int h)
{
    m_width = w;
    m_height = h;

    if (m_engine) {
        m_engine->SetRenderSize(w, h);
    }
}

//This function is called by Qt any time your GL window is supposed to update
//For example, when the function update() is called, paintGL is called implicitly.
void UsdGL::paintGL()
{
    if (!m_engine) {
        return;
    }

    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Get current size (in case resized)
    int w = width();
    int h = height();
    m_width = w;
    m_height = h;

    // Update engine's internal render size
    m_engine->SetRenderSize(w, h);

    // Setup camera matrices from your loaded GfCamera
    GfMatrix4d viewMatrix = m_camera.GetFrustum().ComputeViewMatrix();
    GfMatrix4d projMatrix = m_camera.GetFrustum().ComputeProjectionMatrix();

    m_engine->SetCameraMatrices(viewMatrix, projMatrix);

    // Execute Hydra render pipeline
    m_engine->Prepare();
    m_engine->Render();
}

void UsdGL::tick()
{
    update();
}
