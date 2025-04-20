#include "usdgl.h"

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
    , m_gridSceneIndex(nullptr)
    , m_timer()
    , m_width(400)
    , m_height(400)
{
    connect(&m_timer, &QTimer::timeout, this, &UsdGL::tick);
}

UsdGL::~UsdGL()
{
    makeCurrent();
    // glDeleteFramebuffers(1, &m_fbo);
}

void UsdGL::initializeGL()
{

    // Load USD stage and get camera
    m_stage = pxr::UsdStage::Open(
        "/Users/Dev/Projects/Neural-for-USD/assets/japanesePlaneToy/japanesePlaneToy.usda");
    if (!m_stage) {
        qWarning("Failed to open USD stage.");
        return;
    }
    m_camera = UsdGeomCamera::Define(m_stage, SdfPath("/Xform_MyCam/MyCam")).GetCamera(0);

    m_gridSceneIndex = GridSceneIndex::New();
    m_model->AddSceneIndexBase(m_gridSceneIndex);

    TfToken defaultRenderer = Engine::GetDefaultRendererPlugin();
    m_engine = std::make_unique<Engine>(m_model->GetFinalSceneIndex(), defaultRenderer);

    // OpenGL-related setup
    initializeOpenGLFunctions();  // Qt function to load OpenGL symbols

    // glEnable(GL_DEPTH_TEST);

    // // Set default clear color (background)
    // glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    // glGenFramebuffers(1, &m_fbo);

    // glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

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

    // GLuint textureHandle = m_engine->GetRenderBufferData();

    // Clear buffers
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glBindTexture(GL_TEXTURE_2D, textureHandle);

    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    // glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureHandle, 0);

}

void UsdGL::tick()
{
    update();
}
