#include "usdgl.h"

#include <pxr/usd/usdGeom/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/rendererPlugin.h>

#include <pxr/usdImaging/usdAppUtils/frameRecorder.h>

UsdGL::UsdGL(QWidget* parent)
    : OpenGLContext(parent)
    , mp_scene(std::make_unique<Scene>())
    , mp_engine(nullptr)
    , _engine(nullptr)
    , m_frameBuffer(this, this->width(), this->height(), this->devicePixelRatio())
    , m_timer()
{
    connect(&m_timer, &QTimer::timeout, this, &UsdGL::tick);

    std::string PATH = "/Users/Dev/Projects/Neural-for-USD/assets/testAssets/simpleCube.usda";
    std::string CAM_PATH = "/SimpleCube/primaryCam";
    mp_scene->initialize(PATH, SdfPath(CAM_PATH));
}

UsdGL::~UsdGL()
{
    makeCurrent();
}

void UsdGL::initializeGL()
{
    initializeOpenGLFunctions();

    // OpenGL-related setup

    glClearColor(1., 0.5, 0.5, 1);
    printGLErrorLog();
    m_frameBuffer.create();

    TfToken plugin = Engine::GetDefaultRendererPlugin();
    // _engine = std::make_unique<Engine>(mp_scene->getFinalSceneIndex(), plugin);
    mp_engine = std::make_unique<MyEngine>(this, mp_scene.get(), width(), height());

    m_timer.start(16);
}

void UsdGL::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);

    m_frameBuffer.resize(width(), height(), devicePixelRatio());
    m_frameBuffer.destroy();
    m_frameBuffer.create();


    if (mp_engine) {
        mp_engine->setRenderSize(w, h);
    }

    // if (_engine) {
    //     _engine->SetRenderSize(w, h);
    // }

    qDebug() << "New width is:" << w;
    qDebug() << "New height is:" << h;
    qDebug() << "Device Pixel Ratio is:" << devicePixelRatio();
}

//This function is called by Qt any time your GL window is supposed to update
//For example, when the function update() is called, paintGL is called implicitly.
void UsdGL::paintGL()
{
    if (!mp_engine) {
        return;
    }

    // if (!_engine) {
    //     return;
    // }

    // Get current size (in case resized)
    int w = width();
    int h = height();

    m_frameBuffer.bindFrameBuffer();
    // Update engine's internal render size
    mp_engine->setRenderSize(w, h);
    // Setup engine's camera matrices from scene's primary camera
    mp_engine->setCameraMatrices(mp_scene->getCamView(), mp_scene->getCamView());

    mp_engine->prepareDefaultLighting();

    // Execute Hydra render pipeline
    mp_engine->Render();

    SdfPathVector paths;

    paths.push_back(SdfPath("/SimpleCube/cubeShape"));

    // _engine->SetSelection(paths);

    // _engine->SetRenderSize(w, h);
    // _engine->SetCameraMatrices(mp_scene->getCamView(), mp_scene->getCamView());

    // _engine->Prepare();
    // _engine->Render();

    // qDebug() << "called";

    // GLuint textureHandle = m_engine->GetRenderBufferData();

    // Clear buffers
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glBindTexture(GL_TEXTURE_2D, textureHandle);

    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    // glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureHandle, 0);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glViewport(0, 0, width(), height());
}

void UsdGL::tick()
{
    this->update();
}

void UsdGL::slot_saveEngineRenderToFile(bool signaled)
{
    qDebug() << "Attempting save to file...";

    this->makeCurrent();

    // _engine->GetRenderBufferData();
    mp_engine->renderToFile();
    this->doneCurrent();
}
