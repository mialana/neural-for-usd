#include "mygl.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <QKeyEvent>
#include <QApplication>

MyGL::MyGL(QWidget* parent)
    : OpenGLContext(parent)
    , m_timer()
    , frameRecorder(pxr::TfToken(), true)
    , _hgi(Hgi::CreatePlatformDefaultHgi())
    , myCam(this->width(), this->height())
    , m_mousePosPrev()
{
    connect(&m_timer, &QTimer::timeout, this, &MyGL::tick);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
}

MyGL::~MyGL()
{
    makeCurrent();
}

void MyGL::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.5, 0.5, 0.5, 1);
    printGLErrorLog();

    // stage = UsdStage::Open("/Users/Dev/Projects/Neural-for-USD/assets/testAssets/simpleCube.usda");
    // camera = UsdGeomCamera::Define(stage, SdfPath("/SimpleCube/primaryCam"));

    // stage = UsdStage::Open("/Users/Dev/Projects/Neural-for-USD/assets/japanesePlaneToy/data/mystage.usda");
    // camera = UsdGeomCamera::Define(stage, SdfPath("/Xform_MyCam/MyCam"));

    stage = UsdStage::Open("/Users/Dev/Projects/Neural-for-USD/assets/campfire/campfire.usd");
    camera = UsdGeomCamera::Define(stage, SdfPath("/campfire/primaryCam"));

    // GfFrustum frustum = camera.GetCamera(0).GetFrustum();

    // myCam = MyCamera(width(), height(), frustum.GetPosition(), frustum.ComputeLookAtPoint(), frustum.ComputeUpVector());

    myCam = MyCamera(width(), height());

    frameRecorder.SetColorCorrectionMode(pxr::TfToken::Find("sRGB"));
    frameRecorder.SetComplexity(1.0);
    frameRecorder.SetDomeLightVisibility(true);

    m_timer.start(16);
}

void MyGL::resizeGL(int w, int h)
{
    // glViewport(0, 0, w, h);
    myCam = MyCamera(w, h, myCam.eye, myCam.ref, myCam.worldUp);
    myCam.recomputeAttributes();

    qDebug() << "New width is:" << w;
    qDebug() << "New height is:" << h;
    qDebug() << "Device Pixel Ratio is:" << devicePixelRatio();
}

void MyGL::paintGL()
{
    frameRecorder.getTextureHandle(stage,
                                   camera,
                                   this->defaultFramebufferObject(),
                                   _hgi.get(),
                                   _interop,
                                   myCam.createGfCamera(),
                                   width(),
                                   height(),
                                   devicePixelRatio());
}

void MyGL::tick()
{
    this->update();
}

void MyGL::keyPressEvent(QKeyEvent* e)
{
    float amount = 2.0f;
    if (e->modifiers() & Qt::ShiftModifier) {
        amount = 10.0f;
    }

    switch (e->key()) {
        case (Qt::Key_Escape): QApplication::quit(); break;
        case (Qt::Key_Right): myCam.rotateAboutUp(-amount); break;
        case (Qt::Key_Left): myCam.rotateAboutUp(amount); break;
        case (Qt::Key_Up): myCam.rotateAboutRight(-amount); break;
        case (Qt::Key_Down): myCam.rotateAboutRight(amount); break;

        case (Qt::Key_W): myCam.translateAlongForward(amount); break;
        case (Qt::Key_S): myCam.translateAlongForward(-amount); break;
        case (Qt::Key_D): myCam.translateAlongRight(amount); break;
        case (Qt::Key_A): myCam.translateAlongRight(-amount); break;
        case (Qt::Key_Q): myCam.translateAlongUp(-amount); break;
        case (Qt::Key_E): myCam.translateAlongUp(amount); break;
    }
    myCam.recomputeAttributes();
    update();
}

void MyGL::mousePressEvent(QMouseEvent* e)
{
    qDebug() << "x:" << myCam.eye[0] << "y:" << myCam.eye[1] << "z:" << myCam.eye[2];
    if (e->buttons() & (Qt::LeftButton | Qt::RightButton | Qt::MiddleButton)) {
        m_mousePosPrev = GfVec2d(e->pos().x(), e->pos().y());
    }
    myCam.recomputeAttributes();
    update();
}

void MyGL::mouseMoveEvent(QMouseEvent* e)
{
    GfVec2d pos(e->pos().x(), e->pos().y());
    if (e->buttons() & Qt::LeftButton) {
        // Rotation
        GfVec2d diff = 0.04f * (pos - m_mousePosPrev);
        m_mousePosPrev = pos;
        myCam.rotatePhi(-diff[0]);
        myCam.rotateTheta(-diff[1]);
    } else if (e->buttons() & Qt::RightButton) {
        GfVec2d diff = 0.02f * (pos - m_mousePosPrev);
        m_mousePosPrev = pos;
        myCam.zoom(diff[1]);
    } else if (e->buttons() & Qt::MiddleButton) {
        // Panning
        GfVec2d diff = 0.02f * (pos - m_mousePosPrev);
        m_mousePosPrev = pos;
        myCam.translateAlongRight(-diff[0]);
        myCam.translateAlongUp(diff[1]);
    }
    myCam.recomputeAttributes();
    update();
}

void MyGL::wheelEvent(QWheelEvent* e)
{
    myCam.zoom(e->angleDelta().y() * 0.02f);
    myCam.recomputeAttributes();
    update();
}

void MyGL::slot_triggerRenderPreview(bool signaled)
{
    qDebug() << "Attempting save to file...";

    this->makeCurrent();

    this->doneCurrent();
}

void MyGL::initDefaultStage()
{

}
