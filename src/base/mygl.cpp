#include "mygl.h"

#include <QKeyEvent>
#include <QApplication>

MyGL::MyGL(QWidget* parent)
    : OpenGLContext(parent)
    , m_timer()
    , m_mousePosPrev()
    , m_manager(mkU<StageManager>())
    , m_engine(mkU<RenderEngine>(this))
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

    if (m_engine->initDefaults()) {
        qDebug() << "Underlying USD imaging engine configured successfully.";
    }

    m_manager->initFreeCam(this->width(), this->height());

    m_timer.start(16);
}

void MyGL::resizeGL(int w, int h)
{
    m_manager->initFreeCam(w, h);

    m_engine->resize();

    qDebug() << "New width is:" << w;
    qDebug() << "New height is:" << h;
    qDebug() << "Device Pixel Ratio is:" << devicePixelRatio();
}

void MyGL::paintGL()
{
    m_engine->render(m_manager.get());
}

void MyGL::tick()
{
    this->update();
}

void MyGL::keyPressEvent(QKeyEvent* e)
{
    this->slot_changeRenderEngineMode("free");

    float amount = 2.0f;
    if (e->modifiers() & Qt::ShiftModifier) {
        amount = 10.0f;
    }

    switch (e->key()) {
        case (Qt::Key_Escape): QApplication::quit(); break;
        case (Qt::Key_Right): m_manager->m_freeCam->rotateAboutUp(-amount); break;
        case (Qt::Key_Left): m_manager->m_freeCam->rotateAboutUp(amount); break;
        case (Qt::Key_Up): m_manager->m_freeCam->rotateAboutRight(amount); break;
        case (Qt::Key_Down): m_manager->m_freeCam->rotateAboutRight(-amount); break;

        case (Qt::Key_W): m_manager->m_freeCam->zoom(amount); break;
        case (Qt::Key_S): m_manager->m_freeCam->zoom(-amount); break;
        case (Qt::Key_D): m_manager->m_freeCam->translateAlongRight(amount); break;
        case (Qt::Key_A): m_manager->m_freeCam->translateAlongRight(-amount); break;
        case (Qt::Key_Q): m_manager->m_freeCam->translateAlongUp(-amount); break;
        case (Qt::Key_E): m_manager->m_freeCam->translateAlongUp(amount); break;
    }
    update();
}

void MyGL::mousePressEvent(QMouseEvent* e)
{
    this->slot_changeRenderEngineMode("free");

    if (e->buttons() & (Qt::LeftButton | Qt::RightButton | Qt::MiddleButton)) {
        m_mousePosPrev = GfVec2d(e->pos().x(), e->pos().y());
    }
    update();
}

void MyGL::mouseMoveEvent(QMouseEvent* e)
{
    GfVec2d pos(e->pos().x(), e->pos().y());
    if (e->buttons() & Qt::LeftButton) {
        // Rotation
        GfVec2d diff = 0.04f * (pos - m_mousePosPrev);
        m_mousePosPrev = pos;
        m_manager->m_freeCam->orbitAboutOrigin(-diff[1], -diff[0]);
        // m_manager->m_freeCam->rotatePhi(-diff[0]);
        // m_manager->m_freeCam->rotateTheta(-diff[1]);
    } else if (e->buttons() & Qt::RightButton) {
        GfVec2d diff = 0.02f * (pos - m_mousePosPrev);
        m_mousePosPrev = pos;
        m_manager->m_freeCam->zoom(diff[1]);
    } else if (e->buttons() & Qt::MiddleButton) {
        // Panning
        GfVec2d diff = 0.02f * (pos - m_mousePosPrev);
        m_mousePosPrev = pos;
        m_manager->m_freeCam->translateAlongRight(-diff[0]);
        m_manager->m_freeCam->translateAlongUp(diff[1]);
    }
    update();
}

void MyGL::wheelEvent(QWheelEvent* e)
{
    this->slot_changeRenderEngineMode("free");

    m_manager->m_freeCam->zoom(e->angleDelta().y() * 0.02f);
    update();
}

void MyGL::slot_setStageManagerCurrentFrame(int frame)
{
    m_manager->setCurrentFrame(frame);
}

void MyGL::slot_changeRenderEngineMode(QString mode)
{
    if (mode == "fixed") {
        m_engine->changeMode(RenderEngineMode::FIXED_CAMERA);
    } else if (mode == "free") {
        bool changed = m_engine->changeMode(RenderEngineMode::FREE_CAMERA);

        if (changed) {
            // qDebug() m_manager->getGfCameraAtFrame(m_manager->getCurrentFrame()).GetFrustum().GetPosition();
            m_manager->m_freeCam->setFromGfCamera(
                m_manager->getGfCameraAtFrame(m_manager->getCurrentFrame()));
        }
    }
}

void MyGL::loadStageManager(const QString& stagePath, const QString& domeLightPath)
{
    bool success = m_manager->loadUsdStage(stagePath, domeLightPath);

    if (success) {
        qDebug() << "Usd stage loaded successfully by stage manager";
        m_manager->generateCameraFrames(106);

        m_manager->initFreeCam(this->width(), this->height());
    } else {
        qFatal() << "Stage manager was unable to load Usd stage";
    }
}
