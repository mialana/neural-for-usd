#include "stagemanager.h"

#include "utils/gfaddins.h"

#include <pxr/base/gf/rotation.h>

#include <QFileInfo>
#include <QDir>
#include <QJsonArray>

#include <pcg32.h>

StageManager::StageManager() {}

StageManager::~StageManager() {}

void StageManager::reset()
{
    m_allFrameMeta.clear();
    m_usdStage = nullptr;
    m_currProgress = 0.0;
    m_currentFrame = 0;
    m_numFrames = 0;
}

UsdPrim* StageManager::getPsuedoRoot()
{
    if (!m_usdStage || !m_pseudoRoot) {
        qWarning() << "Tried to get Usd Stage pseudo root before/during stage load.";
    } else if (!m_pseudoRoot) {
        qWarning() << "Tried to get Usd Stage pseudo root while it was null.";
    }
    return m_pseudoRoot.get();
}

void StageManager::setCurrentFrame(int frame)
{
    if (frame < 0 || frame >= m_numFrames) {
        qFatal() << "Invalid frame requested";
    }
    m_currentFrame = frame;

    return;
}

int StageManager::getCurrentFrame() const
{
    return m_currentFrame;
}

int StageManager::getNumFrames() const
{
    return m_numFrames;
}

bool StageManager::loadUsdStage(const QString& stagePath, const QString& domeLightPath)
{
    m_inputStagePath = stagePath;
    m_inputDomeLightPath = domeLightPath;

    QFileInfo info(stagePath);
    QString assetDir = info.dir().absolutePath();
    QString assetName = info.baseName();

    m_outputStagePath = assetDir + "/data/" + assetName + "Stage.usda";
    m_outputDataJsonPath = assetDir + "/data/data.json";
    m_outputImageDir = assetDir + "/data/internalVal";
    m_outputImagePrefix = "r";

    m_usdStage = UsdStage::Open(stagePath.toStdString());
    if (!m_usdStage) {
        qWarning() << "Failed to open stage:" << stagePath;
        return false;
    }

    m_pseudoRoot = mkU<UsdPrim>(m_usdStage->GetPseudoRoot());

    return configureUsdCamera() && configureLuxDomeLight();
}

bool StageManager::initFreeCam(int width, int height) {
    if (!m_freeCam) {
        qDebug() << "Free camera created from scratch.";
        GfCamera currGfCamera = this->getGfCameraAtFrame(getCurrentFrame());
        m_freeCam = mkU<FreeCamera>(width, height, currGfCamera.GetFrustum());
    } else {
        qDebug() << "Free camera created from existing parameters.";
        m_freeCam = mkU<FreeCamera>(width, height, *m_freeCam.get());
    }
    return true;
}

bool StageManager::configureUsdCamera()
{
    SdfPath camPath("/cameras/fixedCam");

    m_geomCamera = UsdGeomCamera::Define(m_usdStage, camPath);

    m_geomCamera.CreateProjectionAttr().Set(UsdGeomTokens->perspective);
    m_geomCamera.CreateHorizontalApertureAttr().Set(25.955f);
    m_geomCamera.CreateVerticalApertureAttr().Set(25.955f);
    m_geomCamera.CreateFocalLengthAttr().Set(70.38f);
    m_geomCamera.CreateFocusDistanceAttr().Set(55.f);
    m_geomCamera.CreateClippingRangeAttr().Set(GfVec2f(1.f, 125.f));

    return true;
}

bool StageManager::configureLuxDomeLight()
{
    SdfPath lightPath("/lights/domeLight");
    m_luxDomeLight = UsdLuxDomeLight::Define(m_usdStage, lightPath);

    m_luxDomeLight.CreateTextureFileAttr().Set(SdfAssetPath(m_inputDomeLightPath.toStdString()));
    m_luxDomeLight.CreateTextureFormatAttr().Set(UsdLuxTokens->latlong);
    m_luxDomeLight.GetExposureAttr().Set(1.f);
    m_luxDomeLight.GetIntensityAttr().Set(0.25f);  // default intensity
    m_luxDomeLight.GetEnableColorTemperatureAttr().Set(true);
    m_luxDomeLight.GetColorTemperatureAttr().Set(4500.f);

    return true;
}

void StageManager::setUsdCameraTransformAtFrame(const GfMatrix4d& transform, int frame)
{
    GfCamera tempGfCamera;
    tempGfCamera.SetTransform(transform);
    m_geomCamera.SetFromCamera(tempGfCamera, frame);
}

bool StageManager::generateCameraFrames(int numFrames)
{
    m_numFrames = numFrames;
    m_allFrameMeta.clear();

    int sqrtVal = (int)(std::sqrt((float)numFrames) + 0.5);
    float invSqrt = 1.f / sqrtVal;

    pcg32 rng;

    for (int i = 0; i < numFrames; ++i) {
        int y = i / sqrtVal;
        int x = i % sqrtVal;

        GfVec2d sample = GfVec2d(x * invSqrt, y * invSqrt);
        sample += GfVec2d(invSqrt / 2.0, invSqrt / 2.0);  // center-in-cell jitter

        GfVec3d pos = GfSquareToHemisphereUniform(sample) * m_cameraOrbitRadius;
        GfVec3d origin = GfVec3d(0.0);

        GfVec3d look = (origin - pos).GetNormalized();  // toward origin
        GfVec3d up(0, 1, 0);
        GfVec3d right = GfCross(look, up).GetNormalized();

        if (right.GetLength() < 1e-6) {
            right = GfCross(look, GfVec3d(0, 0, 1)).GetNormalized();
        }

        up = GfCross(right, look);

        // Build a camera-to-world transform (inverse of LookAt)
        GfMatrix4d worldToCam = GfMatrix4d().SetLookAt(pos, origin, up);
        GfMatrix4d camToWorld = worldToCam.GetInverse();

        setUsdCameraTransformAtFrame(camToWorld, i);

        QString imgPath = getOutputImagePath(i);
        auto frame = mkU<FrameMetadata>(i, imgPath, s_usdToNerfMatrix(camToWorld));
        m_allFrameMeta.push_back(std::move(frame));
    }

    m_usdStage->Export(m_outputStagePath.toStdString());
    return true;
}

QString StageManager::getOutputImagePath(int frame)
{
    return m_outputImageDir + "/" + m_outputImagePrefix
           + QString("r%1.png").arg(frame, 3, 10, QChar('0'));
}

QMap<QString, QString> StageManager::getOutputPathMap() const
{
    return {{"Output Stage", m_outputStagePath},
            {"Output Data Json", m_outputDataJsonPath},
            {"Output Image Directory", m_outputImageDir}};
}

GfCamera StageManager::getGfCameraAtFrame(int frame) const
{
    return GfCamera(m_geomCamera.GetCamera(frame));
}

void StageManager::exportDataJson() const
{
    QJsonObject json;
    QJsonArray frames;

    for (const auto& frameMeta : m_allFrameMeta) {
        frames.append(frameMeta->toJson(m_outputDataJsonPath));
    }

    float aperture, focal;
    m_geomCamera.GetHorizontalApertureAttr().Get(&aperture);
    m_geomCamera.GetFocalLengthAttr().Get(&focal);

    json["frames"] = frames;
    json["camera_angle_x"] = 2.f * atanf(aperture / (2.f * focal));

    QFile file(m_outputDataJsonPath);
    if (file.open(QFile::WriteOnly)) {
        file.write(QJsonDocument(json).toJson());
    }
}

GfMatrix4d StageManager::s_usdToNerfMatrix(const GfMatrix4d& cameraToWorld)
{
    GfMatrix4d newMat = GfMatrix4d();
    newMat = cameraToWorld.GetTranspose();

    pxr::GfRotation rotater = pxr::GfRotation(pxr::GfVec3d(1, 0, 0), -90.0);
    pxr::GfMatrix4d rotation = pxr::GfMatrix4d().SetRotateOnly(rotater);

    newMat = rotation * newMat;

    for (int i = 0; i < 3; i++)
        newMat[i][2] *= -1.0;

    return newMat;
}
