#include "camera.h"

#include <mycpp/mymath.h>
#include <mycpp/myglm.h>
#include <mycpp/mysampling.h>

#include <iostream>
#include <pcg32.h>

#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QFileInfo>
#include <QEventLoop>

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/usd/UsdRender/settingsBase.h>
#include <pxr/usdImaging/usdAppUtils/frameRecorder.h>

#include "myframerecorder.h"

Camera::Camera(QString stageFilePath, QString domeLightPath)
    : m_stageFilePath(stageFilePath)
    , m_domeLightPath(domeLightPath)
    , m_outputPrefix("/r")
    , m_numFrames(106)
    , m_currProgress(0.0)
{
    QString assetDir = QFileInfo(m_stageFilePath).dir().absolutePath();
    QString assetName = QFileInfo(m_stageFilePath).baseName();
    m_outputStageFilePath = assetDir + "/data/" + assetName + "Stage.usda";
    m_outputDataFilePath = assetDir + "/data/data.json";
    m_outputRendersDirPath = assetDir + "/data/internalVal";

    qDebug() << "Stage Path:" << m_stageFilePath.toStdString();

    // Create a USD stage
    m_usdStage = pxr::UsdStage::Open(CCP(m_stageFilePath));

    if (!m_usdStage) {
        std::cout << "Failed to open USD stage." << std::endl;
        return;
    }

    createUsdCamera("MyCam");
    createDomeLight();

    // pxr::UsdPrim render_prim = m_usdStage->DefinePrim(pxr::SdfPath("/Render"),
    //                                                   pxr::TfToken::Find("RenderSettings"));
    // pxr::UsdRenderSettingsBase settingsBase = pxr::UsdRenderSettingsBase(render_prim);
    // settingsBase.CreateResolutionAttr().Set(pxr::GfVec2i(200, 200));

    qDebug() << "Camera initiation status:" << m_usdStage->Export(CCP(m_outputStageFilePath));

    return;
}

bool Camera::createDomeLight()
{
    pxr::SdfAssetPath hdriFilePath = pxr::SdfAssetPath(CCP(m_domeLightPath));

    pxr::UsdLuxDomeLight hdri = pxr::UsdLuxDomeLight::Define(m_usdStage,
                                                             pxr::SdfPath("/lights/domeLight"));

    try {
        // TODO: Make camera scene separate?
        hdri.CreateTextureFileAttr().Set(hdriFilePath);
        hdri.CreateTextureFormatAttr().Set(pxr::UsdLuxTokens->latlong);

        hdri.GetExposureAttr().Set(1.f);
        // TODO: Add as possible attributes to set
        hdri.GetIntensityAttr().Set(0.25f);
        hdri.GetEnableColorTemperatureAttr().Set(true);
        hdri.GetColorTemperatureAttr().Set(4500.f);
        // hdri.GetColorAttr().Set(pxr::GfVec3f(0.5, 0.5, 0.5));
    } catch (std::exception e) {
        qDebug() << "Domelight creation error.";
        return false;
    }

    return true;
}

void Camera::record()
{
    if (!QDir().mkpath(m_outputRendersDirPath)) {
        qFatal() << "Output render path creation failure.";
    }

    generateCameraPoses(m_numFrames);

    pxr::MyFrameRecorder frameRecorder = pxr::MyFrameRecorder(pxr::TfToken(),
                                                              true);

    frameRecorder.SetColorCorrectionMode(pxr::TfToken::Find("sRGB"));
    frameRecorder.SetComplexity(1.0);
    frameRecorder.SetDomeLightVisibility(true);
    frameRecorder.SetImageWidth(1000);
    frameRecorder.SetCameraLightEnabled(true);

    for (int frame = 0; frame < m_numFrames; frame++) {
        QString outputImagePath = m_outputRendersDirPath + m_outputPrefix;
        outputImagePath += QString::number(frame);
        outputImagePath += ".png";

        m_cameraPoses[frame]->m_outputPath = outputImagePath;

        if (frameRecorder.Record(m_usdStage, m_usdCamera, frame, CCP(outputImagePath))) {
            qDebug() << "Recorded frame" << frame;

            m_currProgress = (float)frame / m_numFrames;
            qDebug() << "Curr Progress:" << m_currProgress;
        }
    }
}

/*
Nerf implementation currently uses row-major, z-up, and looking down the -z axis.
OpenUSD naturally uses column-major and y-up.
*/
pxr::GfMatrix4d changeMatrixFormat(const pxr::GfMatrix4d& oldMat)
{
    pxr::GfMatrix4d newMat = pxr::GfMatrix4d();
    newMat = oldMat.GetTranspose();

    pxr::GfRotation rotater = pxr::GfRotation(pxr::GfVec3d(1, 0, 0), -90.0);
    pxr::GfMatrix4d rotation = pxr::GfMatrix4d().SetRotateOnly(rotater);

    newMat = rotation * newMat;

    for (int i = 0; i < 3; i++)
        newMat[i][2] *= -1.0;

    return newMat;
}

bool Camera::generateCameraPoses(int numSamples)
{
    int sqrtVal = (int)(std::sqrt((float)m_numFrames) + 0.5);
    float invSqrtVal = 1.f / sqrtVal;

    // numSamples = sqrtVal * sqrtVal;

    pcg32 rng;

    for (int i = 0; i < m_numFrames; ++i) {
        int y = i / sqrtVal;
        int x = i % sqrtVal;
        glm::vec2 sample;

        glm::vec2 gridOrigin = glm::vec2(x, y) * invSqrtVal;
        glm::vec2 offset;

        offset = glm::vec2(invSqrtVal / 2.f);
        sample = gridOrigin + offset;

        glm::vec3 warpResult = sampling::squareToHemisphereUniform(sample);

        glm::vec3 target = glm::vec3(0.f);
        glm::vec3 look = glm::normalize(target - warpResult);
        glm::vec3 right = glm::normalize(glm::cross(look, glm::vec3(0, 1, 0)));
        if (glm::length(right) < 1e-6f) {
            // look direction was too close to y-axis. use z-axis as pseudo up.
            right = glm::normalize(glm::cross(look, glm::vec3(0.0f, 0.0f, 1.0f)));
        }
        glm::vec3 up = glm::cross(right, look);

        pxr::GfMatrix4d m = pxr::GfMatrix4d().SetLookAt(pxr::GfVec3d(warpResult.x,
                                                                     warpResult.y,
                                                                     warpResult.z),
                                                        pxr::GfVec3d(0.f),
                                                        pxr::GfVec3d(up.x, up.y, up.z));
        m = m.GetInverse();

        setCameraTransformAtFrame(m, i);

        uPtr<CameraPose> currCamPose = mkU<CameraPose>(i, QString("Not set"), changeMatrixFormat(m));
        m_cameraPoses.push_back(std::move(currCamPose));
    }

    m_usdStage->Export(CCP(m_outputStageFilePath));

    return true;
}

void Camera::setCameraTransformAtFrame(pxr::GfMatrix4d transform, int frame)
{
    // SetTransform expects a CameraToWorldTransformation
    m_gfCamera.SetTransform(transform);
    m_usdCamera.SetFromCamera(m_gfCamera, frame);

    return;
}

bool Camera::createGfCamera()
{
    m_gfCamera = pxr::GfCamera(m_usdCamera.GetCamera(0));

    return true;
}

bool Camera::createUsdCamera(const char* name)
{
    const pxr::SdfPath& cameraXformPath = pxr::SdfPath("/Xform_MyCam");
    m_usdCameraXform = pxr::UsdGeomXform::Define(m_usdStage, cameraXformPath);

    const pxr::SdfPath& cameraPath = pxr::SdfPath("/Xform_MyCam/MyCam");
    m_usdCamera = pxr::UsdGeomCamera::Define(m_usdStage, cameraPath);

    m_usdCamera.CreateProjectionAttr().Set(pxr::UsdGeomTokens->perspective);
    m_usdCamera.CreateHorizontalApertureAttr().Set(25.955f);
    m_usdCamera.CreateVerticalApertureAttr().Set(25.955f);
    m_usdCamera.CreateFocalLengthAttr().Set(70.38f);
    m_usdCamera.CreateFocusDistanceAttr().Set(55.f);
    m_usdCamera.CreateClippingRangeAttr().Set(pxr::GfVec2f(1.f, 125.f));

    createGfCamera();

    return true;
}

void Camera::toJson() const
{
    if (!QDir(m_outputRendersDirPath).exists() || m_cameraPoses.empty()) {
        qFatal() << "Camera data has not been recorded yet.";
    }

    QJsonObject json;
    QJsonArray framesArray;

    for (int frame = 0; frame < m_numFrames; frame++) {
        framesArray.append(m_cameraPoses[frame]->toJson());
        qDebug() << "Wrote pose" << frame;
    }

    json["frames"] = framesArray;

    float aperture = 0.f;
    float focal = 0.f;
    m_usdCamera.GetHorizontalApertureAttr().Get(&aperture);
    m_usdCamera.GetFocalLengthAttr().Get(&focal);

    float camera_angle_x = 2.f * atanf(aperture / (2.f * focal));

    json["camera_angle_x"] = camera_angle_x;

    QJsonDocument document;
    document.setObject(json);

    QFile jsonFile(m_outputDataFilePath);
    jsonFile.open(QFile::WriteOnly);
    jsonFile.write(document.toJson());

    jsonFile.close();

    return;
}

double Camera::getCurrProgress() const
{
    return m_currProgress;
}
