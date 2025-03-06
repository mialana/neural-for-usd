#include "camera.h"

#include <MyCpp/mydefines.h>
#include <MyCpp/mymath.h>
#include <MyCpp/myglm.h>
#include <MyCpp/mysampling.h>

#include <iostream>
#include <pcg32.h>

#include <QDir>

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usdImaging/usdAppUtils/frameRecorder.h>

Camera::Camera(QString sfp, QString hfp, QString osfp, QString odfp, QString ordp)
    : m_stageFilePath(sfp)
    , m_hdriFilePath(hfp)
    , m_outputStageFilePath(osfp)
    , m_outputDataFilePath(odfp)
    , m_outputRendersDirPath(ordp)
{
    qDebug() << "Stage Path:" << m_stageFilePath.toLocal8Bit();

    // Create a USD stage
    m_usdStage = pxr::UsdStage::Open(CCP(m_stageFilePath));

    if (!m_usdStage) {
        std::cout << "Failed to open USD stage." << std::endl;
        return;
    }

    createUsdCamera("MyCam");
    createDomeLight();

    qDebug() << "Camera initiation status:" << m_usdStage->Export(CCP(m_outputStageFilePath));

    return;
}

bool Camera::createDomeLight()
{
    pxr::SdfAssetPath hdriFilePath = pxr::SdfAssetPath(CCP(m_hdriFilePath));

    pxr::UsdLuxDomeLight hdri = pxr::UsdLuxDomeLight::Define(m_usdStage,
                                                             pxr::SdfPath("/lights/domeLight"));

    try {
        // TODO: Make camera scene separate?
        hdri.CreateTextureFileAttr().Set(hdriFilePath);
        hdri.CreateTextureFormatAttr().Set(pxr::UsdLuxTokens->latlong);

        hdri.GetExposureAttr().Set(1.f);
        // TODO: Add as possible attributes to set
        // hdri.GetIntensityAttr().Set(0.5f);
        // hdri.GetEnableColorTemperatureAttr().Set(true);
        // hdri.GetColorTemperatureAttr().Set(9000.f);
        // hdri.GetColorAttr().Set(pxr::GfVec3f(0.5, 0.5, 0.5));
    } catch (std::exception e) {
        qDebug() << "Domelight creation error.";
        return false;
    }

    return true;
}

bool Camera::writeData()
{
    return false;
}

bool Camera::record(QString outputPrefix, QProgressBar* b)
{
    if (!QDir().mkpath(m_outputRendersDirPath)) {
        qDebug() << "Output render path creation failure.";
        return false;
    }

    pxr::UsdAppUtilsFrameRecorder frameRecorder = pxr::UsdAppUtilsFrameRecorder(pxr::TfToken(),
                                                                                true);

    frameRecorder.SetColorCorrectionMode(pxr::TfToken::Find("sRGB"));
    frameRecorder.SetComplexity(4.0);
    frameRecorder.SetDomeLightVisibility(true);

    for (int frame = 0; frame < 100; frame++) {
        QString outputImagePath = m_outputRendersDirPath + outputPrefix;
        outputImagePath += QString::number(frame);
        outputImagePath += ".png";

        if (frameRecorder.Record(m_usdStage, m_usdCamera, frame, CCP(outputImagePath))) {
            b->setValue(frame + 1);
        }
    }
    return true;
}

bool Camera::generateCameraTransforms(int numSamples)
{
    int sqrtVal = (int)(std::sqrt((float)numSamples) + 0.5);
    float invSqrtVal = 1.f / sqrtVal;

    numSamples = sqrtVal * sqrtVal;

    pcg32 rng;

    for (int i = 0; i < numSamples; ++i) {
        int y = i / sqrtVal;
        int x = i % sqrtVal;
        glm::vec2 sample;

        glm::vec2 gridOrigin = glm::vec2(x, y) * invSqrtVal;
        glm::vec2 offset;

        offset = glm::vec2(invSqrtVal / 2.f);
        sample = gridOrigin + offset;

        // sample = glm::vec2(rng.nextFloat(), rng.nextFloat());

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

        m_usdCameraParams.SetTransform(m);
        m_usdCamera.SetFromCamera(m_usdCameraParams, i);
    }
    return true;
}

bool Camera::createUsdCameraParams()
{
    m_usdCameraParams = pxr::GfCamera(m_usdCamera.GetCamera(0));

    return true;
}

bool Camera::createUsdCamera(const char* name)
{
    const pxr::SdfPath& cameraXformPath = pxr::SdfPath("/Xform_MyCam");
    m_usdCameraXform = pxr::UsdGeomXform::Define(m_usdStage, cameraXformPath);

    const pxr::SdfPath& cameraPath = pxr::SdfPath("/Xform_MyCam/MyCam");
    m_usdCamera = pxr::UsdGeomCamera::Define(m_usdStage, cameraPath);
    m_usdCamera.CreateProjectionAttr().Set(pxr::UsdGeomTokens->perspective);

    if (createUsdCameraParams()) {
        return generateCameraTransforms(100);
    }

    return false;
}
