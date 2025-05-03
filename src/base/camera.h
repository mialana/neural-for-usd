#pragma once

#include "camerapose.h"

#include <mycpp/mydefines.h>

#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/camera.h>

#include <QProgressBar>
#include <QJsonObject>

class Camera
{
public:
    double m_currProgress;

private:
    QString m_stageFilePath;
    QString m_domeLightPath;
    QString m_outputStageFilePath;
    QString m_outputDataFilePath;
    QString m_outputRendersDirPath;
    QString m_outputPrefix;

    pxr::UsdGeomXform m_usdCameraXform;
    pxr::UsdGeomCamera m_usdCamera;
    pxr::GfCamera m_gfCamera;
    pxr::UsdStageRefPtr m_usdStage;

    std::vector<uPtr<CameraPose>> m_cameraPoses;

    int m_numFrames;

public:
    /**
     * @brief Parameters correlate to path-related members
     * @param sfp
     * @param hfp
     * @param odfp
     * @param ordp
     */
    Camera(QString stageFilePath, QString domeLightPath);

    void record();

    bool generateCameraPoses(int numSamples);

    void toJson() const;

    double getCurrProgress() const;

private:
    bool createGfCamera();
    bool createUsdCamera(const char* name);
    bool createDomeLight();

    /**
     * @brief Uses USD GfCamera route to set our USD Camera with preset params.
     * @param transform
     * @param frame
     */
    void setCameraTransformAtFrame(pxr::GfMatrix4d transform, int frame);
};
