#pragma once

#include "camerapose.h"

#include <mycpp/mydefines.h>

#include <QProgressBar>
#include <QJsonObject>

#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/base/gf/camera.h>

class Camera
{
private:
    const QString m_stageFilePath;
    const QString m_hdriFilePath;
    const QString m_outputStageFilePath;
    const QString m_outputDataFilePath;
    QString m_outputRendersDirPath;

    QString m_outputPrefix;
    int m_numFrames;

    pxr::UsdGeomXform m_usdCameraXform;
    pxr::UsdGeomCamera m_usdCamera;
    pxr::GfCamera m_gfCamera;
    pxr::UsdStageRefPtr m_usdStage;

    std::vector<uPtr<CameraPose>> m_cameraPoses;

    bool createGfCamera();
    bool createUsdCamera(const char* name);
    bool createDomeLight();

    /**
     * @brief Uses USD GfCamera route to set our USD Camera with preset params.
     * @param transform
     * @param frame
     */
    void setCameraTransformAtFrame(pxr::GfMatrix4d transform, int frame);

public:
    /**
     * @brief Parameters correlate to path-related members
     * @param sfp
     * @param hfp
     * @param odfp
     * @param ordp
     */
    Camera(QString sfp, QString hfp, QString osfp, QString odfp, QString ordp);

    bool record(QString outputPrefix, QProgressBar* b, int numFrames);

    bool generateCameraPoses(int numSamples);

    void toJson() const;
};
