#pragma once

#include "framemetadata.h"

#include <mycpp/mydefines.h>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdLux/domeLight.h>

#include <QString>

#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

class StageManager {
public:
    StageManager();
    ~StageManager();

    bool loadScene(const QString& stagePath, const QString& domeLightPath);
    bool generateCameraFrames(int numFrames);
    void exportDataJson() const;

    void setCurrentFrame(int frame);
    int getCurrentFrame() const;
    int getNumFrames() const;

    void setDomeLightIntensity(float intensity);
    float getDomeLightIntensity() const;

    void setCameraOrbitRadius(float radius);
    float getCameraOrbitRadius() const;

    const UsdStageRefPtr& getUsdStage() const;
    const UsdGeomCamera& getGeomCamera() const;
    GfCamera getGfCameraAtFrame(int frame) const;

    QString getOutputImagePath(int frame);
    QMap<QString, QString> getOutputPathMap() const;

    double getProgress() const;

private:
    void reset();
    bool configureUsdCamera();
    bool configureLuxDomeLight();
    void setUsdCameraTransformAtFrame(const GfMatrix4d& transform, int frame);

    static GfMatrix4d s_usdToNerfMatrix(const GfMatrix4d& cameraToWorld);

    UsdStageRefPtr m_usdStage;
    UsdGeomCamera m_geomCamera;
    UsdLuxDomeLight m_luxDomeLight;

    QString m_inputStagePath;
    QString m_inputDomeLightPath;

    QString m_outputStagePath;
    QString m_outputDataJsonPath;
    QString m_outputImageDir;
    QString m_outputImagePrefix;

    int m_numFrames = 0;
    int m_currentFrame = 0;
    double m_currProgress = 0.0;
    float m_cameraOrbitRadius = 1.0f;

    std::vector<uPtr<FrameMetadata>> m_allFrameMeta;
};
