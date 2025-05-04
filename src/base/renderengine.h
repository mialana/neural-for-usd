#pragma once

#include "openglcontext.h"
#include "stagemanager.h"

#include <pxr/usdImaging/usdImagingGL/engine.h>

enum RenderEngineMode {
    FIXED_CAMERA,
    FREE_CAMERA
};

class RenderEngine {
public:
    RenderEngine(OpenGLContext* context);
    ~RenderEngine();

    bool initDefaults();

    bool changeMode(RenderEngineMode mode);

    void render(StageManager* manager);

    void recordAllFixedFrames(StageManager* manager);

    void clearRender();

    void resize();

    void setComplexity(float complexity);
    void setColorCorrectionMode(TfToken mode);
    void setDomeLightVisibility(bool visibility);
    void setCameraLightEnabled(bool enabled);

private:
    OpenGLContext* m_context;

    UsdImagingGLEngine m_imagingEngine;

    RenderEngineMode m_mode;

    bool m_domeLightVisibility;
    bool m_cameraLightEnabled;

    UsdImagingGLRenderParams m_renderParams;

    GlfSimpleMaterial m_material;
};
