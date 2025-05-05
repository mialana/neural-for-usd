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

    bool getIsDirty();
    void makeDirty();

    bool changeMode(RenderEngineMode mode);

    void render(StageManager* manager, bool shouldRecord);

    void record(StageManager* manager);

    void setComplexity(float complexity);
    void setColorCorrectionMode(TfToken mode);
    void setDomeLightVisibility(bool visibility);
    void setCameraLightEnabled(bool enabled);

private:
    void initDefaults();
    void clearRender();
    void resize();

    OpenGLContext* m_context;

    UsdImagingGLEngine m_imagingEngine;

    RenderEngineMode m_mode;

    bool m_domeLightVisibility;
    bool m_cameraLightEnabled;

    UsdImagingGLRenderParams m_renderParams;

    GlfSimpleMaterial m_material;

    bool isDirty;
};
