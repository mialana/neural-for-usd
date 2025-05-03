#pragma once

#include "stagemanager.h"
#include "freecamera.h"

class RenderEngine {
public:
    RenderEngine();
    ~RenderEngine();

    void render(StageManager& stage);

    void recordAllFrames(StageManager& stage);

    void enterFreeCameraMode();
    void enterFixedCameraMode();
    bool isUsingFreeCamera() const;

    void handleMouseInput(float dx, float dy, bool orbiting, bool panning);
    void handleScrollInput(float scrollDelta);

private:
    void renderWithCamera(StageManager& stage, const GfCamera& camera);
    void renderFreeCameraView(StageManager& stage);
    void renderFixedCameraFrame(StageManager& stage);

    bool m_useFreeCamera = false;
    FreeCamera m_freeCamera;
};
