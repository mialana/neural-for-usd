#pragma once

#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/matrix4d.h>

PXR_NAMESPACE_USING_DIRECTIVE

    class FreeCamera {
public:
    FreeCamera();

    void reset();
    void syncFromCamera(const GfCamera& cam);

    // Orbit: rotates around pivot point
    void orbit(float deltaTheta, float deltaPhi);

    // Pan: translates pivot point in view plane
    void pan(float dx, float dy);

    // Zoom: adjusts distance to pivot
    void zoom(float delta);

    void setFromGfCamera(const GfCamera& camera);

    GfCamera getGfCamera() const;

private:
    void updateCamera();

    GfCamera m_camera;

    GfVec3d m_pivot = GfVec3d(0, 0, 0);  // Target of camera
    float m_radius = 1.0f;               // Distance from camera to pivot
    float m_theta = 0.0f;                // Azimuth (horizontal orbit)
    float m_phi = 0.0f;                  // Elevation (vertical orbit)
};

