#include "camera.h"
#include "math_utils.h"
#include "glm_utils.h"

#include <iostream>
#include <pcg32.h>

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/usdImaging/usdviewq/utils.h>

Point2f PolarToCartesian(const float& r, const float& theta)
{
    const float x = r * cos(theta);
    const float y = r * sin(theta);

    return Point2f(x, y);
}

glm::vec3 squareToDiskConcentric(const glm::vec2& sample)
{
    const Point2f offsetSample = 2.f * sample - glm::vec2(1.f);

    float r;
    AngleRad theta;

    // handle undef behavior at origin
    if (fequal(offsetSample.x, 0.f) && fequal(offsetSample.y, 0.f)) {
        return glm::vec3(0.f);
    }

    if (std::abs(offsetSample.x) > std::abs(offsetSample.y)) {  // case 1
        r = offsetSample.x;
        theta = (M_PI / 4.f) * (offsetSample.y / offsetSample.x);
    } else {  // case 2 ("inverse" case)
        r = offsetSample.y;
        theta = (M_PI / 2.f) - ((M_PI / 4.f) * (offsetSample.x / offsetSample.y));
    }

    const Point2f xy = PolarToCartesian(r, theta);

    return glm::vec3(xy, 0.f);
}

glm::vec3 squareToHemisphereCosine(const glm::vec2& sample)
{
    const glm::vec3 xy0 = squareToDiskConcentric(sample);
    const float& x = xy0.x;
    const float& z = xy0.y;

    float y = sqrt(std::fmax(0.f, (1.f - pow(x, 2.f) - pow(z, 2.f))));  // use eq of unit sphere

    return glm::vec3(x, y, z) * 5.f;
}

void Camera::generateCameraTransforms(int numSamples)
{
    // m_usdCameraXform.AddTranslateOp();

    // The square root of the number of samples input
    int sqrtVal = (int)(std::sqrt((float)numSamples) + 0.5);
    // A number useful for scaling a square of size sqrtVal x sqrtVal to 1 x 1
    float invSqrtVal = 1.f / sqrtVal;

    numSamples = sqrtVal * sqrtVal;
    // samples.resize(numSamples * 3);       // 3 floats for position
    // sampleColors.resize(numSamples * 3);  // 3 floats for color
    // float colorScale = 1.f / numSamples;  // For creating a gradient

    pcg32 rng;  // We'll be using the PCG random number generator class to generate samples.
    // You can read more about this random number generator at http://www.pcg-random.org/
    // PBRT ed. 3 also discusses the PCG RNG from pages 1065 - 1066

    for (int i = 1; i < numSamples; ++i) {
        int y = i / sqrtVal;
        int x = i % sqrtVal;
        glm::vec2 sample;                                     // position of sample

        glm::vec2 gridOrigin = glm::vec2(x, y) * invSqrtVal;  // scale to 1 x 1 square
        glm::vec2 offset;

        offset = glm::vec2(invSqrtVal / 2.f);  // offset by half of size of a grid cell
        sample = gridOrigin + offset;

        sample = glm::vec2(rng.nextFloat(), rng.nextFloat());

        glm::vec3 warpResult = squareToHemisphereCosine(sample);

        glm::vec3 target = glm::vec3(0.f);
        glm::vec3 look = glm::normalize(target - warpResult);
        glm::vec3 right = glm::normalize(glm::cross(look, glm::vec3(0, 1, 0)));
        glm::vec3 up = glm::cross(right, look);

        pxr::GfMatrix4d m = pxr::GfMatrix4d().SetLookAt(pxr::GfVec3d(warpResult.x,
                                                                     warpResult.y,
                                                                     warpResult.z),
                                                        pxr::GfVec3d(0.f),
                                                        pxr::GfVec3d(up.x, up.y, up.z));
        pxr::GfMatrix4d t = pxr::GfMatrix4d().SetTranslate(pxr::GfVec3d(4, 0, 3));

        m = m.GetInverse();

        m_usdCameraParams.SetTransform(m);
        m_usdCamera.SetFromCamera(m_usdCameraParams, i);
    }
}

void Camera::createUsdCameraParams()
{
    pxr::GfMatrix4d m = pxr::GfMatrix4d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1);
    // pxr::GfVec3d v = pxr::GfVec3d(0, 0, 5);
    // m_usdCameraXform.AddTranslateOp().Set(v);

    m_usdCameraParams = pxr::GfCamera(m_usdCamera.GetCamera(0));
    m_usdCameraParams.SetTransform(m);

    m_usdCamera.SetFromCamera(m_usdCameraParams, 0);

    generateCameraTransforms(100);

    return;
}

void Camera::createUsdCamera(const pxr::UsdStagePtr& stage, const char* name)
{
    const pxr::SdfPath& cameraXformPath = pxr::SdfPath("/Xform_MyCam");
    m_usdCameraXform = pxr::UsdGeomXform::Define(stage, cameraXformPath);

    const pxr::SdfPath& cameraPath = pxr::SdfPath("/Xform_MyCam/MyCam");
    m_usdCamera = pxr::UsdGeomCamera::Define(stage, cameraPath);
    m_usdCamera.CreateProjectionAttr().Set(pxr::UsdGeomTokens->perspective);

    createUsdCameraParams();

    return;
}

Camera::Camera()
{
    // Path to the USD file
    std::string filePath
        = "/Users/liu.amy05/Documents/Neural-for-USD/assets/japanesePlaneToy/japanesePlaneToy.usda";

    std::cout << filePath << std::endl;

    // Create a USD stage
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filePath);

    if (!stage) {
        std::cout << "Failed to open USD stage." << std::endl;
        return;
    }

    pxr::UsdGeomXform p = pxr::UsdGeomXform::Get(stage, pxr::SdfPath("/japanese_toy"));
    // p.ClearXformOpOrder();
    p.AddRotateXOp().Set(-90.f);
    p.AddTranslateYOp().Set(-0.1);

    createUsdCamera(stage, "MyCam");

    stage->Export("/Users/liu.amy05/Documents/Neural-for-USD/japanesePlaneToy.usda");

    std::cout << "Done" << std::endl;

    return;
}
