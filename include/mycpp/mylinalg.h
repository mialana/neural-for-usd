#pragma once

#include "myglm.h"

#include <QRgb>

namespace linAlg
{
inline QRgb mapDirectionToRgb(const Vector3f& d)
{
    Color3f col;
    Color3i col255;
    QRgb myQRgb;

    col = d + Color3f(1.f);
    col *= 0.5f;

    col255 = col * 255.f;

    myQRgb = qRgb(col255.r, col255.g, col255.b);  // `qRgb()` is QT convenience function

    return myQRgb;
}

// handles manipulation of a 3D point for 4D transformation
inline Point3f doPoint3fXMat4(const glm::mat4& mat, const glm::vec3& pt)
{
    Point4f homogenized = glm::vec4(pt, 1);  // points must apply translation column
    Point4f transformed = mat * homogenized;

    Point3f result = transformed.xyz();  // exp(w) = 1
    return result;
}

// handles manipulation of a 3D vection for 4D transformation
inline Vector3f doVec3fXMat4(const glm::mat4& mat, const glm::vec3& vec)
{
    Vector4f homogenized = glm::vec4(vec, 0);  // vectors disregard translation column
    Vector4f transformed = mat * homogenized;

    Vector3f result = transformed.xyz();  // exp(w) = 0. Do not normalize to cover all use cases.
    return result;
}

}  // namespace linAlg
