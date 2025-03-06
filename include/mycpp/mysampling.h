#pragma once

#include "myglm.h"
#include "mymath.h"

namespace sampling
{
inline Point2f PolarToCartesian(const float& r, const float& theta)
{
    const float x = r * cos(theta);
    const float y = r * sin(theta);

    return Point2f(x, y);
}

inline glm::vec3 squareToDiskConcentric(const glm::vec2& sample)
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

inline glm::vec3 squareToHemisphereCosine(const glm::vec2& sample)
{
    const glm::vec3 xy0 = squareToDiskConcentric(sample);
    const float& x = xy0.x;
    const float& z = xy0.y;

    float y = sqrt(std::fmax(0.f, (1.f - pow(x, 2.f) - pow(z, 2.f))));  // use eq of unit sphere

    return glm::vec3(x, y, z) * 5.f;
}

inline glm::vec3 squareToHemisphereUniform(const glm::vec2& sample)
{
    const float y = sample.x;  // z range is positive

    float r = sqrt(std::fmax(0.f, 1.f - pow(y, 2.f)));

    const AngleRad phi = 2.f * M_PI * sample.y;

    const Point2f xz = PolarToCartesian(r, phi);

    return glm::vec3(xz.x, y, xz.y) * 4.f;
}
}  // namespace Sampling
