vec2 PolarToCartesian(float r, float theta)
{
    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec2(x, y);
}

vec3 squareToDiskConcentric(vec2 xi)
{
    vec2 offsetSample = 2.f * xi - vec2(1.f);  // offset to (-1.f, 1.f) range

    float r, theta;

    if (offsetSample.x == 0.f && offsetSample.y == 0.f) {
        return vec3(0.f);
    }

    if (abs(offsetSample.x) > abs(offsetSample.y)) {
        r = offsetSample.x;
        theta = (PI / 4.f) * (offsetSample.y / offsetSample.x);
    } else {
        r = offsetSample.y;
        theta = (PI / 2.f) - ((PI / 4.f) * (offsetSample.x / offsetSample.y));
    }

    vec2 xy = PolarToCartesian(r, theta);

    return vec3(xy, 0.f);
}

// sample
vec3 squareToHemisphereCosine(vec2 xi)
{
    vec3 xy0 = squareToDiskConcentric(xi);
    float x = xy0.x;
    float y = xy0.y;

    float z = sqrt(max(0.f, 1.f - pow(x, 2.f) - pow(y, 2.f)));

    return vec3(x, y, z);
}

float squareToHemisphereCosinePDF(vec3 w)
{
    return CosTheta(w) * INV_PI;
}

vec3 squareToSphereUniform(vec2 xi)
{
    float z = 1.f - (2.f * xi.x);  // map [0, 1] to [-1, 1]

    float r = sqrt(max(0.f, 1.f - pow(z, 2.f)));

    float phi = TWO_PI * xi.y;

    vec2 xy = PolarToCartesian(r, phi);

    return vec3(xy, z);
}

float squareToSphereUniformPDF(vec3 w)
{
    return INV_FOUR_PI;
}
