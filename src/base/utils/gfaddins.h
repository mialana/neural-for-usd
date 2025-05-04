#include <mycpp/mydefines.h>

#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec3d.h>

#include <QDebug>
#include <sstream>

PXR_NAMESPACE_OPEN_SCOPE

inline GfVec3d GfSquareToHemisphereUniform(const GfVec2d& sample)
{
    double y = sample[0];  // y is vertical axis (elevation)

    double r = std::sqrt(std::max(0.0, 1.0 - y * y));
    double phi = 2.0 * M_PI * sample[1];

    double x = r * std::cos(phi);
    double z = r * std::sin(phi);

    return GfVec3d(x, y, z) * 4.0;
}

template<typename T>
inline void GfDebug(QString label, const T& val)
{
    std::ostringstream oss;
    oss << val;
    qDebug().Nq().Ns() << label << ": " << QString::fromStdString(oss.str());
}

template<typename T>
inline QDebug operator<<(QDebug debug, const T& val)
{
    std::ostringstream oss;
    oss << val; // relies on std::ostream << overload
    return debug.noquote().nospace() << QString::fromStdString(oss.str());
}

PXR_NAMESPACE_CLOSE_SCOPE
