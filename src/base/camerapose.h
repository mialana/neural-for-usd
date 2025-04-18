#pragma once

#include <QJsonObject>

#include <pxr/base/gf/matrix4d.h>

/**
 * @brief The CameraPose class manages data associated with each pose that is generated.
 */
struct CameraPose
{
    int m_identifier;      // the identifier number of the pose
    QString m_outputPath;  // file path the pose will ultimately be rendered to

    pxr::GfMatrix4d m_transform;

    CameraPose(int identifier, QString outputPath, pxr::GfMatrix4d transform);

    QJsonObject toJson() const;
};
