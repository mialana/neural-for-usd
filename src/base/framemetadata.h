#pragma once

#include <QJsonObject>

#include <pxr/base/gf/matrix4d.h>

/**
 * @brief The FrameMetadata struct manages data associated with a generated camera frame.
 */
struct FrameMetadata
{
    int m_identifier;      // the identifier number of the frame
    QString m_outputImagePath;  // absolute file path that the frame will ultimately be rendered to

    pxr::GfMatrix4d m_transform;

    FrameMetadata(int identifier, QString outputPath, pxr::GfMatrix4d transform);

    QJsonObject toJson(const QString& dataJsonPath) const;
};
