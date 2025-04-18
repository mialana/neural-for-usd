#include "camerapose.h"

#include <QJsonArray>

CameraPose::CameraPose(int identifier, QString outputPath, pxr::GfMatrix4d transform)
    : m_identifier(identifier)
    , m_outputPath(outputPath)
    , m_transform(transform)
{}

QJsonObject CameraPose::toJson() const
{
    QJsonObject json;

    json["file_path"] = m_outputPath;

    QJsonArray transformArray;

    for (int i = 0; i < 4; i++) {
        QJsonArray row;
        for (int j = 0; j < 4; j++) {
            row.append(m_transform[i][j]);
        }
        transformArray.append(row);
    }

    json["transformMatrix"] = transformArray;

    return json;
}
