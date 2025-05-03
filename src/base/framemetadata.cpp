#include "framemetadata.h"

#include <QJsonArray>
#include <QFileInfo>
#include <QDir>

FrameMetadata::FrameMetadata(int identifier, QString outputPath, pxr::GfMatrix4d transform)
    : m_identifier(identifier)
    , m_outputImagePath(outputPath)
    , m_transform(transform)
{}

QJsonObject FrameMetadata::toJson(const QString& dataJsonPath) const
{
    QJsonObject json;

    // Compute relative path from data.json's parent dir
    QString dataDir = QFileInfo(dataJsonPath).absolutePath();
    QString relativePath = QDir(dataDir).relativeFilePath(m_outputImagePath);
    json["file_path"] = relativePath;

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
