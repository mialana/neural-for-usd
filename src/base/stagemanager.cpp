#include "stagemanager.h"

#include <QFileInfo>
#include <QDir>

StageManager::StageManager() {}

StageManager::~StageManager() {}

void StageManager::reset() {
    m_allFrameMeta.clear();
    m_usdStage = nullptr;
    m_currProgress = 0.0;
    m_currentFrame = 0;
    m_numFrames = 0;
}
