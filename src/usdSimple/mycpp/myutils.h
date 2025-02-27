#pragma once

#include "glmutils.h"

#include <QTextStream>
#include <QString>
#include <QRgb>

enum Code {
    FG_BLACK = 30,
    FG_RED = 31,
    FG_GREEN = 32,
    FG_YELLOW = 33,
    FG_BLUE = 34,
    FG_CYAN = 36,

    FG_ORANGE = 208,  // not available on some systems
};

namespace console
{
void setColor(QTextStream& stream, Code code);  // set color of stream
void printLineBreak(QTextStream& stream);       // prints a line of '*' to break up logging

/**
 *  @brief Formats all console logging that happens in a function into a section.
 *  Best practices: Templates are declared in header.
 *  
 */
template<typename Ret, typename Fn, typename... Args>
Ret createFunctionSection(const char* title, Fn fn, Args... args)
{
    QTextStream qCout(stdout);
    QString qEndl("\n");

    console::printLineBreak(qCout);

    qCout << title << qEndl;
    qCout.flush();

    if constexpr (std::is_same_v<void, std::invoke_result_t<Fn, Args...>>) {
        fn(args...);
        console::printLineBreak(qCout);
    } else {
        Ret r = fn(args...);
        console::printLineBreak(qCout);
        return r;
    }
}
}  // namespace console

namespace startup
{
/**
 *  @brief Sets up custom logging and handles all OpenGL-related setup
 *  
 *  Call at top of application's `main(..)` function. Delete extraneous function calls 
 *  besides setup of the `QApplication` and the `MainWindow`.
 */
void doSimpleSetup();

// uses `qInstallMessageHandler()` to change the formatting of console logs created by `QDebug`
void setUpCustomLogging();

// handles all OpenGL-related setup
void setOpenGLFormat();

/* Private functions. Included in header for personal reference */

QString _getVersionString(std::pair<int, int> version);  // get OpenGL version as a `QString`
QString _getProfileString(int prof);                     // get OpenGL profile as a `QString`

QString _getTimestamp();                                 // get current timestamp as a `QString`

// `QDebug` message handler with required function signature
void _customizeQDebugHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg);
}  // namespace startup

namespace linAlg
{
QRgb mapDirectionToRgb(const Vector3f& d);

// handles manipulation of a 3D point for 4D transformation
Point3f doPoint3fXMat4(const glm::mat4& mat, const glm::vec3& pt);

// handles manipulation of a 3D vection for 4D transformation
Vector3f doVec3fXMat4(const glm::mat4& mat, const glm::vec3& vec);

}  // namespace linAlg
