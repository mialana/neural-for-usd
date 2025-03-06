#pragma once

#include "myconsole.h"
#include "mydefines.h"

#include <QDateTime>
#include <QString>
#include <QSurfaceFormat>

namespace startup
{
/**
 * @brief get current timestamp as a `QString`
 * @return
 */
inline QString _getTimestamp()
{
    QString result = "[";

    QDateTime dateTime = QDateTime::currentDateTime();
    QString dtString = dateTime.toString(Qt::DateFormat ::ISODateWithMs);
    dtString = dtString.replace('T', ' ');
    QString timeZone = dateTime.timeZoneAbbreviation();

    result += dtString + " " + timeZone + "] ";
    return result;
};

/**
 * @brief get OpenGL version as a `QString`
 * @param version
 * @return 
 */
inline QString _getVersionString(std::pair<int, int> version)
{
    QString result;
    QTextStream stream(&result);
    stream << version.first << "." << version.second;
    return result;
}

/**
 * @brief get OpenGL profile as a `QString`
 * @param prof
 * @return
 */
inline QString _getProfileString(int prof)
{
    return prof == QSurfaceFormat::CoreProfile            ? "Core"
           : prof == QSurfaceFormat::CompatibilityProfile ? "Compatibility"
                                                          : "None";
}

/**
 * @brief `QDebug` message handler with required function signature
 * @param type
 * @param context
 * @param msg
 */
inline void _customizeQDebugHandler(QtMsgType type,
                                    const QMessageLogContext& context,
                                    const QString& msg)
{
    QString out;
    QTextStream qOut(&out);
    QString qEndl("\n");

    QString timestamp = startup::_getTimestamp();  // current time and date info
    QString inputMsg = msg;                        // attached message

    QString location = " [";                       // file and line info
    QTextStream locStream(&location);              // streams are easier to append different types
    locStream << context.file << ":" << context.line << "]";

    /* Begin append to standard out */
    console::setColor(qOut, console::FG_BLACK);

    qOut << timestamp;

    // append type of message in custom color
    switch (type) {
        case QtInfoMsg:
            console::setColor(qOut, console::FG_GREEN);
            qOut << "[INFO] ";
            break;
        case QtDebugMsg:
            console::setColor(qOut, console::FG_CYAN);
            qOut << "[DEBUG] ";
            break;
        case QtWarningMsg:
            console::setColor(qOut, console::FG_YELLOW);
            qOut << "[WARN] ";
            break;
        case QtCriticalMsg:
            console::setColor(qOut, console::FG_ORANGE);
            qOut << "[CRITICAL] ";
            break;
        case QtFatalMsg:
            console::setColor(qOut, console::FG_RED);
            qOut << "[FATAL] ";
            break;
    }

    console::setColor(qOut, console::FG_BLACK);
    qOut << inputMsg;

    // append location if not an info type
    if (type != QtInfoMsg && context.file) {
        console::setColor(qOut, console::FG_BLUE);
        qOut << "  " << location;
    }

    qOut << qEndl;

    QTextStream qStdOut(stdout);
    qStdOut << out;
}

/**
 * @brief uses `qInstallMessageHandler()` to change the formatting of console logs created by `QDebug`
 */
inline void setUpCustomLogging()
{
    qInstallMessageHandler(startup::_customizeQDebugHandler);

    qInfo() << "Use the `qInfo()` macro to print an info message to console.";
    qDebug() << "Use the `qDebug()` macro to print a debug message to console.";
    qWarning() << "Use the `qWarning()` macro to print a warning message to console.";
    qCritical() << "Use the `qCritical()` macro to print a critical message to console.";
    qInfo() << "Use the `qFatal()` macro to instantly abort your program and print a fatal message "
               "to console.";
    qInfo() << "Append the above macros with `.Nq()` to remove extra quotes in a console message.";
    qInfo() << "Append the above macros with `.Ns()` to remove extra spaces in a console message.";
}

/**
 * @brief handles all OpenGL-related setup
 */
inline void setOpenGLFormat()
{
#ifdef __cplusplus
    qInfo() << "C++ version:" << __cplusplus;
#endif

    QSurfaceFormat format;

    format.setOption(QSurfaceFormat::DeprecatedFunctions, false);  // deprecated gl not supported
    std::pair<int, int> version = std::make_pair(4, 0);
    QSurfaceFormat::OpenGLContextProfile profile = QSurfaceFormat::CoreProfile;
    int samples = 4;

    format.setVersion(version.first, version.second);  // Set OpenGL 4.0
    format.setProfile(profile);
    format.setSamples(samples);  // for 4-sample antialiasing. not always supported.

    QSurfaceFormat::setDefaultFormat(format);

    QSurfaceFormat newFormat = QSurfaceFormat::defaultFormat();
    std::pair<int, int> newVersion = format.version();
    QSurfaceFormat::OpenGLContextProfile newProfile = format.profile();
    int newSamples = format.samples();

    qInfo() << "OpenGL Format Info: (expected | actual)";
    qInfo().Nq() << "  Version:" << startup::_getVersionString(version) << "|"
                 << startup::_getVersionString(newVersion);
    qInfo() << "  Profile:" << startup::_getProfileString(profile) << "|"
            << startup::_getProfileString(newProfile);
    qInfo() << "  Samples:" << samples << "|" << newSamples;
}

/**
 *  @brief Sets up custom logging and handles all OpenGL-related setup
 *  
 *  Call at top of application's `main(..)` function. Delete extraneous function calls 
 *  besides setup of the `QApplication` and the `MainWindow`.
 */
inline void doSimpleSetup()
{
    console::createFunctionSection<void, void (*)()>("INSTRUCTIONS FOR CUSTOM LOGGING:",
                                                     startup::setUpCustomLogging);

    console::createFunctionSection<void, void (*)()>("APPLICATION DETAILS:",
                                                     startup::setOpenGLFormat);
}
}  // namespace startup
