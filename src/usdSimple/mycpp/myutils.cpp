#include "myutils.h"

#include "mydefines.h"

#include <QDateTime>
#include <QApplication>
#include <QSurfaceFormat>
#include <QFile>

void console::setColor(QTextStream& stream, Code code)
{
    if (code == FG_ORANGE) {
        stream << "\033[38;5;";
    } else {
        stream << "\033[";
    }

    stream << code << "m";
    stream.flush();
}

void console::printLineBreak(QTextStream& stream)
{
    stream << QString(50, '*') << "\n";
    stream.flush();
}

QString startup::_getVersionString(std::pair<int, int> version)
{
    QString result;
    QTextStream stream(&result);
    stream << version.first << "." << version.second;
    return result;
}

QString startup::_getProfileString(int prof)
{
    return prof == QSurfaceFormat::CoreProfile            ? "Core"
           : prof == QSurfaceFormat::CompatibilityProfile ? "Compatibility"
                                                          : "None";
}

void startup::setOpenGLFormat()
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

QString startup::_getTimestamp()
{
    QString result = "[";

    QDateTime dateTime = QDateTime::currentDateTime();
    QString dtString = dateTime.toString(Qt::DateFormat ::ISODateWithMs);
    dtString = dtString.replace('T', ' ');
    QString timeZone = dateTime.timeZoneAbbreviation();

    result += dtString + " " + timeZone + "] ";
    return result;
};

void startup::_customizeQDebugHandler(QtMsgType type,
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
    console::setColor(qOut, FG_BLACK);

    qOut << timestamp;

    // append type of message in custom color
    switch (type) {
        case QtInfoMsg:
            console::setColor(qOut, FG_GREEN);
            qOut << "[INFO] ";
            break;
        case QtDebugMsg:
            console::setColor(qOut, FG_CYAN);
            qOut << "[DEBUG] ";
            break;
        case QtWarningMsg:
            console::setColor(qOut, FG_YELLOW);
            qOut << "[WARN] ";
            break;
        case QtCriticalMsg:
            console::setColor(qOut, FG_ORANGE);
            qOut << "[CRITICAL] ";
            break;
        case QtFatalMsg:
            console::setColor(qOut, FG_RED);
            qOut << "[FATAL] ";
            break;
    }

    console::setColor(qOut, FG_BLACK);
    qOut << inputMsg;

    // append location if not an info type
    if (type != QtInfoMsg && context.file) {
        console::setColor(qOut, FG_BLUE);
        qOut << "  " << location;
    }

    qOut << qEndl;

    QTextStream qStdOut(stdout);
    qStdOut << out;

    QString outFilePath = QString(STR(PROJECT_PATH)) + "/log.txt";

    QFile outFile(outFilePath);
    outFile.open(QIODevice::Append | QIODevice::Text);

    QTextStream qOutFile(&outFile);
    qOutFile << inputMsg << " " << location << qEndl;
    outFile.close();
}

void startup::setUpCustomLogging()
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

void startup::doSimpleSetup()
{
    console::createFunctionSection<void, void (*)()>("INSTRUCTIONS FOR CUSTOM LOGGING:",
                                                     startup::setUpCustomLogging);

    console::createFunctionSection<void, void (*)()>("APPLICATION DETAILS:",
                                                     startup::setOpenGLFormat);
}

QRgb linAlg::mapDirectionToRgb(const Vector3f& d)
{
    Color3f col;
    Color3i col255;
    QRgb myQRgb;

    col = d + Color3f(1.f);
    col *= 0.5f;

    col255 = col * 255.f;

    myQRgb = qRgb(col255.r, col255.g, col255.b);  // `qRgb()` is QT convenience function

    return myQRgb;
}

Point3f linAlg::doPoint3fXMat4(const glm::mat4& mat, const glm::vec3& pt)
{
    Point4f homogenized = glm::vec4(pt, 1);  // points must apply translation column
    Point4f transformed = mat * homogenized;

    Point3f result = transformed.xyz();  // exp(w) = 1
    return result;
}

Vector3f linAlg::doVec3fXMat4(const glm::mat4& mat, const glm::vec3& vec)
{
    Vector4f homogenized = glm::vec4(vec, 0);  // vectors disregard translation column
    Vector4f transformed = mat * homogenized;

    Vector3f result = transformed.xyz();  // exp(w) = 0. Do not normalize to cover all use cases.
    return result;
}
