#pragma once

#include <QTextStream>
#include <QString>

namespace console
{

enum Code {
    FG_BLACK = 30,
    FG_RED = 31,
    FG_GREEN = 32,
    FG_YELLOW = 33,
    FG_BLUE = 34,
    FG_CYAN = 36,

    FG_ORANGE = 208,  // not available on some systems
};

/**
 * @brief set color of stream
 * @param stream
 * @param code
 */
inline void setColor(QTextStream& stream, Code code)
{
    if (code == FG_ORANGE) {
        stream << "\033[38;5;";
    } else {
        stream << "\033[";
    }

    stream << code << "m";
    stream.flush();
}

/**
 * @brief prints a line of '*' to break up logging
 * @param stream
 */
inline void printLineBreak(QTextStream& stream)
{
    stream << QString(50, '*') << "\n";
    stream.flush();
}

/**
 *  @brief Formats all console logging that happens in a function into a section.
 *  Best practices: Templates are declared in header.
 *  
 */
template<typename Ret, typename Fn, typename... Args>
Ret inline createFunctionSection(const char* title, Fn fn, Args... args)
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
