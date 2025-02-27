#pragma once

#include <QtCore/qglobal.h>

#if defined(MYCPP_LIBRARY)
#define MYCPP_EXPORT Q_DECL_EXPORT
#else
#define MYCPP_EXPORT Q_DECL_IMPORT
#endif
