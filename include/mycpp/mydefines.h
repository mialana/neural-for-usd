#pragma once

#include <QtCore/qglobal.h>

#if defined(MYCPP_LIBRARY)
#define MYCPP_EXPORT Q_DECL_EXPORT
#else
#define MYCPP_EXPORT Q_DECL_IMPORT
#endif

#define Nq noquote
#define Ns nospace

/* To stringize defined macros */
#define _STR(x) #x
#define STR(x) _STR(x)

#define mkU std::make_unique
#define mkS std::make_shared
#define uPtr std::unique_ptr
#define sPtr std::shared_ptr

#define myOpt std::optional

#define CCP(_qStr) (_qStr.toLocal8Bit().data())
#define GLMMat4ToGF(_glmMtxPtr) (*reinterpret_cast<pxr::GfMatrix4d*>(_glmMtxPtr))
