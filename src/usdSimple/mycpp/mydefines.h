#pragma once

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
