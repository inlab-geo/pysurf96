enable_language(Fortran)
file( GLOB surf96_fortran_functions *.f90 )

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
Python_add_library(surf96 MODULE WITH_SOABI ${surf96_fortran_functions})

install(TARGETS surf96 DESTINATION ./pysurf96)
