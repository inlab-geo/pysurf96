enable_language(Fortran)
file( GLOB surf96_fortran_functions *.f )

##add_library(pysurf96 SHARED  ${surf96_fortran_functions})

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
Python_add_library(pysurf96 MODULE WITH_SOABI ${surf96_fortran_functions})

install(TARGETS surf96 DESTINATION ./surf96)