
%module(directors="1") SWIG_MODULE_NAME 
%{
/* includes that are needed to compile */
#include "destina_sys_lib.h"
#include "destina_lib.h"
#include "destin_network.h"
#include "ifsc.h"
#include "proc_tools.h"
#include "SA.h"
%}


/* Lets you use script strings easily with c++ strings */
%include "std_string.i"

/* the other classes to generate wrappers for */
%#include "destin_network.h"
%#include "SA.h"

/* use c++ vector like a python list */
%include "std_vector.i"
namespace std {
%template(IntVector) vector<int>;
%template(ShortVector) vector<short>;
%template(FloatVector) vector<float>;
}


/* carrays.i so you can use a c++ pointer like an array */
%include "carrays.i" 
%array_class(int, SWIG_IntArray);
%array_class(float, SWIG_FloatArray);
%array_functions(float *, SWIG_Float_p_Array);
%array_class(SparseAE::SA, SWIG_SAArray);
%array_class(SparseAE::SAA, SWIG_SAAArray);