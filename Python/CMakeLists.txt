FIND_PACKAGE(PythonLibs)
    
if(PYTHONLIBS_FOUND)
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    SET_SOURCE_FILES_PROPERTIES(swig_destina_python.i PROPERTIES CPLUSPLUS ON)
        
    SET(CMAKE_SWIG_OUTDIR . )            
    SWIG_ADD_MODULE(pydestina python swig_destina_python.i )
    
    SWIG_LINK_LIBRARIES(pydestina ${OpenCV_LIBS} libdestinasys libdestina libsa ${PYTHON_LIBRARY})

endif(PYTHONLIBS_FOUND)
