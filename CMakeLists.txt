cmake_minimum_required (VERSION 2.8)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project (destina)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)

FIND_PACKAGE (OpenCV REQUIRED)
FIND_PACKAGE (OpenMP)
FIND_PACKAGE (SWIG)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories ("${PROJECT_SOURCE_DIR}/build/lib")
add_library (libdestinasys SHARED
	include/destina_sys_lib.h)

add_library (libdestina SHARED
        include/proc_tools.h
        include/ifsc.h
        include/destin_network.h
        include/destina_lib.h)

# denoising autoencoder library
add_library (libsa SHARED
        include/SA.h)

set_target_properties(libdestinasys PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(libdestina PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(libsa PROPERTIES LINKER_LANGUAGE CXX)

include_directories ("${PROJECT_SOURCE_DIR}/include")

add_executable (gabor_test src/gabor_test.cpp)
target_link_libraries(gabor_test ${OpenCV_LIBS})
target_link_libraries(gabor_test libdestinasys)
target_link_libraries(gabor_test libdestina)

add_executable (whitening_test src/whitening_test.cpp)
target_link_libraries(whitening_test ${OpenCV_LIBS})
target_link_libraries(whitening_test libdestinasys)
target_link_libraries(whitening_test libdestina)

add_executable (ifsc_test src/ifsc_test.cpp)
target_link_libraries(ifsc_test ${OpenCV_LIBS})
target_link_libraries(ifsc_test libdestinasys)
target_link_libraries(ifsc_test libdestina)

add_executable (sa src/sa.cpp)
target_link_libraries(sa ${OpenCV_LIBS})
target_link_libraries(sa libdestinasys)
target_link_libraries(sa libdestina)
target_link_libraries(sa libsa)

add_executable (net_test src/network_test.cpp)
target_link_libraries(net_test ${OpenCV_LIBS})
target_link_libraries(net_test libdestinasys)
target_link_libraries(net_test libdestina)
target_link_libraries(net_test libsa)

add_executable (cifar src/cifar_test.cpp)
target_link_libraries(cifar ${OpenCV_LIBS})
target_link_libraries(cifar libdestinasys)
target_link_libraries(cifar libdestina)
target_link_libraries(cifar libsa)

add_executable (classify src/classify_test.cpp)
target_link_libraries(classify ${OpenCV_LIBS})
target_link_libraries(classify libdestinasys)
target_link_libraries(classify libdestina)
target_link_libraries(classify libsa)

# Python Support

INCLUDE("UseSWIG")

add_subdirectory(Python)