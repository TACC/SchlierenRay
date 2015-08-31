cmake_minimum_required(VERSION 2.6)
project(Schlieren)

# project version
SET( ${PROJECT_NAME}_MAJOR_VERSION 0 )
SET( ${PROJECT_NAME}_MINOR_VERSION 2 )
SET( ${PROJECT_NAME}_PATCH_LEVEL 0 )

# project options
OPTION( BUILD_SHARED_LIBS "Set to OFF to build static libraries" ON )
# set default install prefix to project root directory
IF( CMAKE_INSTALL_PREFIX STREQUAL "/usr/local" )
   SET( CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}" )
ENDIF()
INCLUDE (${CMAKE_CURRENT_SOURCE_DIR}/CMake/FindTeem.cmake)
#include directories
INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )
# install include files
INSTALL( DIRECTORY "${PROJECT_SOURCE_DIR}/include"
     DESTINATION . PATTERN "*~" EXCLUDE PATTERN "*CVS*" EXCLUDE )

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH}) 
if (!Teem_FOUND)
  ERROR("teem not found")
endif()

# Using the CUDA_COMPILE macro
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
#set(source_files ernel_functions.cu host.cu)
#CUDA_COMPILE(CUDA_FILES ernel_functions.cu host.cu)

#ADD_EXECUTABLE(cuda_compile_example
#  ${CUDA_FILES}
#  ${source_files}
#  main.cpp kernel_volume.c kernel_post.c
#  )
#TARGET_LINK_LIBRARIES(cuda_compile_example
# ${CUDA_LIBRARIES}
# /usr/local/lib/libteem.so glut GL
# )

# Generating PTX files.
# CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)

# Add a special target to clean nvcc generated files.
#CUDA_BUILD_CLEAN_TARGET()

#find_library(TEEM_LIBRARY NAMES teem)
find_library(GLUT_LIBRARY REQUIRED NAMES glut)
find_library(OPENGL_LIBRARY REQUIRED NAMES OpenGL GL)
find_package(CUDA REQUIRED)
#INCLUDE(/home/carson/svn/teem/build/TeemConfig.cmake)
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${Teem_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${CUDA_SDK_ROOT_DIR}/C/common/inc )
INCLUDE_DIRECTORIES( ${INCLUDE_DIRECTORIES} ${CUDA_SDK_ROOT_DIR}/samples/common/inc )
MESSAGE(STATUS "teem_include_dirs: " ${Teem_INCLUDE_DIRS})
MESSAGE(STATUS "cuda_sdk_dir: " ${CUDA_SDK_ROOT_DIR})

LINK_DIRECTORIES(${Teem_LIBRARY_DIRS})
link_libraries(${Teem_LIBRARIES} ${GLUT_LIBRARY} ${OPENGL_LIBRARY})
MESSAGE("teem libraries: " ${Teem_LIBRARIES})
SET(SOURCE_FILES schlierenfilter.cpp schlierenimagefilter.cpp schlierenrenderer.cpp host_render.cu)

cuda_add_library(${PROJECT_NAME} ${SOURCE_FILES} )
SET(SOURCE_FILES main.cpp ${SOURCE_FILES})
cuda_add_executable(schlierenCL ${SOURCE_FILES})
#target_link_libraries(schlierenCL /usr/local/cuda/lib/libcudart.dylib  )
#target_link_libraries(schlieren ${CUDA_LIBRARIES} )

# create symbolic lib target for calling target lib_XXX
#ADD_CUSTOM_TARGET( lib DEPENDS lib_${PROJECT_NAME} )
# change lib_target properties
#SET( ${PROJECT_NAME}_SOVERSION 1 )
#SET_TARGET_PROPERTIES( lib_${PROJECT_NAME} PROPERTIES
   # create *nix style library versions + symbolic links
#   VERSION "${${PROJECT_NAME}_VERSION}"
#   SOVERSION "${${PROJECT_NAME}_SOVERSION}"
   # allow creating static and shared libs without conflicts
#   CLEAN_DIRECT_OUTPUT 1
   # avoid conflicts between library and binary target names
#   OUTPUT_NAME ${PROJECT_NAME} )

# install library
#INSTALL( TARGETS lib_${PROJECT_NAME} DESTINATION lib PERMISSIONS
#     OWNER_READ OWNER_WRITE OWNER_EXECUTE
#     GROUP_READ GROUP_EXECUTE
#     WORLD_READ WORLD_EXECUTE )

CUDA_BUILD_CLEAN_TARGET()

