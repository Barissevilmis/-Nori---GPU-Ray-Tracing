cmake_minimum_required (VERSION 3.8)
project(norigpu CUDA CXX C)
#SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
add_subdirectory(ext ext_build)


include_directories(
  # Nori include files
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  # tinyformat string formatting library
  ${TFM_INCLUDE_DIR}
  # Eigen linear algebra library
  SYSTEM ${EIGEN_INCLUDE_DIR}
  # OpenEXR high dynamic range bitmap library
  SYSTEM ${OPENEXR_INCLUDE_DIRS}
  # Intel Thread Building Blocks
  SYSTEM ${TBB_INCLUDE_DIR}
  #CUDA: Nvidia GPU
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  # Pseudorandom number generator
  ${PCG32_INCLUDE_DIR}
  # PugiXML parser
  ${PUGIXML_INCLUDE_DIR}
  # Helper functions for statistical hypothesis tests
  ${HYPOTHESIS_INCLUDE_DIR}
  # GLFW library for OpenGL context creation
  SYSTEM ${GLFW_INCLUDE_DIR}
  # GLEW library for accessing OpenGL functions
  SYSTEM ${GLEW_INCLUDE_DIR}
  # NanoVG drawing library
  SYSTEM ${NANOVG_INCLUDE_DIR}
  # NanoGUI user interface library
  SYSTEM ${NANOGUI_INCLUDE_DIR}
  SYSTEM ${NANOGUI_EXTRA_INCS}
  # Portable filesystem API
  SYSTEM ${FILESYSTEM_INCLUDE_DIR}
  # STB Image Write
  SYSTEM ${STB_IMAGE_WRITE_INCLUDE_DIR}
)
# The following lines build the main executable. If you add a source
# code file to Nori, be sure to include it in this list.
add_executable(norigpu 

# # Header files
 include/nori/colorgpu.h
 include/nori/bitmap.h
 include/nori/bsdf.h
 include/nori/common.h
 include/nori/object.h
 include/nori/parser.h
 include/nori/proplist.h
 include/nori/vectorgpu.h
 include/nori/raygpu.h 
 include/nori/mesh.h
 include/nori/bboxgpu.h
 include/nori/emitter.h
 include/nori/framegpu.h
 include/nori/integrator.h
 include/nori/sampler.h
 include/nori/scenegpu.h
 include/nori/timer.h
 include/nori/warpgpu.h
 include/nori/block.h
 include/nori/accelbvh.h
 include/nori/rfilter.h
 include/nori/camera.h
 include/nori/dpdfgpu.h
 include/nori/transform.h

 src/bitmap.cpp
 src/camera.cpp
 src/object.cpp
 src/parser.cpp
 src/proplist.cpp
 src/gui.cpp
 src/common.cpp
 src/main.cpp
 src/rfilters.cpp
 src/block.cpp
 src/meshes.cpp
 
 kernels/area.cu
 kernels/independent.cu
 kernels/diffuse.cu
 kernels/perspective.cu
 kernels/warp.cu
 kernels/integrators.cu
 kernels/scenes.cu
 kernels/accelbvh.cu
)



add_definitions(${NANOGUI_EXTRA_DEFS})
if (CMAKE_GENERATOR STREQUAL "Ninja")
  if (CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -std=c++17 -rdc=true")
    set(CUDA_NVCC_FLAGS "${CMAKE_NVCC_FLAGS} -O3 -std=c++17 -rdc=true")
    #set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fdiagnostics-color=always")
  endif()
endif()
# Force colored output for the ninja generator

# if (CMAKE_GENERATOR STREQUAL "Ninja")
#   if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fcolor-diagnostics")
#   elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fdiagnostics-color=always")
#   endif()
# endif()

if (WIN32)
  target_link_libraries(norigpu tbb_static pugixml IlmImf nanogui ${NANOGUI_EXTRA_LIBS} zlibstatic rt cuda)
else()
  target_link_libraries(norigpu tbb_static pugixml IlmImf nanogui ${NANOGUI_EXTRA_LIBS} rt cuda)
endif()




#target_compile_options(norigpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                       --generate-line-info
#                       --use_fast_math
#                       --relocatable-device-code=true>)
#target_compile_features(norigpu PRIVATE cxx_std_17)

set_target_properties(norigpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON POSITION_INDEPENDENT_CODE ON)
#target_compile_features(norigpu PRIVATE cxx_std_17)
# vim: set et ts=2 sw=2 ft=cmake nospell:
