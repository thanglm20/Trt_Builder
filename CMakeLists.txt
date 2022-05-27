cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS)

project(trt_calib LANGUAGES CXX)

# set CUDA 
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

# set Opencv
set(OpenCV_DIR "D:/ThangLM/prebuilt-libopencv/opencv/build")
find_package(OpenCV  REQUIRED)



# set TensorRT
find_library(TensorRT_LIB nvinfer HINTS 
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
find_library(nvparsers nvparsers HINTS 
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
find_library(nvonnxparser nvonnxparser HINTS 
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
find_library(nvinfer_plugin nvinfer_plugin HINTS 
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
        
find_library(CUDNN_LIB cudnn HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
find_library(CUBLASLT_LIB cublasLt HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
find_library(CUDART_LIB cudart HINTS 
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

include_directories(${PROJECT_SOURCE_DIR}/common)

file(GLOB SOURCE_LIST 
common/*.cpp        
src/TrtBuilder.cpp 
src/main.cpp
)




cuda_add_executable(${PROJECT_NAME} ${SOURCE_LIST})


                
target_link_libraries(${PROJECT_NAME} 
${OpenCV_LIBS}
${CUBLASLT_LIB}
${CUBLAS_LIB}
${CUDNN_LIB}
${CUDART_LIB}
${TensorRT_LIB}
${nvparsers}
${nvonnxparser}
${nvinfer_plugin}
${CUDA_nppidei_LIBRARY}
${CUDA_nppig_LIBRARY}
)


message("CUBLASLT_LIB: ${CUBLASLT_LIB}")
message("CUBLAS_LIB: ${CUDA_CUBLAS_LIBRARIES}")
message("CUDNN_LIB: ${CUDNN_LIB}")
message("CUDART_LIB: ${CUDART_LIB}")
message("TensorRT_LIB: ${TensorRT_LIB}")
message("OpenCV_LIBS: ${OpenCV_LIBS}")

                        