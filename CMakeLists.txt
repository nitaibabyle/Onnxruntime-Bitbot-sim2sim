cmake_minimum_required(VERSION 3.14)

project(bitbot)

# Set paths to ONNX Runtime
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.22.0")

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(
  onnxruntime
)
