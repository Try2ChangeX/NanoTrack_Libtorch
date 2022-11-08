cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(NanoTrack_Libtorch)

set(CMAKE_PREFIX_PATH /home/calyx/libtorch)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(NanoTrack_Libtorch demo_libtorch.cpp nanotrack.cpp)
target_link_libraries(NanoTrack_Libtorch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET NanoTrack_Libtorch PROPERTY CXX_STANDARD 14)
