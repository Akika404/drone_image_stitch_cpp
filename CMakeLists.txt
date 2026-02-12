cmake_minimum_required(VERSION 3.10)
project(drone_image_stitch_cpp)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_executable(drone_image_stitch_cpp
        src/main.cpp
        src/image_loader.cpp
        src/pos_record.cpp
        src/pos_reader.cpp
        src/pos_image_grouper.cpp
)
target_link_libraries(drone_image_stitch_cpp PRIVATE ${OpenCV_LIBS})
