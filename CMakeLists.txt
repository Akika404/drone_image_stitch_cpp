cmake_minimum_required(VERSION 3.10)
project(drone_image_stitch_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: 源文件与执行字符集为 UTF-8，解决中文等 Unicode 字符编译问题
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 可选：手动指定 OpenCV cmake 配置目录（例如 /opt/homebrew/lib/cmake/opencv4）
set(OPENCV_BUILD_PATH "" CACHE PATH "Optional OpenCV config directory")
if(OPENCV_BUILD_PATH)
    set(OpenCV_DIR "${OPENCV_BUILD_PATH}" CACHE PATH "OpenCV config directory" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d stitching)

add_executable(drone_image_stitch_cpp
        src/main.cpp
        src/stitch_app.cpp
        src/stitch_config.cpp
        src/stitch_common.cpp
        src/stitch_robust.cpp
        src/stitch_global.cpp
        src/image_loader.cpp
        src/visual_flight_grouper.cpp
)
target_link_libraries(drone_image_stitch_cpp PRIVATE ${OpenCV_LIBS})

# 将 OpenCV DLL 复制到可执行文件目录，解决运行时 0xC0000135 (DLL 未找到) 错误
set(OPENCV_BIN "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin")
if(EXISTS "${OPENCV_BIN}")
    add_custom_command(TARGET drone_image_stitch_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${OPENCV_BIN}" "$<TARGET_FILE_DIR:drone_image_stitch_cpp>"
        COMMENT "Copying OpenCV DLLs to output directory"
    )
endif()
