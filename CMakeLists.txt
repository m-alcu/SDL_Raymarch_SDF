cmake_minimum_required(VERSION 3.16)

project(SDL_raymarching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(example_glfw_opengl3
    main.cpp
    vendor/imgui/imgui.cpp
    vendor/imgui/imgui_demo.cpp
    vendor/imgui/imgui_draw.cpp
    vendor/imgui/imgui_tables.cpp
    vendor/imgui/imgui_widgets.cpp
    vendor/imgui/imgui_impl_glfw.cpp
    vendor/imgui/imgui_impl_opengl3.cpp
)

target_include_directories(example_glfw_opengl3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui
)

target_compile_options(example_glfw_opengl3 PRIVATE -Wall -Wformat)

find_package(OpenGL REQUIRED)

# Try CMake package config first, then fall back to pkg-config.
set(GLFW_LINK_TARGET "")
find_package(glfw3 CONFIG QUIET)
if(TARGET glfw)
    set(GLFW_LINK_TARGET glfw)
elseif(TARGET glfw3::glfw)
    set(GLFW_LINK_TARGET glfw3::glfw)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
    set(GLFW_LINK_TARGET PkgConfig::GLFW3)
endif()

target_link_libraries(example_glfw_opengl3 PRIVATE
    OpenGL::GL
    ${GLFW_LINK_TARGET}
)
