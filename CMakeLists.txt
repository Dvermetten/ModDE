cmake_minimum_required(VERSION 3.18...3.31)

project(modde LANGUAGES CXX)

# C++17 (D-A4)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Opt-in host-specific optimisation for local builds only. Never enabled for
# distributed wheels (D-A18).
option(MODDE_NATIVE "Compile with -march=native (local builds only)" OFF)

# Eigen is a git submodule (D-A3).
set(MODDE_EIGEN_DIR "${PROJECT_SOURCE_DIR}/external/eigen")
if (NOT EXISTS "${MODDE_EIGEN_DIR}/Eigen/Dense")
  message(FATAL_ERROR
    "Eigen not found in ${MODDE_EIGEN_DIR}. Run: git submodule update --init --recursive")
endif()

find_package(Python 3.10
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT}
  REQUIRED)
find_package(nanobind CONFIG REQUIRED)

file(GLOB MODDE_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")

nanobind_add_module(_core STABLE_ABI NB_STATIC ${MODDE_SOURCES})
target_include_directories(_core PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(_core SYSTEM PRIVATE "${MODDE_EIGEN_DIR}")

if (MSVC)
  target_compile_options(_core PRIVATE /O2 /bigobj /permissive- /W3)
else()
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra)
  if (MODDE_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION modde)

# Type stubs for the extension (+ PEP 561 marker for the whole package)
nanobind_add_stub(
  _core_stub
  MODULE _core
  OUTPUT _core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core
  MARKER_FILE py.typed
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi" "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION modde)
