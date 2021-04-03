cmake_minimum_required(VERSION 3.0.0)
project(Linear VERSION 0.1.0)

include(CTest)
enable_testing()

# Find Torch
set(PyTorch_version "1.7.0")
find_package(Torch ${PyTorch_version} EXACT QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")

# Linear Regression
add_executable(LinearRegression linear_regression.cpp)
set_target_properties(LinearRegression PROPERTIES
CXX_STANDARD 14
CXX_STANDARD_REQUIRED YES
)
target_link_libraries(LinearRegression ${TORCH_LIBRARIES})

# Logistic Regression
add_executable(LogisticRegression logistic_regression.cpp)
set_target_properties(LogisticRegression PROPERTIES
CXX_STANDARD 14
CXX_STANDARD_REQUIRED YES
)
target_link_libraries(LogisticRegression ${TORCH_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
