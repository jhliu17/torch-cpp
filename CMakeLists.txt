cmake_minimum_required(VERSION 3.0.0)
project(torch-cpp VERSION 0.1.0)

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

# MLP
add_executable(MLP)
target_sources(MLP PRIVATE mlp/src/main.cpp
                           mlp/src/mlp.cpp
                           mlp/include/mlp.h 
)
target_include_directories(MLP PRIVATE mlp/include)
set_target_properties(MLP PROPERTIES
CXX_STANDARD 14
CXX_STANDARD_REQUIRED YES
)
target_link_libraries(MLP ${TORCH_LIBRARIES})

# RNN Language Model
add_executable(RNNLM)
target_sources(RNNLM PRIVATE language_model/src/main.cpp
                             language_model/src/rnn_lm.cpp
                             language_model/src/dictionary.cpp
                             language_model/src/corpus.cpp
                             language_model/include/rnn_lm.h
                             language_model/include/dictionary.h
                             language_model/include/corpus.h
                             language_model/include/tqdm.h
)
target_include_directories(RNNLM PRIVATE language_model/include)
set_target_properties(RNNLM PROPERTIES
CXX_STANDARD 14
CXX_STANDARD_REQUIRED YES
)
target_link_libraries(RNNLM ${TORCH_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
