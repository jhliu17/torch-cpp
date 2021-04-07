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

# Corpus Lib
add_library(CORPUS SHARED language_model/src/corpus.cpp
                           language_model/src/dictionary.cpp
)
add_library(CORPUS_STAIC STATIC language_model/src/corpus.cpp
                                language_model/src/dictionary.cpp
)
target_include_directories(CORPUS PRIVATE language_model/include)
target_include_directories(CORPUS_STAIC PRIVATE language_model/include)
SET_TARGET_PROPERTIES(CORPUS PROPERTIES OUTPUT_NAME "corpus"
                                        CXX_STANDARD 14
                                        CXX_STANDARD_REQUIRED YES
                                        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/language_model/lib"
)
SET_TARGET_PROPERTIES(CORPUS_STAIC PROPERTIES OUTPUT_NAME "corpus"
                                              CXX_STANDARD 14
                                              CXX_STANDARD_REQUIRED YES
                                              ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/language_model/lib"
)
target_link_libraries(CORPUS ${TORCH_LIBRARIES})
target_link_libraries(CORPUS_STAIC ${TORCH_LIBRARIES})

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

# RNN Language Model with Dynamic Linking
add_executable(RNNLM_DL)
target_sources(RNNLM_DL PRIVATE language_model/src/main.cpp
                                language_model/src/rnn_lm.cpp
)
target_include_directories(RNNLM_DL PRIVATE language_model/include)
target_link_directories(RNNLM_DL PRIVATE language_model/lib)
set_target_properties(RNNLM_DL PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)
target_link_libraries(RNNLM_DL libcorpus.dylib)
target_link_libraries(RNNLM_DL ${TORCH_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
