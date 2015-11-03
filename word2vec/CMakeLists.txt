
set(WORD2VEC
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec-main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.hpp
        PARENT_SCOPE
)

set(COMPUTE_ACC
        ${CMAKE_CURRENT_SOURCE_DIR}/compute-accuracy.cpp
        PARENT_SCOPE
)
