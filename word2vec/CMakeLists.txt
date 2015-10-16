
set(SOURCE_WORD2VEC
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.cpp
        PARENT_SCOPE
)
set(HEADERS_WORD2VEC
        ${HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.hpp
        PARENT_SCOPE
)
set(SOURCE_COMPUTE_ACC
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/compute-accuracy.cpp
        PARENT_SCOPE
)
