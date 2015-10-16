
set(SOURCE_WORD2VECPP
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation.cpp
        PARENT_SCOPE
)
set(HEADERS_WORD2VECPP
        ${HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
        PARENT_SCOPE
)

set(SOURCE_BIVEC
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/bivec_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bivec.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation.cpp
        PARENT_SCOPE
)

set(HEADERS_BIVEC
        ${HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/word2vec.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bivec.hpp
        PARENT_SCOPE
)