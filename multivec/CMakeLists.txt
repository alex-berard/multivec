
set(SOURCE_MULTIVEC_MONO
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/main-mono.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.cpp
        PARENT_SCOPE
)
set(HEADERS_MULTIVEC_MONO
        ${HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
        PARENT_SCOPE
)

set(SOURCE_MULTIVEC_BI
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/main-bi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bilingual.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.cpp
        PARENT_SCOPE
)

set(HEADERS_MULTIVEC_BI
        ${HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bilingual.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
        PARENT_SCOPE
)
