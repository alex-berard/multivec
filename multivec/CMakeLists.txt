
set(SOURCE_MULTIVEC_MONO
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-mono-main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-mono.cpp
        PARENT_SCOPE
)
set(HEADERS_MULTIVEC_MONO
        ${HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-mono.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
        PARENT_SCOPE
)

set(SOURCE_MULTIVEC_BI
        ${SOURCE}
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-bi-main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-bi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-mono.cpp
        PARENT_SCOPE
)

set(HEADERS_MULTIVEC_BI
        ${HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-mono.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multivec-bi.hpp
        PARENT_SCOPE
)
