
set(MULTIVEC_BI
    ${CMAKE_CURRENT_SOURCE_DIR}/main-bi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bilingual.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bilingual.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vec.hpp    
    PARENT_SCOPE
)

set(MULTIVEC_MONO
    ${CMAKE_CURRENT_SOURCE_DIR}/main-mono.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vec.hpp
    PARENT_SCOPE
)

set(MULTIVEC_LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bilingual.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/distance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/monolingual.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bilingual.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vec.hpp
    PARENT_SCOPE
)