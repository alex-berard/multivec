cmake_minimum_required(VERSION 2.6)
project(multivec++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

find_package(Threads)

add_subdirectory("${PROJECT_SOURCE_DIR}/multivec")
add_subdirectory("${PROJECT_SOURCE_DIR}/word2vec")

set(DEPENDENCIES ${CMAKE_THREAD_LIBS_INIT})

add_executable(multivec ${MULTIVEC})
target_link_libraries(multivec ${DEPENDENCIES})

add_executable(multivec-bi ${MULTIVEC_BI})
target_link_libraries(multivec-bi ${DEPENDENCIES})

add_executable(word2vec ${WORD2VEC})
target_link_libraries(word2vec ${DEPENDENCIES})

add_executable(analogy ${ANALOGY})
target_link_libraries(analogy ${DEPENDENCIES})

add_library(multivec-lib SHARED ${MULTIVEC_LIB})
add_library(multivec-static STATIC ${MULTIVEC_LIB})

set_target_properties(multivec-static PROPERTIES OUTPUT_NAME multivec)
set_target_properties(multivec-lib PROPERTIES OUTPUT_NAME multivec)

install(TARGETS multivec-lib multivec-static DESTINATION lib)
install(TARGETS analogy word2vec multivec-bi multivec DESTINATION bin)
install(FILES multivec/bilingual.hpp  multivec/monolingual.hpp  multivec/serialization.hpp  multivec/utils.hpp  multivec/vec.hpp  word2vec/word2vec.hpp DESTINATION include)

foreach(i RANGE 1 6)
    add_subdirectory("${PROJECT_SOURCE_DIR}/multivec-v${i}")
    add_executable(multivec-v${i} ${MULTIVEC})
    install(TARGETS multivec-v${i} DESTINATION bin)
endforeach(i)
