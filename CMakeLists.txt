cmake_minimum_required(VERSION 2.6)
project(word2vecpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE Release)

find_package( Threads )

add_subdirectory("${PROJECT_SOURCE_DIR}/multivec")
add_subdirectory("${PROJECT_SOURCE_DIR}/word2vec")

set(DEPENDENCIES ${CMAKE_THREAD_LIBS_INIT})

add_executable(multivec-mono ${SOURCE_MULTIVEC_MONO} ${HEADERS_MULTIVEC_MONO})
target_link_libraries( multivec-mono ${DEPENDENCIES} )

add_executable(multivec-bi ${SOURCE_MULTIVEC_BI} ${HEADERS_MULTIVEC_BI})
target_link_libraries( multivec-bi ${DEPENDENCIES} )

add_executable(word2vec ${SOURCE_WORD2VEC} ${HEADERS_WORD2VEC})
target_link_libraries( word2vec ${DEPENDENCIES} )

add_executable(compute-accuracy ${SOURCE_COMPUTE_ACC})
target_link_libraries( compute-accuracy ${DEPENDENCIES} )
