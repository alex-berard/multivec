cmake_minimum_required(VERSION 2.6)
project(word2vecpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE Release)

find_package( Boost REQUIRED COMPONENTS program_options serialization )
find_package( Threads )
include_directories( ${Boost_INCLUDE_DIRS} )

add_subdirectory("${PROJECT_SOURCE_DIR}/bivec")
add_subdirectory("${PROJECT_SOURCE_DIR}/word2vec")

set(DEPENDENCIES ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SERIALIZATION_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

add_executable(word2vecpp ${SOURCE_WORD2VECPP} ${HEADERS_WORD2VECPP})
target_link_libraries( word2vecpp ${DEPENDENCIES} )

add_executable(bivec ${SOURCE_BIVEC} ${HEADERS_BIVEC})
target_link_libraries( bivec ${DEPENDENCIES} )

add_executable(word2vec ${SOURCE_WORD2VEC} ${HEADERS_WORD2VEC})
target_link_libraries( word2vec ${DEPENDENCIES} )

add_executable(compute-accuracy ${SOURCE_COMPUTE_ACC})
target_link_libraries( compute-accuracy ${DEPENDENCIES} )
