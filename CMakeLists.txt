cmake_minimum_required(VERSION 2.6)
project(multivec++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

find_package(Threads)

add_subdirectory("${PROJECT_SOURCE_DIR}/multivec")
add_subdirectory("${PROJECT_SOURCE_DIR}/word2vec")

set(DEPENDENCIES ${CMAKE_THREAD_LIBS_INIT})

add_executable(multivec-mono ${MULTIVEC_MONO})
target_link_libraries(multivec-mono ${DEPENDENCIES})

add_executable(multivec-bi ${MULTIVEC_BI})
target_link_libraries( multivec-bi ${DEPENDENCIES})

add_executable(word2vec ${WORD2VEC})
target_link_libraries(word2vec ${DEPENDENCIES})

add_executable(compute-accuracy ${COMPUTE_ACC})
target_link_libraries(compute-accuracy ${DEPENDENCIES})

add_library(multivec SHARED ${MULTIVEC_LIB})
ADD_LIBRARY(multivec-static STATIC ${MULTIVEC_LIB})

SET_TARGET_PROPERTIES(multivec-static PROPERTIES OUTPUT_NAME multivec)
install(TARGETS multivec multivec-static DESTINATION lib)
install(TARGETS compute-accuracy word2vec multivec-bi multivec-mono DESTINATION bin)
install(FILES multivec/bilingual.hpp  multivec/monolingual.hpp  multivec/serialization.hpp  multivec/utils.hpp  multivec/vec.hpp  word2vec/word2vec.hpp DESTINATION include)

#add_library(multivec STATIC ${MULTIVEC_LIB})
#install(TARGETS multivec DESTINATION lib)


