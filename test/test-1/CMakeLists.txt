# test-1

set(TARGET_NAME test-1)

set(SOURCE_LIST
        test-1.cpp
)

add_executable(
        ${TARGET_NAME}
        ${SOURCE_LIST}
)
target_include_directories(
        ${TARGET_NAME}
        PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
)
