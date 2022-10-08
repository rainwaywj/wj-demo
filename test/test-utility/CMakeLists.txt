# test-utility

set(TARGET_NAME test-utility)

set(SOURCE_LIST
        test-utility.cpp
)

add_executable(
        ${TARGET_NAME}
        ${SOURCE_LIST}
)
target_include_directories(
        ${TARGET_NAME}
        PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
)
