# Add the test executable
add_executable(test_plumerise test_plumerise.F90)

# Link the sofiev_mod library, which is a target defined in src/fortran/CMakeLists.txt
# CMake can find it automatically.
target_link_libraries(test_plumerise PRIVATE sofiev_mod)

# Add the executable as a test to be run by CTest
add_test(NAME FortranPlumeRiseTest COMMAND test_plumerise)
