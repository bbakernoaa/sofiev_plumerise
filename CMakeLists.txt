cmake_minimum_required(VERSION 3.10)
project(SofievPlumeModel LANGUAGES Fortran)

# Add the Fortran subdirectory
add_subdirectory(src/fortran)

# Enable testing and add the test subdirectory
enable_testing()
add_subdirectory(tests/fortran)
