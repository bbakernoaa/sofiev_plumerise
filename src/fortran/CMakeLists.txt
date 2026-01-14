# Create a shared library from the Sofiev module for Python ctypes interoperability
add_library(sofiev_mod SHARED plumerise_sofiev_mod.F90)

# Make the module files (.mod) from this library available to other targets
target_include_directories(sofiev_mod PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Control the output location and name of the shared library
set_target_properties(sofiev_mod PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/sofiev_model"
    OUTPUT_NAME "sofiev"
)

# Create an executable from the driver
add_executable(sofiev_driver sofiev_driver.F90)

# Link the library to the executable
target_link_libraries(sofiev_driver PRIVATE sofiev_mod)
