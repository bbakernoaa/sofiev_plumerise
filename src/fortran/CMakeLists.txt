# Create a library from the Sofiev module
add_library(sofiev_mod STATIC plumerise_sofiev_mod.F90)

# Make the module files (.mod) from this library available to other targets
target_include_directories(sofiev_mod PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Create an executable from the driver
add_executable(sofiev_driver sofiev_driver.F90)

# Link the library to the executable
target_link_libraries(sofiev_driver PRIVATE sofiev_mod)
