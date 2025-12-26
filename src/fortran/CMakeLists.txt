# Create a library from the Sofiev module
add_library(sofiev_mod STATIC plumerise_sofiev_mod.F90)

# Create an executable from the driver
add_executable(sofiev_driver sofiev_driver.F90)

# Link the library to the executable
target_link_libraries(sofiev_driver PRIVATE sofiev_mod)
