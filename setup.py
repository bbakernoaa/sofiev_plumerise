import os
import shutil
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py

# --- Custom Build Command ---
class CustomBuild(build_py):
    def run(self):
        # --- 1. Compile Fortran Code ---
        # Get the project root directory
        project_dir = os.path.abspath(os.path.dirname(__file__))

        # Define a temporary build directory for CMake
        build_temp = os.path.join(project_dir, "build", "fortran_build")
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        print("--- Configuring and Building Fortran library ---")
        # Correctly point CMake to the project's root directory
        subprocess.check_call(["cmake", project_dir], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)

        # --- 2. Copy the compiled library into the package ---
        # The library will be in the `src/fortran` subdirectory of the build dir
        lib_src_dir = os.path.join(build_temp, "src", "fortran")
        # The destination is the `sofiev_model` package inside the build/lib dir
        lib_dst_dir = os.path.join(os.path.abspath(self.build_lib), "sofiev_model")

        # Ensure the destination directory exists
        if not os.path.exists(lib_dst_dir):
            os.makedirs(lib_dst_dir)

        # Find the library file (e.g., 'libsofiev.so')
        lib_filename = ""
        for f in os.listdir(lib_src_dir):
            if f.startswith("libsofiev") and (f.endswith(".so") or f.endswith(".dylib") or f.endswith(".dll")):
                lib_filename = f
                break

        if not lib_filename:
            raise FileNotFoundError("Could not find compiled Fortran library.")

        shutil.copy2(os.path.join(lib_src_dir, lib_filename), lib_dst_dir)
        print(f"--- Copied {lib_filename} to {lib_dst_dir} ---")

        # --- 3. Run the standard Python build ---
        super().run()

# --- Setup Configuration ---
setup(
    cmdclass={"build_py": CustomBuild},
    include_package_data=True,
    package_data={
        "sofiev_model": ["*.so", "*.dll", "*.dylib"],
    },
    zip_safe=False,
)
