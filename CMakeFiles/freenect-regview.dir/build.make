# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/scao/Desktop/kinview

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/scao/Desktop/kinview

# Include any dependencies generated for this target.
include CMakeFiles/freenect-regview.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/freenect-regview.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/freenect-regview.dir/flags.make

CMakeFiles/freenect-regview.dir/regview.c.o: CMakeFiles/freenect-regview.dir/flags.make
CMakeFiles/freenect-regview.dir/regview.c.o: regview.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/scao/Desktop/kinview/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/freenect-regview.dir/regview.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/freenect-regview.dir/regview.c.o   -c /home/scao/Desktop/kinview/regview.c

CMakeFiles/freenect-regview.dir/regview.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect-regview.dir/regview.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/scao/Desktop/kinview/regview.c > CMakeFiles/freenect-regview.dir/regview.c.i

CMakeFiles/freenect-regview.dir/regview.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect-regview.dir/regview.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/scao/Desktop/kinview/regview.c -o CMakeFiles/freenect-regview.dir/regview.c.s

CMakeFiles/freenect-regview.dir/regview.c.o.requires:
.PHONY : CMakeFiles/freenect-regview.dir/regview.c.o.requires

CMakeFiles/freenect-regview.dir/regview.c.o.provides: CMakeFiles/freenect-regview.dir/regview.c.o.requires
	$(MAKE) -f CMakeFiles/freenect-regview.dir/build.make CMakeFiles/freenect-regview.dir/regview.c.o.provides.build
.PHONY : CMakeFiles/freenect-regview.dir/regview.c.o.provides

CMakeFiles/freenect-regview.dir/regview.c.o.provides.build: CMakeFiles/freenect-regview.dir/regview.c.o

# Object files for target freenect-regview
freenect__regview_OBJECTS = \
"CMakeFiles/freenect-regview.dir/regview.c.o"

# External object files for target freenect-regview
freenect__regview_EXTERNAL_OBJECTS =

freenect-regview: CMakeFiles/freenect-regview.dir/regview.c.o
freenect-regview: CMakeFiles/freenect-regview.dir/build.make
freenect-regview: /usr/lib/x86_64-linux-gnu/libGLU.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libGL.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libSM.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libICE.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libX11.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libXext.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libglut.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libXmu.so
freenect-regview: /usr/lib/x86_64-linux-gnu/libXi.so
freenect-regview: CMakeFiles/freenect-regview.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C executable freenect-regview"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect-regview.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/freenect-regview.dir/build: freenect-regview
.PHONY : CMakeFiles/freenect-regview.dir/build

CMakeFiles/freenect-regview.dir/requires: CMakeFiles/freenect-regview.dir/regview.c.o.requires
.PHONY : CMakeFiles/freenect-regview.dir/requires

CMakeFiles/freenect-regview.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/freenect-regview.dir/cmake_clean.cmake
.PHONY : CMakeFiles/freenect-regview.dir/clean

CMakeFiles/freenect-regview.dir/depend:
	cd /home/scao/Desktop/kinview && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/scao/Desktop/kinview /home/scao/Desktop/kinview /home/scao/Desktop/kinview /home/scao/Desktop/kinview /home/scao/Desktop/kinview/CMakeFiles/freenect-regview.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/freenect-regview.dir/depend

