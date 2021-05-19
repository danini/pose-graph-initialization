cmake_minimum_required(VERSION 3.8)

project(PoseGraphBuilder LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if GFLags should be enabled
option(USE_GFLAGS "Enable Google Flags." ON)

# indicate if GLog should be enabled
option(USE_GLOG "Enable Google Log." ON)

option(CREATE_SAMPLE_PROJECT "Building sample C++ project." ON)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV 4.0)

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing -fpermissive")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
  link_libraries(stdc++fs)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)


# ==============================================================================
# Find Eigen
# ==============================================================================
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# ==============================================================================
# Find GFlags
# ==============================================================================
find_package(gflags REQUIRED)
if(DEFINED gflags_VERSION_STRING OR DEFINED GFLAGS_VERSION_STRING)
	# different cmake variable with lower or upper case exist
	set(GFLAGS_INCLUDE_DIR ${GFLAGS_INCLUDE_DIR})
	set(GFLAGS_LIBRARIES ${GFLAGS_LIBRARIES})
  message( "Found Gflags ${GFLAGS_VERSION_STRING}" )
  message( STATUS "> GFLAGS_INCLUDE_DIR:   ${GFLAGS_INCLUDE_DIR}" )
  message( STATUS "> GFLAGS_LIBRARIES:   ${GFLAGS_LIBRARIES}" )
endif()

# ==============================================================================
# Find GLOG
# ==============================================================================
find_package(Glog REQUIRED)
if(NOT Glog_FOUND)
	message(FATAL_ERROR "Glog not found.")
else()
	message("Glog library found.")
endif()

if(IS_MSVC)
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()


# ==============================================================================
# Find Sophus
# ==============================================================================
find_package(Sophus)
message(STATUS "Search Dependency: Sophus")

if(Sophus_FOUND)
	if (TARGET Sophus::Sophus)
		get_target_property(Sophus_INCLUDE_DIR Sophus::Sophus INTERFACE_INCLUDE_DIRECTORIES)
	else()
		find_path(SOPHUS_INCLUDE_DIR "sophus/common.hpp")
	endif()
	include_directories(AFTER ${Sophus_INCLUDE_DIR})
else()
    message(WARNING "Using Sophus is set, but Sophus is NOT found. Sophus is used only in the pose estimation example project which will, thus, not be built.")
endif()

# ==============================================================================
# Find OpenMP
# ==============================================================================
find_package(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
          SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
          set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

endif (USE_OPENMP)

# ==============================================================================
# Includes
# ==============================================================================
SET(SOURCE_DIR "src/pyposegraphbuilder")

# ==============================================================================
# Structure: Pose Graph Builder Library
# ==============================================================================
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR}/include)

# Set header files for the library
file(GLOB_RECURSE HDRS_GRAPHBUILDER
	"${SOURCE_DIR}/include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GRAPHBUILDER
	"${SOURCE_DIR}/include/*.cpp"
)

add_library(${PROJECT_NAME} STATIC 
	${HDRS_GRAPHBUILDER} 
	${SRCS_GRAPHBUILDER})

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
	Eigen3::Eigen
	glog
	gflags
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(SampleProject
		examples/cpp_example.cpp)
		
	target_link_libraries(SampleProject PUBLIC
		${PROJECT_NAME}
		${OpenCV_LIBS}
		Eigen3::Eigen
		glog
		gflags
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)