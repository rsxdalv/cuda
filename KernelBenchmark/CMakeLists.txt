# CMakeLists.txt for CUDA
cmake_minimum_required( VERSION 2.8.0 )

# FindCUDA.cmake Fix, since CMake defaults to "" 
set( CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/" )

find_package(CUDA QUIET REQUIRED)

# nvcc options
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2 
-gencode arch=compute_20,code=sm_20 
-gencode arch=compute_30,code=sm_30 
-gencode arch=compute_35,code=sm_35 
-gencode arch=compute_37,code=sm_37 
-gencode arch=compute_50,code=sm_50 
)

# avoiding nvcc -rdynamic flag
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

# indicate include dir
include_directories(
/usr/local/cuda/include
/usr/local/cuda/samples/common/inc
) 

# indicate library dir 
link_directories(
/usr/local/cuda/lib64		
)


# Indicate target & source files to compile
cuda_add_executable(
MatrixMult # target
main.cu # libraries
)
	 
# Indicate target & libraries to link
target_link_libraries(
MatrixMult # target
-lcudart # libraries
) 
