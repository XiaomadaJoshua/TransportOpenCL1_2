#include <vector>
#include <string>	
#include <iostream>
//#include "cl.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(){
/*
	std::vector<cl::Platform> platforms;
	std::string versionInfo;
	cl::Platform::get(&platforms);
	std::cout << platforms.size() << std::endl;
	for(int i = 0; i < platforms.size(); i++){
		int err;
		err = platforms[i].getInfo(CL_PLATFORM_VERSION, &versionInfo);
		std::cout << versionInfo << std::endl;
	}*/
	
	int cudaVersion, driverVersion;
	cudaRuntimeGetVersion(&cudaVersion);
	cudaDriverGetVersion(&driverVersion);
	
	printf(" cuda runtime version: %d\n cuda driver version: %d\n", cudaVersion, driverVersion);
	return 0;
}


/*	std::vector<cl::Device> devs;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devs);
	device = devs[0];
	context = cl::Context(device);
	queue = cl::CommandQueue(context, device);
	int err;
	err = device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
	err = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
	err = device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSizes);
	err = device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &globalMemSize);
*/

