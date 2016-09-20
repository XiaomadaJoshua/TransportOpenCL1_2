#include "stdafx.h"

#include "OpenCLStuff.h"
#include <vector>
#include <fstream>
#include <iostream>
#include "ParticleStatus.h"
#include <string>

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

OpenCLStuff::OpenCLStuff()
{
	int err;
	std::string info;
	cl::Platform::get(&platform);
	err = platform.getInfo(CL_PLATFORM_NAME, &info);
	std::cout << "CL_PLATFORM_NAME:\t" << info << std::endl;
	std::vector<cl::Device> devs;
#if(__IFGPU__ == 1)
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devs);
#else
	platform.getDevices(CL_DEVICE_TYPE_CPU, &devs);
#endif
	device = devs[0];
	err = device.getInfo(CL_DEVICE_NAME, &info);
	std::cout << "CL_DEVICE_NAME:\t" << info << std::endl;
	context = cl::Context(device);
	queue = cl::CommandQueue(context, device);
	int clockFrequency;

	err = device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
	std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS:\t" << maxComputeUnits << std::endl;
	err = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
	err = device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSizes);
	err = device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &globalMemSize);
//	err = device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);

	err = device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &clockFrequency);

}

OpenCLStuff::OpenCLStuff(cl::Platform & platform_, cl::Device & device_)
	:platform(platform_), device(device_){
	int err;
	std::string info;

	err = platform.getInfo(CL_PLATFORM_NAME, &info);
	std::cout << "CL_PLATFORM_NAME:\t" << info << std::endl;

	err = device.getInfo(CL_DEVICE_NAME, &info);
	std::cout << "CL_DEVICE_NAME:\t" << info << std::endl;

	context = cl::Context(device);
	queue = cl::CommandQueue(context, device);

	err = device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);
	err = device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxMecAllocSize);
}

cl_uint OpenCLStuff::nBatch(){
	return 65536;
	return globalMemSize / (sizeof(PS)*SECONDARYNUMBERRATIO*2);
}



OpenCLStuff::~OpenCLStuff()
{
}


int OpenCLStuff::convertToString(const char * filename, std::string & s)
{
	size_t size;
	char*  str;

	// create a file stream object by filename
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));


	if (!f.is_open())
	{
		return SDK_FAILURE;
	}
	else
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return SDK_FAILURE;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';

		s = str;
		delete[] str;
		return SDK_SUCCESS;
	}
}

/*
int OpenCLStuff::convertToSource(const char *filename, cl::Program::Sources & source){
	std::string sourceStr;
	convertToString(filename, sourceStr);
	source = cl::Program::Sources(1,std::make_pair(sourceStr.c_str(), sourceStr.length()+1));
	
	return SDK_SUCCESS;
}*/
