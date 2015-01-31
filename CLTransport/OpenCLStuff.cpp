#include "OpenCLStuff.h"
#include <vector>
#include <fstream>
#include "ParticleStatus.h"

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

OpenCLStuff::OpenCLStuff()
{
	cl::Platform::get(&platform);
	std::vector<cl::Device> devs;
	platform.getDevices(CL_DEVICE_TYPE_CPU, &devs);
	device = devs[0];
	context = cl::Context(device);
	queue = cl::CommandQueue(context, device);
	int err;

	int cockFrequency;

	err = device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
	err = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
	err = device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSizes);
	err = device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &globalMemSize);
//	err = device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);

	err = device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &cockFrequency);

}

cl_uint OpenCLStuff::nBatch(){ 
	return globalMemSize / (sizeof(PS)*10); 
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
