#pragma once
#include <CL/cl.hpp>
#include "Macro.h"

class OpenCLStuff
{
public:
	OpenCLStuff();
	virtual ~OpenCLStuff();
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Event event;
	cl_uint maxComputeUnits; 
	size_t maxWorkGroupSize;
	cl_uint3 maxWorkItemSizes;
	cl_ulong globalMemSize;

	static int convertToString(const char *filename, std::string& s);
	cl_uint nBatch();
};

