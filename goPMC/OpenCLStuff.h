#pragma once
#include "cl.hpp"
#include "Macro.h"
#include <vector>

class OpenCLStuff
{
public:
	OpenCLStuff();
	OpenCLStuff(cl::Platform &, cl::Device &);
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
	cl_ulong maxMecAllocSize;

	static int convertToString(const char *filename, std::string& s);
	cl_uint nBatch();
};

