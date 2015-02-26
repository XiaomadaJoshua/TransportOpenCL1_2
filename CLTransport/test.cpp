#include "DensCorrection.h"
#include "OpenCLStuff.h"
#include "MacroCrossSection.h"
#include "RSPW.h"
#include "MSPR.h"
#include "Phantom.h"
#include "randomKernel.h"
#include "Proton.h"
#include "ParticleStatus.h"
#include "MCEngine.h"

#include <string>

/*
int test(){
	OpenCLStuff stuff;
	cl_bool temp;
	temp = stuff.device.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
	cl_uint maxComputeUnits, maxWorkGroupSize;
	cl_ulong globalMemSize;
	cl_uint3 maxWorkItemSize;
	stuff.device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
	stuff.device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
	stuff.device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &maxWorkItemSize);
	stuff.device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);

	cl::ImageFormat format(CL_R, CL_FLOAT);
	cl::Image1D image1d(stuff.context, CL_MEM_READ_WRITE, format, 1);
	cl::Image1DArray image1darray(stuff.context, CL_MEM_READ_ONLY, format, 1, 1, 0);

	DensCorrection DCF;
	DCF.setData("input/densityCorrection.dat", stuff);
	MacroCrossSection MCS;
	MCS.setData("input/mcpro.imfp", stuff);
	RSPW resSPW;
	resSPW.setData("input/mcpro.rstpw", stuff);
	MSPR massSPR;
	massSPR.setData("input/mcpro.mater", stuff);
	
	std::string source;
	OpenCLStuff::convertToString("testKernel.cl", source);
	cl::Program program;
	program = cl::Program(stuff.context, source);
	program.build("-cl-single-precision-constant");
	std::string info;
	info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(stuff.device);
	
	cl::make_kernel<cl::Image1D &, cl::Buffer &, cl::Buffer &> testKernel(program, "test");
	cl::make_kernel<cl::Image2D &, cl::Buffer &, cl::Buffer &> testKernel2D(program, "test2D");

	cl_float in[4] = { 0.5, 1.5, 2, MCS.size() - 1 + 0.5 };
	cl_float2 in2D[4]{ 
		0.5, 0.5,
		1.0, 0.5,
		1.5, 1.5,
		massSPR.size() - 1 + 0.5, massSPR.lookNMaterial() - 1 + 0.5
	};

	cl_float4 out[4];
	cl::Buffer input(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float) * 4);
	stuff.queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(cl_float) * 4, in);

	cl::Buffer input2D(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float2) * 4);
	stuff.queue.enqueueWriteBuffer(input2D, CL_TRUE, 0, sizeof(cl_float2) * 4, in2D);

	cl::NDRange ndrg(4);
	cl::EnqueueArgs arg(stuff.queue, ndrg);
	cl::Buffer output(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float4) * 4);
	testKernel2D(arg, massSPR.gpu(), output, input2D);
	stuff.queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(cl_float4) * 4, out);

	testKernel(arg, resSPW.gpu(), output, input);
	stuff.queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(cl_float4) * 4, out);

	cl_float3 voxSize = { 0.2f, 0.1f, 0.2f};
	cl_int3 phantomSize = { 100, 300, 100};
	Phantom phantom(stuff, voxSize, phantomSize, DCF, massSPR);

	cl::make_kernel<cl::Image3D &, cl::Buffer &, cl_float3, cl::Buffer &> testKernelPhantom(program, "testPhantom");
	cl::Buffer value(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float16) );
	cl::Buffer cordGPU(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float4));
	cl_float3 cord = { 30, 30, 30 };
	stuff.queue.enqueueWriteBuffer(cordGPU, CL_TRUE, 0, sizeof(cl_float4), &cord);

	cl::NDRange range(1);
	cl::EnqueueArgs arg1(stuff.queue, range);
	testKernelPhantom(arg1, phantom.voxelGPU(), phantom.doseCounterGPU(), cord, value);
	cl_float16 result;
	stuff.queue.enqueueReadBuffer(value, CL_TRUE, 0, sizeof(cl_float16), &result);

	cl::make_kernel<unsigned, cl::Buffer &> testPiKernel(program, "testPi");
	cl::Buffer hitsp(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_uint2)*1e5);
	testPiKernel(cl::EnqueueArgs(stuff.queue, cl::NDRange(1e5)), 1, hitsp);
	cl_uint2 * hitsHost = new cl_uint2[1e5];
	stuff.queue.enqueueReadBuffer(hitsp, CL_TRUE, 0, sizeof(cl_uint2)*1e5, hitsHost);

	unsigned hit = 0;
	unsigned tries = 0;
	for (int i = 0; i < 1e5; i++){
		hit += hitsHost[i].s[0];
		tries += hitsHost[i].s[1];
	}

	double pi = 4.0*hit / tries;


	float rand[2];
	size_t tempID = 0;
//	getUniform(tempID, rand);

	cl_float2 width;
	width.s[0] = 5.0f;
	width.s[1] = 5.0f;
	cl_float3 sourceCenter;
	sourceCenter.s[0] = 0.0f;
	sourceCenter.s[1] = -15.0f;
	sourceCenter.s[2] = 0.0f;
	Proton protonBeam(stuff, 1e5, 100.0f, width, sourceCenter);

	PS * beam = new PS[stuff.maxComputeUnits*stuff.maxWorkGroupSize];
	float *raw = new float[10000];
	stuff.queue.enqueueReadBuffer(protonBeam.currentBeam(), CL_TRUE, 0, sizeof(PS)*maxComputeUnits*maxWorkGroupSize, beam);
	stuff.queue.enqueueReadBuffer(protonBeam.currentBeam(), CL_TRUE, 0, sizeof(float)*maxComputeUnits*maxWorkGroupSize, raw);

	return 0;
}*/


int main(){
	int temp;
	MCEngine mc("proton_config");
	mc.simulate(MINPROTONENERGY);
	
//	std::cin >> temp;
}


/*int main(){
	OpenCLStuff stuff;
	cl_float2 width;
	cl_float3 center;
	width.s[0] = 0;
	width.s[1] = 0;
	center.s[0] = 0;
	center.s[1] = 0;
	center.s[2] = 0;

	Proton p(stuff, 2, 100, width, center);
	p.reload(stuff);
}*/