#include "ParticleStatus.h"
#include "OpenCLStuff.h"
#include "Secondary.h"
#include "Phantom.h"
#include "MacroCrossSection.h"
#include "RSPW.h"
#include "MSPR.h"
#include "Macro.h"
#include "OpenCLStuff.h"
#include <time.h>


ParticleStatus::ParticleStatus(){
}

ParticleStatus::ParticleStatus(OpenCLStuff & stuff, cl_float T, cl_float2 width_, cl_float3 sourceCenter_, cl_ulong nParticles_):energy(T), width(width_), sourceCenter(sourceCenter_), nParticles(nParticles_) {
	std::string source;
	OpenCLStuff::convertToString("ParticleStatus.cl", source);
	int err;
	program = cl::Program(stuff.context, source);
	err = program.build("-cl-single-precision-constant");
	std::string info;
	info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(stuff.device);
	err = program.createKernels(&particleKernels);
}



void ParticleStatus::load(OpenCLStuff & stuff, cl_ulong nParticles_, cl_float T, cl_float2 width, cl_float3 sourceCenter_, cl_float mass, cl_float charge)
{
	stuff.queue.finish();
	particleStatus.clear();
	int err;
	particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS) * nParticles_, NULL, &err));

//	PS * particleTest = new PS[nParticles_];
//	err = stuff.queue.enqueueReadBuffer(particleStatus[0], CL_TRUE, 0, sizeof(PS) * nParticles_, particleTest);
//	int tempSize = sizeof(PS);

	cl::make_kernel<cl::Buffer &, cl_float, cl_float2, cl_float3, cl_float, cl_float, cl_int> initParticlesKernel(program, "initParticles", &err);

	globalRange = cl::NDRange(nParticles_);
	cl::EnqueueArgs arg (stuff.queue, globalRange);
	srand((unsigned int)time(NULL));
	cl_int randSeed = rand();
	initParticlesKernel(arg, particleStatus[0], T, width, sourceCenter_, mass, charge, randSeed);
}


ParticleStatus::~ParticleStatus(){
}

int ParticleStatus::reload(OpenCLStuff & stuff){
	if (nParticles == 0)
		return 0;
	if (nParticles > stuff.nBatch()){
		nParticles -= stuff.nBatch();
		load(stuff, stuff.nBatch(), energy, width, sourceCenter, MP, CP);
	}
	else{
		load(stuff, nParticles, energy, width, sourceCenter, MP, CP);
		nParticles = 0;
	}
	return 1;
}


void ParticleStatus::propagate(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma, 
	RSPW * resStpPowWater, MSPR * massStpPowRatio, ParticleStatus * secondary){
	int err;

	cl::EnqueueArgs arg(stuff.queue, globalRange);
	/*
	particleKernels[1].setArg(0, particleStatus[0]);
	particleKernels[1].setArg(1, phantom->doseCounterGPU());
	particleKernels[1].setArg(2, phantom->voxelGPU());
	particleKernels[1].setArg(3, phantom->voxelSize());
	particleKernels[1].setArg(4, macroSigma->gpu());
	particleKernels[1].setArg(5, resStpPowWater->gpu());
	particleKernels[1].setArg(6, massStpPowRatio->gpu());
	particleKernels[1].setArg(7, secondary->particleStatus[0]);
	particleKernels[1].setArg(8, randSeed);

	stuff.queue.enqueueNDRangeKernel(particleKernels[1], 0, globalRange);
	*/
	
	cl::make_kernel < cl::Buffer &, cl::Buffer &, cl::Image3D &, cl_float3, cl::Image1D &, cl::Image1D &, cl::Image2D &, cl::Buffer &, cl::Buffer &, cl_int> propagateKernel(program, "propagate", &err);

	stuff.queue.finish();
	time_t timer;
	srand((unsigned int)time(NULL));
	cl_int randSeed = rand();
	propagateKernel(arg, particleStatus.back(), phantom->doseCounterGPU(), phantom->voxelGPU(), phantom->voxelSize(), macroSigma->gpu(),
		resStpPowWater->gpu(), massStpPowRatio->gpu(), secondary->particleStatus[0], secondary->nSecondBuffer(), randSeed);
	stuff.queue.finish();
}
