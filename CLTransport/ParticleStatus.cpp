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
#include <iostream>

ParticleStatus::ParticleStatus(){
}

ParticleStatus::ParticleStatus(OpenCLStuff & stuff, cl_float T, cl_float2 width_, cl_float3 sourceCenter_, cl_ulong nParticles_)
	:energy(T), width(width_), sourceCenter(sourceCenter_), nParticles(nParticles_) {
	buildProgram(stuff);
}

void ParticleStatus::buildProgram(OpenCLStuff & stuff){

	std::string source;
	OpenCLStuff::convertToString("ParticleStatus.cl", source);
	std::string include1, include2;
	OpenCLStuff::convertToString("Macro.h", include1);
	OpenCLStuff::convertToString("randomKernel.h", include2);
	source = include1 + include2 + source;
	
	int err;
	program = cl::Program(stuff.context, source);
	err = program.build("-cl-single-precision-constant -I.");
	std::string info;
	info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(stuff.device);
	if (err != 0){
		std::cout << "build result: " << err << std::endl;
		std::cout << info << std::endl;
	}

}



void ParticleStatus::load(OpenCLStuff & stuff, cl_ulong nParticles_, cl_float T, cl_float2 width, cl_float3 sourceCenter_, cl_float mass, cl_float charge)
{
	particleStatus.clear();
	int err;
	int tempSize = sizeof(PS);
	particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS) * nParticles_, NULL, &err));

	cl::make_kernel<cl::Buffer &, cl_float, cl_float2, cl_float3, cl_float, cl_float, cl_int> initParticlesKernel(program, "initParticles");

	globalRange = cl::NDRange(nParticles_);
	cl::EnqueueArgs arg (stuff.queue, globalRange);
	cl_int randSeed = rand();
	initParticlesKernel(arg, particleStatus[0], T, width, sourceCenter_, mass, charge, randSeed);

//	PS * particleTest = new PS[nParticles_]();
//	err = stuff.queue.enqueueReadBuffer(particleStatus[0], CL_TRUE, 0, sizeof(PS) * nParticles_, particleTest);
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
	
	cl::make_kernel < cl::Buffer &, cl::Buffer &, cl::Image3D &, cl_float3, cl::Image2D &, cl::Image2D &, cl::Image2D &, cl::Buffer &, cl::Buffer &, cl_int, cl::Buffer &> propagateKernel(program, "propagate", &err);	
	
	cl_int randSeed = rand();
//	std::cout << randSeed << std::endl;
	cl::Buffer mutex(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
	cl_int initialMutext = 0;
	stuff.queue.enqueueWriteBuffer(mutex, CL_TRUE, 0, sizeof(cl_int), &initialMutext);

	stuff.queue.finish();
	propagateKernel(arg, particleStatus.back(), phantom->doseCounterGPU(), phantom->voxelGPU(), phantom->voxelSize(), macroSigma->gpu(),
		resStpPowWater->gpu(), massStpPowRatio->gpu(), secondary->particleStatus[0], secondary->nSecondBuffer(), randSeed, mutex);
	std::cout << "number of protons in this batch: " << *globalRange << std::endl;
	stuff.queue.finish();

}
