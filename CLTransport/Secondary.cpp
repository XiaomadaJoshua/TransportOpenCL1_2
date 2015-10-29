#include "Secondary.h"
#include "OpenCLStuff.h"
#include "Macro.h"
#include <iostream>


Secondary::Secondary(OpenCLStuff & stuff)
{
	buildProgram(stuff);
	size = SECONDARYNUMBERRATIO * stuff.nBatch();
	int err;
	particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS)*size, NULL, &err));
	nSecondary = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
	err = stuff.queue.enqueueWriteBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &size);
}

void Secondary::propagate(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma,
	RSPW * resStpPowWater, MSPR * massStpPowRatio, ParticleStatus * secondary){

	stuff.queue.enqueueReadBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &nParticles);
	while (size - nParticles > stuff.nBatch()){
		std::cout << "simulate secondary" << std::endl;
		particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS)*stuff.nBatch()));
		stuff.queue.enqueueCopyBuffer(particleStatus[0], particleStatus[1], sizeof(PS)*nParticles, 0, sizeof(PS)*stuff.nBatch());		
		nParticles += stuff.nBatch();
		stuff.queue.enqueueWriteBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &nParticles);
		globalRange = cl::NDRange(stuff.nBatch());
		ParticleStatus::propagate(stuff, phantom, macroSigma, resStpPowWater, massStpPowRatio, this);
		particleStatus.pop_back();
		stuff.queue.enqueueReadBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &nParticles);
	}

}

void Secondary::clear(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma,
	RSPW * resStpPowWater, MSPR * massStpPowRatio){
	stuff.queue.enqueueReadBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &nParticles);
	int err;
	std::cout << "clear secondary" << std::endl;
	while (nParticles != size){
		particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS)*(size - nParticles), NULL, &err));
		err = stuff.queue.enqueueCopyBuffer(particleStatus[0], particleStatus[1], sizeof(PS)*nParticles, 0, sizeof(PS)*(size - nParticles));

		stuff.queue.finish();

		PS * particleTest = new PS[size - nParticles];
		err = stuff.queue.enqueueReadBuffer(particleStatus[0], CL_TRUE, sizeof(PS)*(nParticles), sizeof(PS)*(size - nParticles), particleTest);
		int tempSize = sizeof(PS);

		stuff.queue.enqueueWriteBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &size);
		globalRange = cl::NDRange(size - nParticles);
		ParticleStatus::propagate(stuff, phantom, macroSigma, resStpPowWater, massStpPowRatio, this);
		particleStatus.pop_back();
		stuff.queue.enqueueReadBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &nParticles);
	}

}


Secondary::~Secondary()
{
}
