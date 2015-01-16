#include "Secondary.h"
#include "OpenCLStuff.h"
#include "Macro.h"


Secondary::Secondary(OpenCLStuff & stuff)
{
	size = SECONDARYNUMBERRATIO * stuff.nBatch();
	int err;
	particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS)*size, NULL, &err));
	nSecondary = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_uint));
	stuff.queue.enqueueWriteBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_int), &size);
}

void Secondary::propagate(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma,
	RSPW * resStpPowWater, MSPR * massStpPowRatio, ParticleStatus * secondary){

	stuff.queue.enqueueReadBuffer(nSecondary, CL_TRUE, 0, sizeof(cl_uint), &nParticles);
	while (size - nParticles > stuff.nBatch()){
		particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS)*stuff.nBatch()));
		stuff.queue.enqueueCopyBuffer(particleStatus[0], particleStatus[1], nParticles, 0, sizeof(PS)*stuff.nBatch());
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
	while (nParticles != size){
		particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS)*(size - nParticles)));
		stuff.queue.enqueueCopyBuffer(particleStatus[0], particleStatus[1], nParticles, 0, sizeof(PS)*(size - nParticles));
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
