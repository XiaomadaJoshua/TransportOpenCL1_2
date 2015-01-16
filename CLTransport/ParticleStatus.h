#pragma once
#include <CL/cl.hpp>
#include <vector>
#include "Macro.h"
class OpenCLStuff;
class Phantom;
class MacroCrossSection;
class RSPW;
class MSPR;
class Secondary;

struct PS{
	cl_float3 pos, dir;
	cl_float energy, maxSigma, mass, charge;
};

class ParticleStatus
{
public:
	ParticleStatus();
	ParticleStatus(OpenCLStuff & stuff, cl_float T, cl_float2 width, cl_float3 sourceCenter_, cl_ulong nParticles_);
	void load(OpenCLStuff & stuff, cl_ulong nParticles, cl_float T, cl_float2 width, cl_float3 sourceCenter, cl_float mass, cl_float charge);
	~ParticleStatus();
	cl::Buffer & currentBeam() { return particleStatus[0]; }

	cl_ulong nParticlesLeft(){ return nParticles; }
	virtual int reload(OpenCLStuff & stuff);
	virtual void propagate(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma, 
		RSPW * resStpPowWater, MSPR * massStpPowRatio, ParticleStatus * secondary);
	virtual cl::Buffer & nSecondBuffer() = 0;
	virtual void clear(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma,
		RSPW * resStpPowWater, MSPR * massStpPowRatio) {}

protected:
	std::vector<cl::Buffer> particleStatus;
	std::vector<cl::Kernel> particleKernels;
	cl::Program program;
	cl::NDRange globalRange;

	cl_float energy;
	cl_float2 width;
	cl_float3 sourceCenter;
	cl_uint nParticles;
};

