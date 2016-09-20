#pragma once
#include "cl.hpp"
#include <vector>
#include "Macro.h"
class OpenCLStuff;
class Phantom;
class MacroCrossSection;
class RSPW;
class MSPR;
class Secondary;



#if(__LINUX__ == 1)
struct PS{
	cl_float3 pos, dir;
	cl_float energy, maxSigma, mass, charge, weight;
};
#else
// use this for visual studio
struct __declspec(align(16)) PS{
	cl_float3 pos, dir;
	cl_float energy, maxSigma, mass, charge, weight;
}; 
#endif



class ParticleStatus
{
public:
	ParticleStatus();
	ParticleStatus(OpenCLStuff & stuff, cl_float T, cl_float2 width, cl_float3 sourceCenter_, cl_ulong nParticles_);
	void buildProgram(OpenCLStuff & stuff);
	void load(OpenCLStuff & stuff, cl_ulong nParticles_, cl_float T, cl_float2 width, cl_float3 sourceCenter, cl_float mass, cl_float charge);
	void load(OpenCLStuff & stuff, cl_uint nParticles_, cl_uint offset, cl_float * T, cl_float3 * pos, cl_float3 * dir, cl_float * weight, cl_float3 translation);
	~ParticleStatus();
	cl::Buffer & currentBeam() { return particleStatus[0]; }

	cl_ulong nParticlesLeft(){ return nParticles; }
	virtual int reload(OpenCLStuff & stuff);
	virtual int reload(OpenCLStuff & stuff, cl_float * T, cl_float3 * pos, cl_float3 * dir, cl_float * weight, cl_uint nHistory, cl_float3 translation);
	virtual void propagate(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma, 
		RSPW * resStpPowWater, MSPR * massStpPowRatio, ParticleStatus * secondary, cl_int scoringQuantity);
	virtual cl::Buffer & nSecondBuffer() = 0;
	virtual void clear(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma,
		RSPW * resStpPowWater, MSPR * massStpPowRatio) {}

protected:
	std::vector<cl::Buffer> particleStatus;
	cl::Program program;
	cl::NDRange globalRange;
	std::vector<cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl_float3>> loadSourceKernel;
	std::vector<cl::make_kernel < cl::Buffer &, cl::Buffer &, cl::Image3D &, cl_float3, cl::Image2D &, cl::Image2D &, cl::Image2D &, cl::Buffer &, cl::Buffer &, cl_int, cl::Buffer &, cl_int>> propagateKernel;
	

	cl_float energy;
	cl_float2 width;
	cl_float3 sourceCenter;
	cl_uint nParticles;
};

