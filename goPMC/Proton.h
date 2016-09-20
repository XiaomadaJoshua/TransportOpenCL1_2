#pragma once
#include "cl.hpp"
#include "ParticleStatus.h"
class OpenCLStuff;
class MacroCrossSection;
class RSPW;
class MSPR;
class Phantom;

class Proton : public ParticleStatus
{
public:
	Proton(OpenCLStuff & stuff, cl_ulong nParticles_, cl_float T, cl_float2 width, cl_float3 sourceCenter_);
	Proton(OpenCLStuff &, cl_uint);
	virtual ~Proton();
	int reload(OpenCLStuff & stuff);

	// never used, real one is in secondary class
	virtual cl::Buffer & nSecondBuffer(){ return particleStatus[1]; }
private:
	static cl_float mass, charge;
};

