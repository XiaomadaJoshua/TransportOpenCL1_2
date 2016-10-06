#pragma once
#include "cl.hpp"
#include"ParticleStatus.h"
class OpenCLStuff;

class Secondary : public ParticleStatus
{
public:
	Secondary(OpenCLStuff & stuff);
	virtual ~Secondary();
	cl::Buffer & nSecondBuffer(){ return nSecondary; }
	virtual void propagate(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma,
		RSPW * resStpPowWater, MSPR * massStpPowRatio, ParticleStatus * secondary, cl_int scoringQuantity);
	virtual void clear(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma,
		RSPW * resStpPowWater, MSPR * massStpPowRatio, cl_int scoringQuantity);
private:
	cl_uint size;
	cl::Buffer nSecondary;
};
