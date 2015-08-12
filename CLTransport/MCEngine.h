#pragma once
#include "cl.hpp"
#include <string>
#include "OpenCLStuff.h"
#include "Macro.h"
class OpenCLStuff;
class Phantom;
class ParticleStatus;
class Proton;
class Secondary;
class DensCorrection;
class MacroCrossSection;
class MSPR;
class RSPW;


class MCEngine
{
public:
	MCEngine(const char *);
	virtual ~MCEngine();
	void simulate(float minEnergy);
	void output();
private:
	ParticleStatus * primary, *secondary;
	Phantom * phantom;
	MacroCrossSection * macroSigma;
	MSPR * mspr;
	RSPW * rspw;
	DensCorrection * densCorrection;
	std::string outDir;
	OpenCLStuff  stuff;
	
	cl_uint nPaths;
};

