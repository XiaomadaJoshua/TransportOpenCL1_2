#include "stdafx.h"

#include "Proton.h"
#include "OpenCLStuff.h"
#include "Macro.h"
#include "Phantom.h"
#include "MacroCrossSection.h"
#include "MSPR.h"
#include "RSPW.h"

cl_float Proton::mass = MP;
cl_float Proton::charge = CP;

Proton::Proton(OpenCLStuff & stuff, cl_ulong nParticles_, cl_float T, cl_float2 width_, cl_float3 sourceCenter_) 
	: ParticleStatus(stuff, T, width_, sourceCenter_, nParticles_){
}

Proton::Proton(OpenCLStuff & stuff, cl_uint nHistory){
	ParticleStatus::buildProgram(stuff);
	nParticles = nHistory;
}

int Proton::reload(OpenCLStuff & stuff){
	if (nParticles == 0)
		return 0;
	if (nParticles > stuff.nBatch()){
		nParticles -= stuff.nBatch();
		load(stuff, stuff.nBatch(), energy, width, sourceCenter, mass, charge);
	}
	else{
		load(stuff, nParticles, energy, width, sourceCenter, mass, charge);
		nParticles = 0;
	}
	return 1;
}






Proton::~Proton()
{
}
