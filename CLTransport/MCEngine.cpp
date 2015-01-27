#include "MCEngine.h"
#include <fstream>
#include <string>
#include "Macro.h"
#include "MacroCrossSection.h"
#include "MSPR.h"
#include "RSPW.h"
#include "DensCorrection.h"
#include "ParticleStatus.h"
#include "Proton.h"
#include "Secondary.h"
#include "Phantom.h"
#include "OpenCLStuff.h"





MCEngine::MCEngine(const char * file){
	// TODO Auto-generated constructor stub
	// initialize from file
	ifstream ifs(file, fstream::in);
	char buff[300];
	string directory;

	// initialize physics data
	ifs.getline(buff, 300);
	getline(ifs, directory);
	macroSigma = new MacroCrossSection();
	macroSigma->setData(directory.c_str(), stuff);

	ifs.getline(buff, 300);
	getline(ifs, directory);
	mspr = new MSPR();
	mspr->setData(directory.c_str(), stuff);

	ifs.getline(buff, 300);
	getline(ifs, directory);
	rspw = new RSPW();
	rspw->setData(directory.c_str(), stuff);

	ifs.getline(buff, 300);
	getline(ifs, directory);
	densCorrection = new DensCorrection();
	densCorrection->setData(directory.c_str(), stuff);


	// initialize protons
	ifs.getline(buff, 300);
	float energy;
	cl_float2 bWidth;
	unsigned long nPaths;
	cl_float3 bSource;
	ifs >> bWidth.s[0] >> ws >> bWidth.s[1] >> ws >> energy >> ws >> nPaths >> ws;
	ifs >> bSource.s[0] >> bSource.s[1] >> bSource.s[2] >> ws;
	primary = new Proton(stuff, nPaths, energy, bWidth, bSource);

	//initialize phantom
	ifs.getline(buff, 300);
	cl_float3 voxSize, phantomIso;
	ifs >> voxSize.s[0] >> voxSize.s[1] >> voxSize.s[2] >> ws;
	ifs >> phantomIso.s[0] >> phantomIso.s[1] >> phantomIso.s[2] >> ws;
	cl_int3 size;
	ifs >> size.s[0] >> size.s[1] >> size.s[2] >> ws;
	phantom = new Phantom(stuff, voxSize, size, *densCorrection, *mspr);

	//initialize secondary particles container
	secondary = new Secondary(stuff);

	// output directory
	ifs.getline(buff, 300);
	getline(ifs, outDir);
}


void MCEngine::simulate(float minEnergy){
	while (primary->nParticlesLeft() != 0){
		primary->reload(stuff);
		primary->propagate(stuff, phantom, macroSigma, rspw, mspr, secondary);
		secondary->propagate(stuff, phantom, macroSigma, rspw, mspr, secondary);
	}
	secondary->clear(stuff, phantom, macroSigma, rspw, mspr);
//	phantom->finalize(stuff);
	phantom->output(stuff, outDir);
}


MCEngine::~MCEngine()
{
}
