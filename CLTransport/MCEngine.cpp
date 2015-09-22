#include "MCEngine.h"
#include <fstream>
#include <iostream>
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
#include <stdio.h>
#include <ctime>



MCEngine::MCEngine(const char * file){
	// TODO Auto-generated constructor stub
	// initialize from file
	std::cout << "initialize from file: " << file << std::endl;
	ifstream ifs(file, fstream::in);
	char buff[300];
	string directory;
	
	// initialize physics data
	std::cout << "initialize physics data" << std::endl;
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
	std::cout << "initialize protons" << std::endl;
	ifs.getline(buff, 300);
	float energy;
	cl_float2 bWidth;
	cl_float3 bSource;
	ifs >> bWidth.s[0] >> ws >> bWidth.s[1] >> ws >> energy >> ws >> nPaths >> ws;
	ifs >> bSource.s[0] >> bSource.s[1] >> bSource.s[2] >> ws;
	primary = new Proton(stuff, nPaths, energy, bWidth, bSource);

	//initialize phantom
	std::cout << "initialize phantom" << std::endl;
	ifs.getline(buff, 300);
	cl_float3 voxSize, phantomIso;
	ifs >> voxSize.s[0] >> voxSize.s[1] >> voxSize.s[2] >> ws;
	ifs >> phantomIso.s[0] >> phantomIso.s[1] >> phantomIso.s[2] >> ws;
	cl_int3 size;
	ifs >> size.s[0] >> size.s[1] >> size.s[2] >> ws;
	ifs.getline(buff, 300);
	std::string CTFile;
	getline(ifs, CTFile);
	if (CTFile.find("null") != string::npos)
		phantom = new Phantom(stuff, voxSize, size, *densCorrection, *mspr, NULL);
	else
		phantom = new Phantom(stuff, voxSize, size, *densCorrection, *mspr, CTFile.c_str());


	//initialize secondary particles container
	secondary = new Secondary(stuff);

	// output directory
	ifs.getline(buff, 300);
	getline(ifs, outDir);

	ifs.close();
	
	std::cout << "initialize finished" << std::endl;
}


void MCEngine::simulate(float minEnergy){
	std::clock_t start = std::clock();
	double duration;
	int i = 0;
	while (primary->nParticlesLeft() != 0){
		primary->reload(stuff);
		primary->propagate(stuff, phantom, macroSigma, rspw, mspr, secondary);
		std::cout << "simulate primary protons in batch " << i+1 << std::endl;
		secondary->propagate(stuff, phantom, macroSigma, rspw, mspr, secondary);
		std::cout << "simulate secondary protons in batch " << i+1 << std::endl;
		phantom->tempStore(stuff);
		i++;
	}
	secondary->clear(stuff, phantom, macroSigma, rspw, mspr);
	phantom->tempStore(stuff);
	phantom->finalize(stuff, nPaths);
	
	duration = (std::clock() - start)/(double)CLOCKS_PER_SEC;
	phantom->output(stuff, outDir);
	
	std::cout << "number of batches: " << i << std::endl;
	std::cout << "total simulation time: " << duration << std::endl;
}


MCEngine::~MCEngine()
{
}
