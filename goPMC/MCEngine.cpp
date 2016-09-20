#include "stdafx.h"

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
#include "OpenCLStuff.h"

const std::map<std::string, cl_int> goPMC::MCEngine::types{
		{ "DOSE2MEDIUM", 0 },
		{ "DOSE2WATER", 1 },
		{ "FLUENCE", 2 },
		{ "LETD", 3 }
};

const std::map<std::string, std::string> Unit{
		{ "DOSE2MEDIUM", "MeV/g/primary" },
		{ "DOSE2WATER", "MeV/g/primary" },
		{ "FLUENCE", "cm^-2/primary"},
		{ "LETD", "MeV^2cm^3/g^2/primary" }
};


goPMC::MCEngine::MCEngine():nPaths(0),totalWeight(0.0){
	start = std::clock();

}

void goPMC::MCEngine::initializeComputation(cl::Platform & platform, cl::Device & device){
	stuff = new OpenCLStuff(platform, device);
	secondary = new Secondary(*stuff);
}

void goPMC::MCEngine::initializePhysics(const std::string & dir){
	std::string file = dir + "/MCS.bin";
	macroSigma = new MacroCrossSection();
	macroSigma->setData(file.c_str(), *stuff);

	file = dir + "/MSPR.bin";
	mspr = new MSPR();
	mspr->setData(file.c_str(), *stuff);

	file = dir + "/RSPW.bin";
	rspw = new RSPW();
	rspw->setData(file.c_str(), *stuff);

	file = dir + "/DC.bin";
	densCorrection = new DensCorrection();
	densCorrection->setData(file.c_str(), *stuff);

	std::cout << "Physics data initialization finished.\n";
}

void goPMC::MCEngine::initializePhantom(const std::string & dicomDir){
	phantom = new Phantom(*stuff, *densCorrection, *mspr, dicomDir);

	std::cout << "Phantom initialization finished.\n";
}

void goPMC::MCEngine::simulate(cl_float * T, cl_float3 * pos, cl_float3 * dir, cl_float * weight, cl_uint nHistory, std::string & quantity_){
	std::cout << "Simulation starts." << '\n' << "Scoring quantity is " << quantity_ << std::endl;
	quantity = quantity_;
	nPaths += nHistory;
	for (cl_uint i = 0; i < nHistory; i++)
		totalWeight += weight[i];
	cl_float3 phantomSize;
	phantomSize.s[0]= phantom->phantomResolution().s[0] * phantom->voxelSize().s[0];
	phantomSize.s[1] = phantom->phantomResolution().s[1] * phantom->voxelSize().s[1];
	phantomSize.s[2] = phantom->phantomResolution().s[2] * phantom->voxelSize().s[2];

	cl_float3 translation;
	translation.s[0] = (-phantomSize.s[0] + phantom->voxelSize().s[0])*0.5 - phantom->patientOffSet().s[0];
	translation.s[1] = (-phantomSize.s[1] + phantom->voxelSize().s[1])*0.5 - phantom->patientOffSet().s[1];
	translation.s[2] = (-phantomSize.s[2] + phantom->voxelSize().s[2])*0.5 - phantom->patientOffSet().s[2];

	primary = new Proton(*stuff, nHistory);

	srand((unsigned int)time(NULL));
	int i = 0;
	while (primary->nParticlesLeft() != 0){
		primary->reload(*stuff, T, pos, dir, weight, nHistory, translation);
		std::cout << std::endl << "simulation batch " << i + 1 << std::endl;
		primary->propagate(*stuff, phantom, macroSigma, rspw, mspr, secondary, types.at(quantity));
		secondary->propagate(*stuff, phantom, macroSigma, rspw, mspr, secondary, types.at(quantity));
		phantom->tempStore(*stuff);
		i++;
	}

	std::cout << "\ntotal number of batches: " << i << std::endl;
}

void goPMC::MCEngine::clearCounter(){
	phantom->clearCounter(*stuff);

	double duration;
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "total number of source particle simulated: " << nPaths << std::endl;
	std::cout << "total weight: " << totalWeight << std::endl;
	std::cout << "total simulation time: " << duration << "seconds." << std::endl;
	nPaths = 0;
	totalWeight = 0.0;
}

void goPMC::MCEngine::getResult(std::vector<cl_float> & doseMean, std::vector<cl_float> & doseStd){
	secondary->clear(*stuff, phantom, macroSigma, rspw, mspr);
	phantom->tempStore(*stuff);
	phantom->finalize(*stuff, totalWeight, types.at(quantity));

	std::cout << "Returning simulation results of" << quantity << ".\t Unit: " << Unit.at(quantity) << std::endl;
	phantom->output(*stuff, doseMean, doseStd);


}

/*
goPMC::MCEngine::MCEngine(const char * file){
	// TODO Auto-generated constructor stub
	// initialize from file
	stuff = new OpenCLStuff();
	std::cout << "initialize from file: " << file << std::endl;
	ifstream ifs(file, fstream::in);
	char buff[300];
	string directory;
	
	// initialize physics data
	std::cout << "initialize physics data" << std::endl;
	ifs.getline(buff, 300);
	getline(ifs, directory);
	macroSigma = new MacroCrossSection();
	macroSigma->setData(directory.c_str(), *stuff);

	ifs.getline(buff, 300);
	getline(ifs, directory);
	mspr = new MSPR();
	mspr->setData(directory.c_str(), *stuff);

	ifs.getline(buff, 300);
	getline(ifs, directory);
	rspw = new RSPW();
	rspw->setData(directory.c_str(), *stuff);

	ifs.getline(buff, 300);
	getline(ifs, directory);
	densCorrection = new DensCorrection();
	densCorrection->setData(directory.c_str(), *stuff);


	// initialize protons
	std::cout << "initialize protons" << std::endl;
	ifs.getline(buff, 300);
	float energy;
	cl_float2 bWidth;
	cl_float3 bSource;
	ifs >> bWidth.s[0] >> ws >> bWidth.s[1] >> ws >> energy >> ws >> nPaths >> ws;
	ifs >> bSource.s[0] >> bSource.s[1] >> bSource.s[2] >> ws;
	primary = new Proton(*stuff, nPaths, energy, bWidth, bSource);

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
		phantom = new Phantom(*stuff, voxSize, size, *densCorrection, *mspr, NULL);
	else
		phantom = new Phantom(*stuff, voxSize, size, *densCorrection, *mspr, CTFile.c_str());


	//initialize secondary particles container
	secondary = new Secondary(*stuff);

	// output directory
	ifs.getline(buff, 300);
	getline(ifs, outDir);

	ifs.close();
	
	std::cout << "initialize finished" << std::endl;
}


void goPMC::MCEngine::simulate(float minEnergy){
	srand((unsigned int)time(NULL));
	std::clock_t start = std::clock();
	double duration;
	int i = 0;
	while (primary->nParticlesLeft() != 0){
		primary->reload(*stuff);
		std::cout << std::endl << "simulation batch " << i + 1 << std::endl;
		primary->propagate(*stuff, phantom, macroSigma, rspw, mspr, secondary);
		secondary->propagate(*stuff, phantom, macroSigma, rspw, mspr, secondary);
		phantom->tempStore(*stuff);
		i++;
	}
	secondary->clear(*stuff, phantom, macroSigma, rspw, mspr);
	phantom->tempStore(*stuff);
	phantom->finalize(*stuff, nPaths);
	
	duration = (std::clock() - start)/(double)CLOCKS_PER_SEC;
	phantom->output(*stuff, outDir);
	
	std::cout << "number of batches: " << i << std::endl;
	std::cout << "total simulation time: " << duration << std::endl;
}
*/

goPMC::MCEngine::~MCEngine()
{
	delete primary; 
	delete secondary;
	delete phantom;
	delete macroSigma;
	delete mspr;
	delete rspw;
	delete densCorrection;
	delete stuff;
}
