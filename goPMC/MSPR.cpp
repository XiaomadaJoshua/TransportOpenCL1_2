/*
 * MSPR.cpp
 *
 *  Created on: Oct 2, 2014
 *      Author: S158879
 */
#include "stdafx.h"

#include <fstream>
using namespace std;
#include "MSPR.h"
#include "OpenCLStuff.h"

MSPR::MSPR():nMaterial(0), nEnergy(0), startingHU(NULL), minEnergy(0), energyInterval(0), fs(NULL){
	// TODO Auto-generated constructor stub

}

MSPR::~MSPR() {
	// TODO Auto-generated destructor stub
	delete [] startingHU;
	delete [] fs;
}

bool MSPR::setData(const char * filename, const OpenCLStuff & stuff){
	ifstream ifs;
	nMaterial = 25;
	nEnergy = 350;
	minEnergy = 1.0f;
	energyInterval = 1.0f;
	startingHU = new float[nMaterial];
	fs = new float[nMaterial*nEnergy];

	ifs.open(filename, std::ios::binary | std::ios::in);
	ifs.read((char *)(startingHU), nMaterial*sizeof(float) / sizeof(char));
	ifs.read((char *)(fs), nMaterial*nEnergy*sizeof(float) / sizeof(char));
	ifs.close();

	cl::ImageFormat format(CL_R, CL_FLOAT);
	mspr = cl::Image2D(stuff.context, CL_MEM_READ_ONLY, format, nEnergy, nMaterial);
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = nEnergy;
	region[1] = nMaterial;
	region[2] = 1;
	stuff.queue.enqueueWriteImage(mspr, CL_TRUE, origin, region, 0, 0, fs);

	return true;
}

const float * MSPR::lookStartingHU() const{
	return startingHU;
}

unsigned int MSPR::lookNMaterial() const{
	return nMaterial;
}

