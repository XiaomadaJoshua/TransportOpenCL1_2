/*
 * MSPR.cpp
 *
 *  Created on: Oct 2, 2014
 *      Author: S158879
 */
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
	ifs.open(filename, ios::in);
	if(!ifs.is_open())
		return false;
	char temp[256];
	ifs.getline(temp, 256);
	ifs.getline(temp, 256);
	ifs >> nMaterial  >> ws;
	ifs.getline(temp, 256);
	ifs >> nEnergy >> ws;
	startingHU = new float[nMaterial];
	fs = new float[nMaterial*nEnergy];
	for(unsigned int j=0; j<nMaterial; j++){
		ifs.getline(temp, 256);
		int matID = j;
		float nominalDens, energy;
		ifs >> matID >> startingHU[matID] >> nominalDens >> nEnergy >> minEnergy >> energyInterval >> ws;
		ifs.getline(temp, 256);
		for(unsigned int i=0; i<nEnergy; i++)
			ifs >> energy >> fs[j*nEnergy + i] >> ws;
	}

	ifs.close();
	cl::ImageFormat format(CL_R, CL_FLOAT);
	mspr = cl::Image2D(stuff.context, CL_MEM_READ_ONLY, format, nEnergy, nMaterial);
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = nEnergy;
	region[1] = nMaterial;
	region[2] = 1;
	stuff.queue.enqueueWriteImage(mspr, CL_TRUE, origin, region, 0, 0, fs);
}

const float * MSPR::lookStartingHU() const{
	return startingHU;
}

unsigned int MSPR::lookNMaterial() const{
	return nMaterial;
}

