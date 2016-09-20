/*
 * RSPW.cpp
 *
 *  Created on: Oct 1, 2014
 *      Author: S158879
 */
#include "stdafx.h"

#include "RSPW.h"
#include <stdlib.h>
#include <fstream>
#include "OpenCLStuff.h"

using namespace std;


RSPW::RSPW():
minElectronEnergy(0), minProtonEnergy(0), maxProtonEnergy(0), energyInterval(0), nData(0), data(NULL){
	// TODO Auto-generated constructor stub

}

RSPW::~RSPW() {
	// TODO Auto-generated destructor stub
}

bool RSPW::setData(const char * filename, const OpenCLStuff & stuff){
	ifstream ifs;
	minElectronEnergy = 0.1f;
	minProtonEnergy = 0.5f;
	maxProtonEnergy = 350.0f;
	energyInterval = 0.5f;
	nData = 700;
	data = new Lwb[nData];

	ifs.open(filename, std::ios::binary | std::ios::in);
	ifs.read((char *)(data), nData*sizeof(Lwb) / sizeof(char));
	ifs.close();

	cl::ImageFormat format(CL_RG, CL_FLOAT);
	rspw = cl::Image2D(stuff.context, CL_MEM_READ_ONLY, format, nData, 1);
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = nData;
	region[1] = 1;
	region[2] = 1;
	stuff.queue.enqueueWriteImage(rspw, CL_TRUE, origin, region, 0, 0, data);
	delete[] data;

	return true;
}
/*

float RSPW::lookLw(float e) const{
	if(e < minProtonEnergy)
		return data[0].Lw;
	if(e > maxProtonEnergy)
		return data[nData - 1].Lw;
	int floor = static_cast<int>((e - minProtonEnergy)/energyInterval);
	int ceiling = floor + 1;
	float lw1 = data[floor].Lw;
	float lw2 = data[ceiling].Lw;
	float e1 = data[floor].energy;
	float e2 = data[ceiling].energy;
	return lw1 + (e - e1)*(lw2 - lw1)/(e2 - e1);
}

float RSPW::lookB(float e) const{
	if(e < minProtonEnergy)
		return data[0].b;
	if(e > maxProtonEnergy)
		return data[nData - 1].b;
	int floor = static_cast<int>((e - minProtonEnergy)/energyInterval);
	int ceiling = floor + 1;
	float b1 = data[floor].b;
	float b2 = data[ceiling].b;
	float e1 = data[floor].energy;
	float e2 = data[ceiling].energy;
	return b1 + (e - e1)*(b2 - b1)/(e2 - e1);
}
*/
