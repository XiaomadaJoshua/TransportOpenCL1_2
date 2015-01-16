/*
 * RSPW.cpp
 *
 *  Created on: Oct 1, 2014
 *      Author: S158879
 */

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
	delete [] data;
}

bool RSPW::setData(const char * filename, const OpenCLStuff & stuff){
	ifstream ifs;
	ifs.open(filename, ios::in);
	if(!ifs.is_open())
		return false;
	char temp[256];
	ifs.getline(temp, 256);
	ifs.getline(temp, 256);
	ifs >> minElectronEnergy >> minProtonEnergy >> maxProtonEnergy >> energyInterval >> nData >> ws;
	data = new Lwb[nData];
	ifs.getline(temp, 256);
	ifs.getline(temp, 256);
	float Sw, Mw, energy;
	for(unsigned int i=0; i<nData; i++){
		ifs >> energy >> Sw >> Mw >> data[i].Lw >> data[i].b >> ws;
	}

	ifs.close();

	cl::ImageFormat format(CL_RG, CL_FLOAT);
	rspw = cl::Image1D(stuff.context, CL_MEM_READ_ONLY, format, nData);
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = nData;
	region[1] = 1;
	region[2] = 1;
	stuff.queue.enqueueWriteImage(rspw, CL_TRUE, origin, region, 0, 0, data);
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
