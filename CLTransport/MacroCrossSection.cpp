/*
 * MacroCrossSection.cpp
 *
 *  Created on: Oct 1, 2014
 *      Author: S158879
 */

#include "MacroCrossSection.h"
#include <stdlib.h>
#include <fstream>
#include "OpenCLStuff.h"
using namespace std;

MacroCrossSection::MacroCrossSection():
minElectronEnergy(0), minProtonEnergy(0), maxProtonEnergy(0), energyInterval(0), nData(0), macroSigma(NULL){
	// TODO Auto-generated constructor stub

}

MacroCrossSection::~MacroCrossSection() {
	// TODO Auto-generated destructor stub
	delete [] macroSigma;
}

bool MacroCrossSection::setData(const char * filename, const OpenCLStuff & stuff){
	ifstream ifs;
	ifs.open(filename, ios::in);
	if(!ifs.is_open())
		return false;
	char temp[256];
	ifs.getline(temp, 256);
	ifs.getline(temp, 256);
	ifs >> minElectronEnergy >> minProtonEnergy >> maxProtonEnergy >> energyInterval >> nData >> ws;
	macroSigma = new MCS[nData];
	ifs.getline(temp, 256);
	float thisEnergy;
	for(unsigned int i=0; i<nData; i++)
		ifs >> thisEnergy >> macroSigma[i].sigIon >> macroSigma[i].sigPPE >> macroSigma[i].sigPOE >> macroSigma[i].sigPOI >> ws;

	ifs.close();

	cl::ImageFormat format(CL_RGBA, CL_FLOAT);
	mcs = cl::Image1D(stuff.context, CL_MEM_READ_ONLY, format, nData);
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = nData;
	region[1] = 1;
	region[2] = 1;
	stuff.queue.enqueueWriteImage(mcs, CL_TRUE, origin, region, 0, 0, macroSigma);
}
/*
float MacroCrossSection::lookIon(float e) const{
if(e < minProtonEnergy)
return macroSigma[0].sigIon;
if(e > maxProtonEnergy)
return macroSigma[nData-1].sigIon;
int floor((e-minProtonEnergy)/energyInterval), ceiling(1 + (e-minProtonEnergy)/energyInterval);
float sig1 = macroSigma[floor].sigIon;
float sig2 = macroSigma[ceiling].sigIon;
float e1 = macroSigma[floor].energy;
float e2 = macroSigma[ceiling].energy;
return sig1 + (e - e1)*(sig2 - sig1)/(e2 - e1);
}

float MacroCrossSection::lookPE(float e) const{
if(e < minProtonEnergy)
return macroSigma[0].sigPPE;
if(e > maxProtonEnergy)
return macroSigma[nData-1].sigPPE;
int floor((e-minProtonEnergy)/energyInterval), ceiling(1 + (e-minProtonEnergy)/energyInterval);
float sig1 = macroSigma[floor].sigPPE;
float sig2 = macroSigma[ceiling].sigPPE;
float e1 = macroSigma[floor].energy;
float e2 = macroSigma[ceiling].energy;
return sig1 + (e - e1)*(sig2 - sig1)/(e2 - e1);
}

float MacroCrossSection::lookOE(float e) const{
if(e < minProtonEnergy)
return macroSigma[0].sigPOE;
if(e > maxProtonEnergy)
return macroSigma[nData-1].sigPOE;
int floor((e-minProtonEnergy)/energyInterval), ceiling(1 + (e-minProtonEnergy)/energyInterval);
float sig1 = macroSigma[floor].sigPOE;
float sig2 = macroSigma[ceiling].sigPOE;
float e1 = macroSigma[floor].energy;
float e2 = macroSigma[ceiling].energy;
return sig1 + (e - e1)*(sig2 - sig1)/(e2 - e1);
}

float MacroCrossSection::lookOI(float e) const{
if(e < minProtonEnergy)
return macroSigma[0].sigPOI;
if(e > maxProtonEnergy)
return macroSigma[nData-1].sigPOI;
int floor((e-minProtonEnergy)/energyInterval), ceiling(1 + (e-minProtonEnergy)/energyInterval);
float sig1 = macroSigma[floor].sigPOI;
float sig2 = macroSigma[ceiling].sigPOI;
float e1 = macroSigma[floor].energy;
float e2 = macroSigma[ceiling].energy;
return sig1 + (e - e1)*(sig2 - sig1)/(e2 - e1);
}
*/




