/*
 * DensCorrection.cpp
 *
 *  Created on: Sep 30, 2014
 *      Author: S158879
 */

#include "DensCorrection.h"
#include <fstream>
#include "OpenCLStuff.h"

using namespace std;

DensCorrection::DensCorrection(): densCorrectionFactor(NULL), nFactor(0) {
	// TODO Auto-generated constructor stub

}

DensCorrection::~DensCorrection() {
	// TODO Auto-generated destructor stub
	delete [] densCorrectionFactor;
}

bool DensCorrection::setData(const char * filename, const OpenCLStuff & stuff){
	ifstream ifs;
	ifs.open(filename, ios::in);
	if(!ifs.is_open())
		return false;
	char temp[256];
	ifs.getline(temp, 256);
	ifs >> nFactor >> ws;
	ifs.getline(temp, 256);
	densCorrectionFactor = new cl_float [nFactor];
	for(int i=0; i<nFactor; i++){
		ifs >> densCorrectionFactor[i];
		ifs.ignore(3);
	}

	ifs.close();

	cl::ImageFormat format(CL_R, CL_FLOAT);
	dcf = cl::Image2D(stuff.context, CL_MEM_READ_ONLY, format, nFactor, 1);
	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = nFactor;
	region[1] = 1;
	region[2] = 1;
	stuff.queue.enqueueWriteImage(dcf, CL_TRUE, origin, region, 0, 0, densCorrectionFactor);

	return true;
}

