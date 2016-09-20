/*
 * DensCorrection.cpp
 *
 *  Created on: Sep 30, 2014
 *      Author: S158879
 */
#include "stdafx.h"

#include "DensCorrection.h"
#include <fstream>
#include <iostream>
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
	nFactor = 3996;
	densCorrectionFactor = new cl_float[nFactor];

	ifs.open(filename, std::ios::binary | std::ios::in);
	ifs.read((char *)(densCorrectionFactor), nFactor*sizeof(cl_float) / sizeof(char));
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

