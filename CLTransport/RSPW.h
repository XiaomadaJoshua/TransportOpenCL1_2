/*
 * RSPW.h
 *
 *  Created on: Oct 1, 2014
 *      Author: S158879
 */

#ifndef RSPW_H_
#define RSPW_H_
#include <CL/cl.hpp>

class OpenCLStuff;

class RSPW{
public:
	RSPW();
	virtual ~RSPW();
	virtual bool setData(const char *, const OpenCLStuff &);
	cl::Image1D & gpu() {return rspw; }

//	float lookLw(float e) const;
//	float lookB(float e) const;

private:
	float minElectronEnergy, minProtonEnergy, maxProtonEnergy, energyInterval;
	cl_uint nData;
	struct Lwb{
		cl_float Lw, b;
	};
	Lwb * data;
	
	cl::Image1D rspw;
};

#endif /* RSPW_H_ */
