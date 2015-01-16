/*
 * MSPR.h
 *
 *  Created on: Oct 2, 2014
 *      Author: S158879
 */

#ifndef MSPR_H_
#define MSPR_H_
#include <CL/cl.hpp>

class OpenCLStuff;


class MSPR{
public:
	MSPR();
	virtual ~MSPR();
	virtual bool setData(const char *, const OpenCLStuff &);

	const float * lookStartingHU() const;
	unsigned int lookNMaterial() const;
	unsigned int size() const{ return nEnergy; }
//	float lookMSPR(int matID, float energy) const;

	cl::Image2D & gpu() { return mspr; }

private:
	unsigned int nMaterial, nEnergy;
	float * startingHU;
	float minEnergy, energyInterval;
	float * fs;
	cl::Image2D mspr;
};

#endif /* MSPR_H_ */
