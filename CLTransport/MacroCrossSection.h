/*
 * MacroCrossSection.h
 *
 *  Created on: Oct 1, 2014
 *      Author: S158879
 */

#ifndef MACROCROSSSECTION_H_
#define MACROCROSSSECTION_H_
#include <CL/cl.hpp>
using namespace std;

class OpenCLStuff;

class MacroCrossSection{
public:
	MacroCrossSection();
	virtual ~MacroCrossSection();
	virtual bool setData(const char *, const OpenCLStuff &);
	cl_uint size() const{ return nData; }

/*	float lookIon(float e) const;
	float lookPE(float e) const;
	float lookOE(float e) const;
	float lookOI(float e) const;*/
	cl::Image1D & gpu() { return mcs; }

private:
	float minElectronEnergy, minProtonEnergy, maxProtonEnergy, energyInterval;
	cl_uint nData;
	struct MCS{
		cl_float sigIon, sigPPE, sigPOE, sigPOI;
	};
	MCS * macroSigma;

	cl::Image1D mcs;
};

#endif /* MACROCROSSSECTION_H_ */
