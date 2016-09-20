/*
 * DensCorrection.h
 *
 *  Created on: Sep 30, 2014
 *      Author: S158879
 */

#ifndef DENSCORRECTION_H_
#define DENSCORRECTION_H_
#include "cl.hpp"

class OpenCLStuff;

class DensCorrection{
public:
	DensCorrection();
	virtual ~DensCorrection();
	virtual bool setData(const char * filename, const OpenCLStuff & stuff);
	float operator [] (int i) const;
	int size() const;
	cl_float * data() const{
		return densCorrectionFactor;
	}
	cl::Image2D & gpu() { return dcf; }


private:
	cl::Image2D dcf;
	cl_float * densCorrectionFactor;
	int nFactor;
};

inline float DensCorrection::operator [] (int i) const{
	return densCorrectionFactor[i];
}

inline int DensCorrection::size() const{
	return nFactor;
}

#endif /* DENSCORRECTION_H_ */
