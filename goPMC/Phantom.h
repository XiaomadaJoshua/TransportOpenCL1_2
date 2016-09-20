#pragma once
#include "cl.hpp"
#include <string>
#include "read_dicomrt.h"
#include <vector>

class OpenCLStuff;
class DensCorrection;
class MSPR;


class Phantom
{
public:
	Phantom(OpenCLStuff &, cl_float3 voxSize_, cl_int3 size_, const DensCorrection & densityCF, const MSPR & massSPR, const char* CTFile = NULL);
	Phantom(OpenCLStuff &, const DensCorrection & densityCF, const MSPR & massSPR, const std::string & dicomDir);
	virtual ~Phantom();
	cl::Image3D & voxelGPU(){ return voxelAttributes; }
	cl::Buffer & doseCounterGPU(){ return doseCounter; }
//	cl::Buffer & errorCounterGPU(){ return errorCounter; }
	cl_float3 voxelSize() const { return voxSize; }
	cl_int3 phantomResolution() const { return size; }
	cl_float3 patientOffSet() const { return offSet; }
	void finalize(OpenCLStuff & stuff, cl_double totalWeight, cl_int scoringQuantity);
//	void output(OpenCLStuff & stuff, std::string & outDir);
	void output(OpenCLStuff & stuff, std::vector<cl_float> &, std::vector<cl_float> &);
	void tempStore(OpenCLStuff & stuff);
	void clearCounter(OpenCLStuff &);
private:
	cl_float3 voxSize;
	cl_int3 size;
	cl_float3 offSet;
	short * CT;
	ppt::RTData * data;
	cl_float4 * attributes;
	cl::Image3D voxelAttributes;
	cl::Program program;
	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 5 in float8 is primary dose,
	// 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	cl::Buffer doseCounter;
	cl::Buffer doseBuff;
	cl::Buffer batchBuff;
	cl::Buffer errorBuff;



	float ct2den(cl_float huValue) const;
	cl_float setMaterial(cl_float huValue, const MSPR & massSPR) const;
	cl_float ct2eden(cl_float material, cl_float density) const;
	void processAttributes(OpenCLStuff &, const DensCorrection & densityCF, const MSPR & massSPR, short * ct);
};

