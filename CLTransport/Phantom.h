#pragma once
#include "cl.hpp"
class OpenCLStuff;
class DensCorrection;
class MSPR;

class Phantom
{
public:
	Phantom(OpenCLStuff &, cl_float3 voxSize_, cl_int3 size_, const DensCorrection & densityCF, const MSPR & massSPR, const char* CTFile = NULL);
	virtual ~Phantom();
	cl::Image3D & voxelGPU(){ return voxelAttributes; }
	cl::Buffer & doseCounterGPU(){ return doseCounter; }
//	cl::Buffer & errorCounterGPU(){ return errorCounter; }
	cl_float3 voxelSize() const { return voxSize; }
	void finalize(OpenCLStuff & stuff, cl_uint nPaths);
	void output(OpenCLStuff & stuff, std::string & outDir);
//	void tempStore(OpenCLStuff & stuff);

private:
	cl_float3 voxSize;
	cl_int3 size;
	cl_float4 * attributes;
	cl::Image3D voxelAttributes;
	cl::Program program;
	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 5 in float8 is primary dose,
	// 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	cl::Buffer doseCounter;
	cl::Buffer doseBuff;
//	cl::Buffer errorCounter;
	cl::Buffer errorBuff;

	cl_float * totalDose;
	cl_float * primaryFluence;
	cl_float * secondaryFluence;
	cl_float * LET;
	cl_float * secondaryLET;
	cl_float * heavyDose;
	cl_float * primaryDose;
	cl_float * secondaryDose;

	cl_float * doseErr;
	cl_float * primaryFluenceErr;
	cl_float * secondaryFluenceErr;
	cl_float * LETErr;
	cl_float * secondaryLETErr;
	cl_float * heavyDoseErr;
	cl_float * primaryDoseErr;
	cl_float * secondaryDoseErr;


	float ct2den(cl_float huValue) const;
	cl_float setMaterial(cl_float huValue, const MSPR & massSPR) const;
	cl_float ct2eden(cl_float material, cl_float density) const;
};

