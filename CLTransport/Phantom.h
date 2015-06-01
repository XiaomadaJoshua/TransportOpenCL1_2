#pragma once
#include <CL/cl.hpp>
#include <string>
class OpenCLStuff;
class DensCorrection;
class MSPR;

class Phantom
{
public:
	Phantom(OpenCLStuff &, cl_float3 voxSize_, cl_int3 size_, const DensCorrection & densityCF, const MSPR & massSPR);
	Phantom(OpenCLStuff &, const std::string & CTConfig, const std::string & CTFile, const DensCorrection & densityCF, const MSPR & massSPR);
	virtual ~Phantom();
	cl::Image3D & voxelGPU(){ return voxelAttributes; }
	cl::Buffer & doseCounterGPU(){ return doseCounter; }
	cl_float3 voxelSize() const { return voxSize; }
	cl_float3 phantomShift() const { return shift; }
	void finalize(OpenCLStuff & stuff);
	void output(OpenCLStuff & stuff, std::string & outDir);
	void tempStore(OpenCLStuff & stuff);

private:
	cl_float3 voxSize;
	cl_float3 shift;
	cl_int3 size;
	cl_float4 * attributes;
	cl::Image3D voxelAttributes;
	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 5 in float8 is primary dose,
	// 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	cl::Buffer doseCounter;
	cl::Program program;

	cl_float * totalDose;
	cl_float * primaryFluence;
	cl_float * secondaryFluence;
	cl_float * primaryLET;
	cl_float * secondaryLET;
	cl_float * heavyDose;
	cl_float * primaryDose;
	cl_float * secondaryDose;

	cl::Buffer doseBuff;

	void initialize(const short * CT, OpenCLStuff & stuff, const DensCorrection & densityCF, const MSPR & massSPR);
	float ct2den(cl_float huValue) const;
	cl_float setMaterial(cl_float huValue, const MSPR & massSPR) const;
	cl_float ct2eden(cl_float material, cl_float density) const;
	std::vector<float> getFloatQuantity(std::string input, std::string quantity);
	std::string getStringQuantity(std::string input, std::string quantity);
};

