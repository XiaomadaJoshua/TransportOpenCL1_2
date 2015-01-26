#pragma once
#include "cl.hpp"
class OpenCLStuff;
class DensCorrection;
class MSPR;

class Phantom
{
public:
	Phantom(OpenCLStuff &, cl_float3 voxSize_, cl_int3 size_, const DensCorrection & densityCF, const MSPR & massSPR);
	virtual ~Phantom();
	cl::Image3D & voxelGPU(){ return voxelAttributes; }
	cl::Buffer & doseCounterGPU(){ return doseCounter; }
	cl_float3 voxelSize() const { return voxSize; }
	void finalize(OpenCLStuff & stuff);
	void output(OpenCLStuff & stuff, std::string & outDir);

private:
	cl_float3 voxSize;
	cl_int3 size;
	cl_float4 * attributes;
	cl::Image3D voxelAttributes;
	cl::Buffer doseCounter;
	cl::Program program;


	float ct2den(cl_float huValue) const;
	cl_float setMaterial(cl_float huValue, const MSPR & massSPR) const;
	cl_float ct2eden(cl_float material, cl_float density) const;
};

