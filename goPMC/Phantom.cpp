#include "stdafx.h"

#include <stdio.h>
#include "Phantom.h"
#include "OpenCLStuff.h"
#include "DensCorrection.h"
#include "MSPR.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "tinydir.h"
#include <vector>

const char *Phantom_ocl =
"#define INF 1.0e20\n"
"#define WATERDENSITY 1.0 // g/cm^3\n"
"#define MP 938.272046	//proton mass, in MeV\n"
"#define CP 1.00000 //proton charge\n"
"#define ME 0.510998928  //electron mass, in MeV\n"
"#define MO 14903.3460795634 //oxygen mass in MeV\n"
"#define MINELECTRONENERGY 0.1 // MeV\n"
"#define TWOPIRE2MENEW 0.08515495201157892 //2pi*r_e^2*m_e*n_{ew}, where r_e in cm, m_e in eV, n_ew = 3.34e23/cm^3\n"
"#define XW 36.514 	//radiation length of water, in cm\n"
"#define PI 3.1415926535897932384626433\n"
"#define SECONDPARTICLEVOLUME 10000\n"
"#define EMINPOI 1.0	//minimun energy used in p-o inelastic event, in MeV\n"
"#define EBIND 3.0	//initial binding energy used in p-o inelastic, in MeV\n"
"#define MAXSTEP 0.2 //in cm\n"
"#define MAXENERGYRATIO 0.25 //Max energy decay ratio of initial energy in a step\n"
"#define MINPROTONENERGY 1.0 //Min proton energy to transport\n"
"#define ZERO 1e-6\n"
"#define EPSILON 1e-20\n"
"#define MC 11177.928732 //carbon mass in MeV\n"
"#define CC 6.0000 //carbon charge\n"
"#define MINCARBONENERGY 5.0 //Min carbon energy to transport in MeV\n"
"#define SECONDARYNUMBERRATIO 2 // ratio of nbatch over maxWorkGroupSize\n"
"#define PPETHRESHOLD 10.0 // energy threshold of proton proton interaction\n"
"#define POETHRESHOLD 7.0 // energy threshold of proton oxygen elastic interaction\n"
"#define POITHRESHOLD 20.0 // energy threshold of proton oxygen inelastic interaction\n"
"#define NDOSECOUNTERS 8 // number of dosecounters\n"
"\n"
"\n"
"#define MIN(a,b) (a > b ? b : a)\n"
"#define MIN3(a,b,c) (a > b ? b : a) > c ? c : (a > b ? b : a)\n"
"#define ABS(a) a > 0 ? a : -a\n"
"\n"
"\n"
"__kernel void initializeDoseCounter(__global float * doseCounter){\n"
"	size_t gid = get_global_id(0);\n"
"	doseCounter[gid] = 0.0f;\n"
"}\n"
"\n"
"\n"
"__kernel void finalize(__global float * batchBuff, __global float * doseBuff, __global float * errorBuff, read_only image3d_t voxels, float3 voxSize, double totalWeight, int scoringQuantity){\n"
"	size_t idx = get_global_id(0);\n"
"	size_t idy = get_global_id(1);\n"
"	size_t idz = get_global_id(2);\n"
"	size_t sizeX = get_global_size(0), sizeY = get_global_size(1), sizeZ = get_global_size(2);\n"
"	size_t absId = idx + idy*sizeX + idz*sizeX*sizeY;\n"
"	size_t nVoxels = sizeX*sizeY*sizeZ;\n"
"	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
"	float4 vox = read_imagef(voxels, voxSampler, (float4)(idx, idy, idz, 0.0f));\n"
"	float volume = voxSize.x*voxSize.y*voxSize.z;\n"
"	float mass = vox.s2*volume;\n"
"	float mean = 0.0f, var = 0.0f, std;\n"
"\n"
"	for(int i = 0; i < NDOSECOUNTERS; i++){\n"
"		mean += batchBuff[absId + i*nVoxels];\n"
"		var += batchBuff[absId + i*nVoxels]*batchBuff[absId + i*nVoxels];\n"
"//		if(idx == 25 && idy == 25 && idz == 0)\n"
"//			printf(\"%v8f\\n\", doseCounter[absId + i*nVoxels]);\n"
"	}\n"
"\n"
"	std = sqrt(NDOSECOUNTERS*var - mean*mean);\n"
"\n"
"	switch(scoringQuantity){\n"
"	case(2):\n"
"		mean = mean/volume/totalWeight;\n"
"		std = std/volume/totalWeight;\n"
"		break;\n"
"	case(3):\n"
"		mean = mean/mass/vox.s2/totalWeight;\n"
"		std = std/mass/vox.s2/totalWeight;\n"
"		break;\n"
"	default:\n"
"		mean = mean/mass/totalWeight;\n"
"		std = std/mass/totalWeight;\n"
"		break;\n"
"	}\n"
"\n"
"	\n"
"	doseBuff[absId] = mean;\n"
"	errorBuff[absId] = std;\n"
"\n"
"// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is nothing, \n"
"	//5 in float8 is primary dose, 6 in float8 is secondary dose, 7 in float8 is heavy dose.\n"
"/*	mean.s0 = mean.s0/mass/nPaths;\n"
"	mean.s5 = mean.s5/mass/nPaths;\n"
"	mean.s6 = mean.s6/mass/nPaths;\n"
"	mean.s7 = mean.s7/mass/nPaths;\n"
"	mean.s1 = mean.s1/volume/nPaths;\n"
"	mean.s2 = mean.s2/volume/nPaths;\n"
"	if(mean.s0 > ZERO)\n"
"		mean.s3 = mean.s3/vox.s2/mean.s0/nPaths;\n"
"	\n"
"	\n"
"	std.s0 = std.s0/mass/nPaths;\n"
"	std.s5 = std.s5/mass/nPaths;\n"
"	std.s6 = std.s6/mass/nPaths;\n"
"	std.s7 = std.s7/mass/nPaths;\n"
"	std.s1 = std.s1/volume/nPaths;\n"
"	std.s2 = std.s2/volume/nPaths;\n"
"	if(mean.s0 > ZERO)\n"
"		std.s3 = std.s3/vox.s2/mean.s0/nPaths;\n"
"\n"
"	doseBuff[absId] = mean;\n"
"//	if(idx == 25 && idy == 25 && idz == 0)\n"
"//		printf(\"%f\\n\", doseBuff[absId].s1);\n"
"	errorBuff[absId] = std;\n"
"//	if(idx == 25 && idy == 25 && idz == 0)\n"
"//		printf(\"%f\\n\", errorBuff[absId].s1);\n"
"\n"
"//	if(doseBuff[absId].s0 > 0.0f)\n"
"//		printf(\"buff, %v8f\\n\", doseBuff[absId]);\n"
"*/\n"
"}\n"
"\n"
"__kernel void tempStore(__global float * doseCounter, __global float * batchBuff){\n"
"	size_t idx = get_global_id(0);\n"
"	size_t idy = get_global_id(1);\n"
"	size_t idz = get_global_id(2);\n"
"	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);\n"
"	int nVoxels = get_global_size(0)*get_global_size(1)*get_global_size(2); \n"
"\n"
"//	printf(\"idx = %d, idy = %d, idz = %d, absId = %d, nVoxels = %d\\n\", idx, idy, idz, absId, nVoxels);\n"
"	\n"
"	for(int i  = 0; i < NDOSECOUNTERS; i++){\n"
"//		if(idx == 25 && idy == 25 && idz == 0)\n"
"//			printf(\"%f\\n\", doseCounter[absId + i*nVoxels].s1);\n"
"		batchBuff[absId + i*nVoxels] += doseCounter[absId + i*nVoxels];\n"
"		doseCounter[absId + i*nVoxels] = 0;\n"
"\n"
"//		if(idx == 25 && idy == 25 && idz == 2)\n"
"//			printf(\"%v8f\\n\", doseCounter[absId + i*nVoxels]);\n"
"	}\n"
"//	if(doseBuff[absId].s0 > 0.0f)\n"
"//		printf(\"buff, %v8f\\n\", doseBuff[absId]);\n"
"\n"
"}\n"
;



using namespace ppt;
using namespace std;


void collect_files(vector<string> &fns, const string &folder)
{
	fns.clear();
	tinydir_dir dir;
	tinydir_open(&dir, folder.c_str());

	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);

		//printf("%s", file.name);
		string fn = folder;
		fn += "\\";
		fn += file.name;
		if (!file.is_dir)
		{
			if (fn.length()>4)
			{
				if (!string(".dcm").compare(0, 4, fn, fn.length() - 4, 4))
				{
					//printf("%s\n",fn.c_str());
					fns.push_back(fn);
				}
			}
		}

		tinydir_next(&dir);
	}

	tinydir_close(&dir);
}


Phantom::Phantom(OpenCLStuff & stuff, cl_float3 voxSize_, cl_int3 size_, const DensCorrection & densityCF, const MSPR & massSPR, const char* CTFile)
	:voxSize(voxSize_), size(size_)
{
	offSet.s[0] = 0.0f;
	offSet.s[1] = 0.0f;
	offSet.s[2] = 0.0f;
	
	int nVoxels = size.s[0] * size.s[1] * size.s[2];


	if (CTFile != NULL){
		FILE * fp = fopen(CTFile, "r");
		if (fp == NULL)
		{
			std::cout << "Error in opening CT file" << std::endl;
			exit(1);
		}
		CT = new short[nVoxels];
		printf("Reading CT number from %s\n", CTFile);

		fread(CT, sizeof(short)*nVoxels, 1, fp);
		fclose(fp);
		
		processAttributes(stuff, densityCF, massSPR, CT);
		
	}
	else{
		//initialize a water phantom
		std::cout << "initialize a water phantom" << std::endl;
		for (int i = 0; i < nVoxels; i++){
			attributes[i].s[0] = 0.0f; // ct value
			attributes[i].s[1] = setMaterial(attributes[i].s[0], massSPR); // material number
			attributes[i].s[2] = 1.0; // density
			attributes[i].s[3] = 1.0; // edensity
		}
	}

	delete[] CT;
}

Phantom::Phantom(OpenCLStuff & stuff, const DensCorrection & densityCF, const MSPR & massSPR, const std::string & dicomDir){
	data = new RTData;
	vector<string> fns;

	collect_files(fns, dicomDir);
	std::cout << "Reading CT from " << dicomDir << '\n';
	ReadDicomRT io(fns);
	io.ReadCT(*data);
	if (data->ct->size[0] > 256 || data->ct->size[1] > 256){
		std::cout << "Downsampling CT data from\t" << data->ct->size[0] << 'x' << data->ct->size[1] 
			<< 'x' << data->ct->size[2] << "\tto\t";
		data->ct->DownSampleXY();
		std::cout << data->ct->size[0] << 'x' << data->ct->size[1]
			<< 'x' << data->ct->size[2] << '\n';
	}
	size.s[0] = data->ct->size[0];
	size.s[1] = data->ct->size[1];
	size.s[2] = data->ct->size[2];

	voxSize.s[0] = data->ct->spacing[0];
	voxSize.s[1] = data->ct->spacing[1];
	voxSize.s[2] = data->ct->spacing[2];

	offSet.s[0] = data->ct->offset[0];
	offSet.s[1] = data->ct->offset[1];
	offSet.s[2] = data->ct->offset[2];

	processAttributes(stuff, densityCF, massSPR, data->ct->ct);
	//io.ReadPlan(data);

	//for(int i=0;i<data.ss->roi_names.size();i++)
	//{
	//	printf("%s\n",data.ss->roi_names[i].c_str());
	//}

	//data.ss->DownSampleXY();
	/*
	std::cout << "phantom size : " << size.s[0] << '\t' << size.s[1] << '\t' << size.s[2] << std::endl;
	std::cout << "phantom offset : " << offSet.s[0] << '\t' << offSet.s[1] << '\t' << offSet.s[2] << std::endl;
	std::cout << "phantom vox size : " << voxSize.s[0] << '\t' << voxSize.s[1] << '\t' << voxSize.s[2] << std::endl;
	
	char fn[256];
	sprintf(fn, "c:\\ct_%d_%d_%d.bin", size.s[0], size.s[1], size.s[2]);
	FILE *pf = fopen(fn, "wb");
	fwrite(data->ct->ct, sizeof(short), size.s[0]*size.s[1]*size.s[2], pf);
	fclose(pf);
	*/

	delete data;
	//sprintf(fn,"c:\\ss_%d_%d_%d.bin",data.ct->size[0],data.ct->size[1],data.ct->size[2]);
	//pf = fopen(fn,"wb");
	//fwrite(data.ss->roi,sizeof(int),data.ct->size[0]*data.ct->size[1]*data.ct->size[2],pf);
	//fclose(pf);
}

void Phantom::processAttributes(OpenCLStuff & stuff, const DensCorrection & densityCF, const MSPR & massSPR, short * ct){
	int err;
	cl::ImageFormat format(CL_RGBA, CL_FLOAT);
	voxelAttributes = cl::Image3D(stuff.context, CL_MEM_READ_ONLY, format, size.s[0], size.s[1], size.s[2], 0, 0, NULL, &err);

	int nVoxels = size.s[0] * size.s[1] * size.s[2];

	attributes = new cl_float4[nVoxels];

	//		float * densityCheck = new float[nVoxels];

	//initialize from CT file
	for (int i = 0; i < nVoxels; i++){
		attributes[i].s[0] = (float)ct[i]; // ct value
		attributes[i].s[0] = attributes[i].s[0] > massSPR.lookStartingHU()[0] ? attributes[i].s[0] : massSPR.lookStartingHU()[0];
		attributes[i].s[1] = setMaterial(attributes[i].s[0], massSPR); // material number
		attributes[i].s[2] = ct2den(attributes[i].s[0]); // density
		int ind = attributes[i].s[0] + 1000;
		ind = ind > densityCF.size() - 1 ? densityCF.size() - 1 : ind;
		attributes[i].s[2] *= densityCF[ind];
		attributes[i].s[3] = ct2eden(attributes[i].s[1], attributes[i].s[2]); // edensity

		//			densityCheck[i] = attributes[i].s[2];
	}

	//		std::ofstream ofsDensity("Output/density", std::ios::out | std::ios::trunc | std::fstream::binary);
	//		ofsDensity.write((const char *)(densityCheck), nVoxels * sizeof(float) / sizeof(char));
	//		ofsDensity.close();
	//		delete[] densityCheck;

	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = size.s[0];
	region[1] = size.s[1];
	region[2] = size.s[2];
	err = stuff.queue.enqueueWriteImage(voxelAttributes, CL_TRUE, origin, region, 0, 0, attributes);
	
	delete[] attributes;

	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 5 in float8 is primary dose,
	// 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	doseCounter = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float)*nVoxels*NDOSECOUNTERS, NULL, &err);
	if (err != 0){
		std::cout << "dose counter initialize failed. Maybe too many dose counters.\n" << std::endl;
		exit(1);
	}

	batchBuff = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float)*nVoxels*NDOSECOUNTERS, NULL, &err);
	if (err != 0){
		std::cout << "batch Buff initialize failed. Maybe too many dose counters.\n" << std::endl;
		exit(1);
	}

	std::string source(Phantom_ocl), include;
	/*	OpenCLStuff::convertToString("opencl_source/Phantom.cl", source);
	OpenCLStuff::convertToString("obfuscation_output/ReplacementFor_Macro.h", include);
	source = include + source;
	*/

	program = cl::Program(stuff.context, source);
	err = program.build("-cl-single-precision-constant -I.");
	std::string info;
	info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(stuff.device);
	if (err != 0){
		std::cout << "build result: " << err << std::endl;
		std::cout << info << std::endl;
		exit(-1);
	}

	cl::make_kernel<cl::Buffer &> initDoseCounterKernel(program, "initializeDoseCounter", &err);
	cl::NDRange globalRange(nVoxels*NDOSECOUNTERS);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	initDoseCounterKernel(arg, doseCounter);
	initDoseCounterKernel(arg, batchBuff);


	doseBuff = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float)*nVoxels, NULL, &err);
	errorBuff = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float)*nVoxels, NULL, &err);

	globalRange = cl::NDRange(nVoxels);
	cl::EnqueueArgs arg2(stuff.queue, globalRange);
	initDoseCounterKernel(arg2, doseBuff);
	initDoseCounterKernel(arg2, errorBuff);
}

Phantom::~Phantom()
{

}


cl_float Phantom::ct2den(cl_float huValue) const{
	//	convert HU to dens, in g/cm^3

	cl_float temp = 0.0f;
	//	MGH calibration curve
	if (huValue >= -1000 && huValue < -88)
		temp = 0.00121f + 0.001029700665188f*(1000.0f + huValue);
	else if (huValue >= -88 && huValue < 15)
		temp = 1.018f + 0.000893f*huValue;
	else if (huValue >= 15 && huValue < 23)
		temp = 1.03f;
	else if (huValue >= 23 && huValue < 101)
		temp = 1.003f + 0.001169f*huValue;
	else if (huValue >= 101 && huValue < 2001)
		temp = 1.017f + 0.000592f*huValue;
	else if (huValue >= 2001 && huValue < 2995)
		temp = 2.201f + 0.0005f*(-2000.0f + huValue);
	else
		temp = 4.54f;
	return temp;
}

cl_float Phantom::setMaterial(cl_float huValue, const MSPR & massSPR) const{
	if (huValue < massSPR.lookStartingHU()[0]){
		std::cout << "invalid HU" << std::endl;
		exit(EXIT_FAILURE);
	}
	for (unsigned int i = 0; i<massSPR.lookNMaterial() - 1; i++)
		if (huValue >= massSPR.lookStartingHU()[i] && huValue < massSPR.lookStartingHU()[i + 1]){
		int materialId = i;
		return (cl_float)(materialId);
		}
	int materialId = massSPR.lookNMaterial() - 1;
	return (cl_float)(materialId);
}

cl_float Phantom::ct2eden(cl_float material, cl_float density) const{
	//	convert HU to electron dens, in unit of ne_w=3.34e23/cm^3
	//  MGH CT convertion curve
	int Nele = 13;
	float Z[13] = { 1.0f, 6.0f, 7.0f, 8.0f, 12.0f, 15.0f, 16.0f, 17.0f, 18.0f, 20.0f, 11.0f, 19.0f, 22.0f };
	float A[13] = { 1.00794, 12.0107, 14.0067, 15.9994, 24.305, 30.9737, 32.065, 35.453, 39.948, 40.078, 22.9898, 39.0983, 47.867 };
	float ZperA[13];

	for (int iele = 0; iele < Nele; iele++){
		ZperA[iele] = Z[iele] / A[iele];
	}

	float composition[25][13] =
	{ { 0.0, 0.0, 0.755, 0.232, 0.0, 0.0, 0.0, 0.0, 0.013, 0.0, 0.0, 0.0, 0.0 },
	{ 0.103, 0.105, 0.031, 0.749, 0.0, 0.002, 0.003, 0.003, 0.0, 0.0, 0.002, 0.002, 0.0 },
	{ 0.116, 0.681, 0.002, 0.198, 0.0, 0.0, 0.001, 0.001, 0.0, 0.0, 0.001, 0.0, 0.0 },
	{ 0.113, 0.567, 0.009, 0.308, 0.0, 0.0, 0.001, 0.001, 0.0, 0.0, 0.001, 0.0, 0.0 },
	{ 0.110, 0.458, 0.015, 0.411, 0.0, 0.001, 0.002, 0.002, 0.0, 0.0, 0.001, 0.0, 0.0 },
	{ 0.108, 0.356, 0.022, 0.509, 0.0, 0.001, 0.002, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0 },
	{ 0.106, 0.284, 0.026, 0.578, 0.0, 0.001, 0.002, 0.002, 0.0, 0.0, 0.0, 0.001, 0.0 },
	{ 0.103, 0.134, 0.030, 0.723, 0.0, 0.002, 0.002, 0.002, 0.0, 0.0, 0.002, 0.002, 0.0 },
	{ 0.094, 0.207, 0.062, 0.622, 0.0, 0.0, 0.006, 0.003, 0.0, 0.0, 0.006, 0.0, 0.0 },
	{ 0.095, 0.455, 0.025, 0.355, 0.0, 0.021, 0.001, 0.001, 0.0, 0.045, 0.001, 0.001, 0.0 },
	{ 0.089, 0.423, 0.027, 0.363, 0.0, 0.030, 0.001, 0.001, 0.0, 0.064, 0.001, 0.001, 0.0 },
	{ 0.082, 0.391, 0.029, 0.372, 0.0, 0.039, 0.001, 0.001, 0.0, 0.083, 0.001, 0.001, 0.0 },
	{ 0.076, 0.361, 0.030, 0.380, 0.001, 0.047, 0.002, 0.001, 0.0, 0.101, 0.001, 0.0, 0.0 },
	{ 0.071, 0.335, 0.032, 0.387, 0.001, 0.054, 0.002, 0.0, 0.0, 0.117, 0.001, 0.0, 0.0 },
	{ 0.066, 0.310, 0.033, 0.394, 0.001, 0.061, 0.002, 0.0, 0.0, 0.132, 0.001, 0.0, 0.0 },
	{ 0.061, 0.287, 0.035, 0.400, 0.001, 0.067, 0.002, 0.0, 0.0, 0.146, 0.001, 0.0, 0.0 },
	{ 0.056, 0.265, 0.036, 0.405, 0.002, 0.073, 0.003, 0.0, 0.0, 0.159, 0.001, 0.0, 0.0 },
	{ 0.052, 0.246, 0.037, 0.411, 0.002, 0.078, 0.003, 0.0, 0.0, 0.170, 0.001, 0.0, 0.0 },
	{ 0.049, 0.227, 0.038, 0.416, 0.002, 0.083, 0.003, 0.0, 0.0, 0.181, 0.001, 0.0, 0.0 },
	{ 0.045, 0.210, 0.039, 0.420, 0.002, 0.088, 0.003, 0.0, 0.0, 0.192, 0.001, 0.0, 0.0 },
	{ 0.042, 0.194, 0.040, 0.425, 0.002, 0.092, 0.003, 0.0, 0.0, 0.201, 0.001, 0.0, 0.0 },
	{ 0.039, 0.179, 0.041, 0.429, 0.002, 0.096, 0.003, 0.0, 0.0, 0.210, 0.001, 0.0, 0.0 },
	{ 0.036, 0.165, 0.042, 0.432, 0.002, 0.100, 0.003, 0.0, 0.0, 0.219, 0.001, 0.0, 0.0 },
	{ 0.034, 0.155, 0.042, 0.435, 0.002, 0.103, 0.003, 0.0, 0.0, 0.225, 0.001, 0.0, 0.0 },
	{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 } };

	cl_float edensity = 0;
	cl_int materialId = round(material);
	for (int iele = 0; iele < Nele; iele++){
		edensity += composition[materialId][iele] * ZperA[iele];
	}
	edensity *= 6.02214129e23*density / 3.342774e23;
	return edensity;
}

void Phantom::finalize(OpenCLStuff & stuff, cl_double totalWeight, cl_int scoringQuantity){
	cl::make_kernel<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Image3D &, cl_float3, cl_double, cl_int> finalizeKernel(program, "finalize");
	cl::NDRange globalRange(size.s[0], size.s[1], size.s[2]);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	finalizeKernel(arg, batchBuff, doseBuff, errorBuff, voxelAttributes, voxSize, totalWeight, scoringQuantity);
}

void Phantom::clearCounter(OpenCLStuff & stuff){
	int err;
	cl::make_kernel<cl::Buffer &> initDoseCounterKernel(program, "initializeDoseCounter", &err);
	cl::NDRange globalRange(size.s[0]*size.s[1]*size.s[2]*NDOSECOUNTERS);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	initDoseCounterKernel(arg, doseCounter);
	initDoseCounterKernel(arg, batchBuff);
}

void Phantom::output(OpenCLStuff & stuff, std::vector<cl_float> & doseMean, std::vector<cl_float> & doseStd){
	int nVoxels = size.s[0] * size.s[1] * size.s[2];
	int err;

	doseMean.resize(nVoxels);
	doseStd.resize(nVoxels);

	err = stuff.queue.finish();
	err = stuff.queue.enqueueReadBuffer(doseBuff, CL_TRUE, 0, sizeof(cl_float) * nVoxels, doseMean.data());
	err = stuff.queue.enqueueReadBuffer(errorBuff, CL_TRUE, 0, sizeof(cl_float) * nVoxels, doseStd.data());
/*	std::string fileTotalBin = "dose_mean.bin";
	std::string fileDoseErr = "dose_std.bin";

	std::ofstream ofsTotal(fileTotalBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsTotal.write((const char *)(dose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsTotal.close();

	std::ofstream ofsTotalErr(fileDoseErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsTotal.write((const char *)(error), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsTotal.close();

	delete[] dose;
	delete[] error;
*/
}

/*
void Phantom::output(OpenCLStuff & stuff, std::string & outDir){
	int nVoxels = size.s[0] * size.s[1] * size.s[2];
	cl_float8 * dose = new cl_float8[nVoxels]();
	cl_float8 * error = new cl_float8[nVoxels]();

	int err;

	err = stuff.queue.finish();
	err = stuff.queue.enqueueReadBuffer(doseBuff, CL_TRUE, 0, sizeof(cl_float8) * nVoxels, dose);
	err = stuff.queue.enqueueReadBuffer(errorBuff, CL_TRUE, 0, sizeof(cl_float8) * nVoxels, error);

	totalDose = new cl_float[nVoxels]();
	primaryFluence = new cl_float[nVoxels]();
	secondaryFluence = new cl_float[nVoxels]();
	LET = new cl_float[nVoxels]();
	secondaryLET = new cl_float[nVoxels]();
	heavyDose = new cl_float[nVoxels]();
	primaryDose = new cl_float[nVoxels]();
	secondaryDose = new cl_float[nVoxels]();

	doseErr = new cl_float[nVoxels]();
	primaryFluenceErr = new cl_float[nVoxels]();
	secondaryFluenceErr = new cl_float[nVoxels]();
	LETErr = new cl_float[nVoxels]();
	secondaryLETErr = new cl_float[nVoxels]();
	heavyDoseErr = new cl_float[nVoxels]();
	primaryDoseErr = new cl_float[nVoxels]();
	secondaryDoseErr = new cl_float[nVoxels]();


	std::string fileTotalBin = outDir+"totalDose.bin";
	std::string filePFBin = outDir + "primaryFluence.bin";
	std::string fileSFBin = outDir + "secondaryFluence.bin";
	std::string fileLETBin = outDir + "LET.bin";
	std::string fileSLETBin = outDir + "secondaryLET.bin";
	std::string fileHeavyBin = outDir + "heavyDose.bin";
	std::string filePDBin = outDir + "primaryDose.bin";
	std::string fileSDBin = outDir + "secondaryDose.bin";

	std::string fileDoseErr = outDir + "doseErr.bin";
	std::string filePFErr = outDir + "primaryFluenceErr.bin";
	std::string fileSFErr = outDir + "secondaryFluenceErr.bin";
	std::string fileLETErr = outDir + "LETErr.bin";
	std::string fileSLETErr = outDir + "secondaryLETErr.bin";
	std::string fileHeavyErr = outDir + "heavyDoseErr.bin";
	std::string filePDErr = outDir + "primaryDoseErr.bin";
	std::string fileSDErr = outDir + "secondaryDoseErr.bin";

	for (int i = 0; i < nVoxels; i++){
		totalDose[i] = dose[i].s[0];
		primaryFluence[i] = dose[i].s[1];
		secondaryFluence[i] = dose[i].s[2];
		LET[i] = dose[i].s[3];
		secondaryLET[i] = dose[i].s[4];
		heavyDose[i] = dose[i].s[7];
		primaryDose[i] = dose[i].s[5];
		secondaryDose[i] = dose[i].s[6];

		doseErr[i] = error[i].s[0];
		primaryFluenceErr[i] = error[i].s[1];
		secondaryFluenceErr[i] = error[i].s[2];
		LETErr[i] = error[i].s[3];
		secondaryLETErr[i] = error[i].s[4];
		heavyDoseErr[i] = error[i].s[7];
		primaryDoseErr[i] = error[i].s[5];
		secondaryDoseErr[i] = error[i].s[6];
	}


	std::ofstream ofsTotal(fileTotalBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsTotal.write((const char *)(totalDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsTotal.close();

	std::ofstream ofsPF(filePFBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsPF.write((const char *)(primaryFluence), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsPF.close();

	std::ofstream ofsSF(fileSFBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsSF.write((const char *)(secondaryFluence), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsSF.close();

	std::ofstream ofsLET(fileLETBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsLET.write((const char *)(LET), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsLET.close();

//	std::ofstream ofsSLET(fileSLETBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
//	ofsSLET.write((const char *)(secondaryLET), nVoxels * sizeof(cl_float) / sizeof(char));
//	ofsSLET.close();

	std::ofstream ofsHeavy(fileHeavyBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsHeavy.write((const char *)(heavyDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsHeavy.close();

	std::ofstream ofsPD(filePDBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsPD.write((const char *)(primaryDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsPD.close();

	std::ofstream ofsSD(fileSDBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsSD.write((const char *)(secondaryDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsSD.close();




	std::ofstream ofsTotalErr(fileDoseErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsTotalErr.write((const char *)(doseErr), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsTotalErr.close();

	std::ofstream ofsPFErr(filePFErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsPFErr.write((const char *)(primaryFluenceErr), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsPFErr.close();

	std::ofstream ofsSFErr(fileSFErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsSFErr.write((const char *)(secondaryFluenceErr), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsSFErr.close();

	std::ofstream ofsLETErr(fileLETErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsLETErr.write((const char *)(LETErr), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsLETErr.close();

	//	std::ofstream ofsSLET(fileSLETBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	//	ofsSLET.write((const char *)(secondaryLET), nVoxels * sizeof(cl_float) / sizeof(char));
	//	ofsSLET.close();

	std::ofstream ofsHeavyErr(fileHeavyErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsHeavyErr.write((const char *)(heavyDoseErr), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsHeavyErr.close();

	std::ofstream ofsPDErr(filePDErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsPDErr.write((const char *)(primaryDoseErr), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsPDErr.close();

	std::ofstream ofsSDErr(fileSDErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsSDErr.write((const char *)(secondaryDoseErr), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsSDErr.close();
}
*/

void Phantom::tempStore(OpenCLStuff & stuff){
	cl::make_kernel<cl::Buffer &, cl::Buffer &> tempStoreKernel(program, "tempStore");
	cl::NDRange globalRange(size.s[0], size.s[1], size.s[2]);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	tempStoreKernel(arg, doseCounter, batchBuff);
}
