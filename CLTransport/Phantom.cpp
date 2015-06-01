#include "Phantom.h"
#include "OpenCLStuff.h"
#include "DensCorrection.h"
#include "MSPR.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using std::string;

Phantom::Phantom(OpenCLStuff & stuff, cl_float3 voxSize_, cl_int3 size_, const DensCorrection & densityCF, const MSPR & massSPR)
	:voxSize(voxSize_), size(size_)
{
	shift.s[0] = 0.0f;
	shift.s[1] = 0.0f;
	shift.s[2] = 0.0f;

	initialize(NULL, stuff, densityCF, massSPR);
}

Phantom::Phantom(OpenCLStuff & stuff, const string & CTConfig, const string & CTFile, const DensCorrection & densityCF, const MSPR & massSPR){
	char ch, buffer[500];
	cl_float3 phantomSize;
	std::ifstream ifs(CTConfig, std::fstream::in);
	ifs.getline(buffer, 500);
	ifs.ignore(100, '"');
	ifs >> phantomSize.s[0] >> ch >> phantomSize.s[1] >> ch >> phantomSize.s[2] >> ch
		>> shift.s[0] >> ch >> shift.s[1] >> ch >> shift.s[2]
		>> ch >> size.s[0] >> ch >> size.s[1] >> ch >> size.s[2];
	voxSize.s[0] = phantomSize.s[0] * 0.1f / size.s[0];
	voxSize.s[1] = phantomSize.s[1] * 0.1f / size.s[1]; 
	voxSize.s[2] = phantomSize.s[2] * 0.1f / size.s[2];

	ifs.close();

	FILE * fp = fopen(CTFile.c_str(), "r");
	if (fp == NULL)
	{
		std::cout << "Error in opening CT file" << std::endl;
		exit(1);
	}

	short * CT = (short *)malloc(sizeof(short) * size.s[0] * size.s[1] * size.s[2]);
	printf("Reading CT number...\n");

	fread(CT, sizeof(short)*size.s[0] * size.s[1] * size.s[2], 1, fp);
	fclose(fp);

	initialize(CT, stuff, densityCF, massSPR);
	delete[] CT;
}

void Phantom::initialize(const short * CT, OpenCLStuff & stuff, const DensCorrection & densityCF, const MSPR & massSPR){

	int err;
	cl::ImageFormat format(CL_RGBA, CL_FLOAT);
	voxelAttributes = cl::Image3D(stuff.context, CL_MEM_READ_ONLY, format, size.s[0], size.s[1], size.s[2], 0, 0, NULL, &err);

	int nVoxels = size.s[0] * size.s[1] * size.s[2];
	attributes = new cl_float4[nVoxels];

	if (CT != NULL){
		for (int i = 0; i < nVoxels; i++){
			attributes[i].s[0] = (float)CT[i]; // ct value
			attributes[i].s[0] = attributes[i].s[0] > massSPR.lookStartingHU()[0] ? attributes[i].s[0] : massSPR.lookStartingHU()[0];
			attributes[i].s[1] = setMaterial(attributes[i].s[0], massSPR); // material number
			attributes[i].s[2] = ct2den(attributes[i].s[0]); // density
			int ind = attributes[i].s[0] + 1000;
			ind = ind > densityCF.size() - 1 ? densityCF.size() - 1 : ind;
			attributes[i].s[2] *= densityCF[ind];
			attributes[i].s[3] = ct2eden(attributes[i].s[1], attributes[i].s[2]); // edensity
		}
	}
	else{
		for (int i = 0; i < nVoxels; i++){
			attributes[i].s[0] = 0.0f; // ct value
			attributes[i].s[1] = setMaterial(attributes[i].s[0], massSPR); // material number
			attributes[i].s[2] = 1.0; // density
			attributes[i].s[3] = 1.0; // edensity
		}
	}

	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = size.s[0];
	region[1] = size.s[1];
	region[2] = size.s[2];
	stuff.queue.enqueueWriteImage(voxelAttributes, CL_TRUE, origin, region, 0, 0, attributes);

	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 5 in float8 is primary dose,
	// 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	doseCounter = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float8)*nVoxels*NDOSECOUNTERS, NULL, &err);

	string source;
	OpenCLStuff::convertToString("Phantom.cl", source);
	program = cl::Program(stuff.context, source);
	program.build("-cl-single-precision-constant");
	string info;
	info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(stuff.device);
	cl::make_kernel<cl::Buffer &> initDoseCounterKernel(program, "initializeDoseCounter", &err);

	cl::NDRange globalRange(nVoxels*NDOSECOUNTERS);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	err = stuff.queue.finish();
	initDoseCounterKernel(arg, doseCounter);
	err = stuff.queue.finish();

	doseBuff = cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_float8)*nVoxels, NULL, &err);
	globalRange = cl::NDRange(nVoxels);
	cl::EnqueueArgs arg2(stuff.queue, globalRange);
	initDoseCounterKernel(arg2, doseBuff);
}




Phantom::~Phantom()
{
	delete[] attributes;
	delete[] totalDose;
	delete[] primaryFluence;
	delete[] secondaryFluence;
	delete[] primaryLET;
	delete[] secondaryLET;
	delete[] heavyDose;
	delete[] primaryDose;
	delete[] secondaryDose;
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

void Phantom::finalize(OpenCLStuff & stuff){
	cl::make_kernel<cl::Buffer &, cl::Image3D &, cl_float3> finalizeKernel(program, "finalize");
	cl::NDRange globalRange(size.s[0], size.s[1], size.s[2]);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	finalizeKernel(arg, doseBuff, voxelAttributes, voxSize);
}

void Phantom::output(OpenCLStuff & stuff, string & outDir){
	int nVoxels = size.s[0] * size.s[1] * size.s[2];
	cl_float8 * dose = new cl_float8[nVoxels];
	stuff.queue.finish();
	stuff.queue.enqueueReadBuffer(doseBuff, CL_TRUE, 0, sizeof(cl_float8) * nVoxels, dose);

	totalDose = new cl_float[nVoxels]();
	primaryFluence = new cl_float[nVoxels]();
	secondaryFluence = new cl_float[nVoxels]();
	primaryLET = new cl_float[nVoxels]();
	secondaryLET = new cl_float[nVoxels]();
	heavyDose = new cl_float[nVoxels]();
	primaryDose = new cl_float[nVoxels]();
	secondaryDose = new cl_float[nVoxels]();


	string fileTotal = outDir+"totalDose.dat";
	string fileTotalBin = outDir+"totalDose.bin";
	string filePF = outDir + "primaryFluence.dat";
	string filePFBin = outDir + "primaryFluence.bin";
	string fileSF = outDir + "secondaryFluence.dat";
	string fileSFBin = outDir + "secondaryFluence.bin";
	string filePLET = outDir + "primaryLET.dat";
	string filePLETBin = outDir + "primaryLET.bin";
	string fileSLET = outDir + "secondaryLET.dat";
	string fileSLETBin = outDir + "secondaryLET.bin";
	string fileHeavy = outDir + "heavyDose.dat";
	string fileHeavyBin = outDir + "heavyDose.bin";
	string filePD = outDir + "primaryDose.dat";
	string filePDBin = outDir + "primaryDose.bin";
	string fileSD = outDir + "secondaryDose.dat";
	string fileSDBin = outDir + "secondaryDose.bin";

	std::ofstream ofsTotal(fileTotal, std::ios::out | std::ios::trunc);
	std::ofstream ofsPF(filePF, std::ios::out | std::ios::trunc);
	std::ofstream ofsSF(fileSF, std::ios::out | std::ios::trunc);
	std::ofstream ofsPLET(filePLET, std::ios::out | std::ios::trunc);
	std::ofstream ofsSLET(fileSLET, std::ios::out | std::ios::trunc);
	std::ofstream ofsHeavy(fileHeavy, std::ios::out | std::ios::trunc);
	std::ofstream ofsPD(filePD, std::ios::out | std::ios::trunc);
	std::ofstream ofsSD(fileSD, std::ios::out | std::ios::trunc);

	for (int i = 0; i < nVoxels; i++){
		totalDose[i] = dose[i].s[0];
		primaryFluence[i] = dose[i].s[1];
		secondaryFluence[i] = dose[i].s[2];
		primaryLET[i] = dose[i].s[3];
		secondaryLET[i] = dose[i].s[4];
		heavyDose[i] = dose[i].s[7];
		primaryDose[i] = dose[i].s[5];
		secondaryDose[i] = dose[i].s[6];

		ofsTotal << totalDose[i] << '\t';
		ofsPF << primaryFluence[i] << '\t';
		ofsSF << secondaryFluence[i] << '\t';
		ofsPLET << primaryLET[i] << '\t';
		ofsSLET << secondaryLET[i] << '\t';
		ofsHeavy << heavyDose[i] << '\t';
		ofsPD << primaryDose[i] << '\t';
		ofsSD << secondaryDose[i] << '\t';
		
		if ((i + 1) % size.s[0] == 0){
			ofsTotal << '\n';
			ofsPF << '\n';
			ofsSF << '\n';
			ofsPLET << '\n';
			ofsSLET << '\n';
			ofsHeavy << '\n';
			ofsPD << '\n';
			ofsSD << '\n';
		}

		if ((i + 1) % (size.s[0] * size.s[1]) == 0){
			ofsTotal << '\n';
			ofsPF << '\n';
			ofsSF << '\n';
			ofsPLET << '\n';
			ofsSLET << '\n';
			ofsHeavy << '\n';
			ofsPD << '\n';
			ofsSD << '\n';
		}
	}
	ofsTotal.close();
	ofsPF.close();
	ofsSF.close();
	ofsPLET.close();
	ofsSLET.close();
	ofsHeavy.close();
	ofsPD.close();
	ofsSD.close();

	ofsTotal.open(fileTotalBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsTotal.write((const char *)(totalDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsTotal.close();

	ofsPF.open(filePFBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsPF.write((const char *)(primaryFluence), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsPF.close();

	ofsSF.open(fileSFBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsSF.write((const char *)(secondaryFluence), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsSF.close();

	ofsPLET.open(filePLETBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsPLET.write((const char *)(primaryLET), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsPLET.close();

	ofsSLET.open(fileSLETBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsSLET.write((const char *)(secondaryLET), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsSLET.close();

	ofsHeavy.open(fileHeavyBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsHeavy.write((const char *)(heavyDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsHeavy.close();

	ofsPD.open(filePDBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsPD.write((const char *)(primaryDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsPD.close();

	ofsSD.open(fileSDBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsSD.write((const char *)(secondaryDose), nVoxels * sizeof(cl_float) / sizeof(char));
	ofsSD.close();
}

void Phantom::tempStore(OpenCLStuff & stuff){
	cl::make_kernel<cl::Buffer &, cl::Buffer &> tempStoreKernel(program, "tempStore");
	cl::NDRange globalRange(size.s[0], size.s[1], size.s[2]);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	tempStoreKernel(arg, doseCounter, doseBuff);
}



string Phantom::getStringQuantity(string input, string quantity)
//	get a string value from an input string
{
	string value;

	unsigned found = input.find(quantity);
	//	founded
	if (found >= 0 && found < input.size())
	{
		for (int i = found + quantity.size(); i < input.size(); i++)
		{
			if (input[i] == '"')
			{
				int j = i + 1;
				while (input[j] != '"')
				{
					value += input[j];
					j++;
				}
				return value;
			}
		}
	}
	//	else return empty string
	return value;

}


std::vector<float> Phantom::getFloatQuantity(string input, string quantity)
//	get a float value from an input string
{
	std::vector<float> value;

	unsigned found = input.find(quantity);
	if (found >= 0 && found < input.size())
	{
		for (int i = found + quantity.size(); i < input.size();)
		{
			if (input[i] == '.' || isdigit(input[i]) || input[i] == '-')
			{
				string temp;
				while (input[i] == '.' || isdigit(input[i]) || input[i] == '-')
				{
					temp += input[i];
					i++;
				}

				value.push_back(atof(temp.c_str()));
			}
			else
				i++;
		}
	}

	return value;
}
