// test.cpp : Defines the entry point for the console application.
//

#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>

#define NDOSECOUNTERS 1
#include "goPMC.h"
#define N 100000

void initSource(cl_float * T, cl_float3 * pos, cl_float3 * dir, cl_float * weight){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::minstd_rand0 g1(seed1);  // minstd_rand0 is a standard linear_congruential_engine
	std::fill_n(T, N, 120.0f);
	std::fill_n(weight, N, 0.1f);
	for (int i = 0; i < N; i++){
		pos[i].s[0] = 5 * float(g1()) / g1.max() - 15;
		pos[i].s[1] = -20;
		pos[i].s[2] = 5 * float(g1()) / g1.max() + 25;
	}
	cl_float3 temp2 = { 0.0f, 1.0f, 0.0f };
	std::fill_n(dir, N, temp2);
}

int main()
{
	cl::Platform platform;
	cl::Platform::get(&platform);

	std::vector<cl::Device> devs;
	platform.getDevices(CL_DEVICE_TYPE_CPU, &devs);
	cl::Device device = devs[0];
	goPMC::MCEngine mcEngine;
	mcEngine.initializeComputation(platform, device);
	mcEngine.initializePhysics();
	mcEngine.initializePhantom("T:\\Physics Research\\PICO Users\\Nan Qin\\DiCOMData");
	
	cl_float * T = new cl_float[N];
	cl_float3 * pos = new cl_float3[N];
	cl_float3 * dir = new cl_float3[N];
	cl_float * weight = new cl_float[N];
	initSource(T, pos, dir, weight);

	//Scoring quantity could be one of {DOSE2MEDIUM, DOSE2WATER, FLUENCE, LETD}
	//LETD is dose weighted LET, to get dose averaged LET, divide it by DOSE2MEDIUM from another simulation run

	std::string quantity("DOSE2WATER"); 
	mcEngine.simulate(T, pos, dir, weight, N, quantity);
	
	std::vector<cl_float> doseMean, doseStd;
	mcEngine.getResult(doseMean, doseStd);
	mcEngine.clearCounter();




	std::string fileTotalBin = "Output/dose_mean.bin";
	std::string fileDoseErr = "Output/dose_std.bin";

	std::ofstream ofsTotal(fileTotalBin, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsTotal.write((const char *)(doseMean.data()), doseMean.size() * sizeof(cl_float) / sizeof(char));
	ofsTotal.close();

	std::ofstream ofsTotalErr(fileDoseErr, std::fstream::out | std::fstream::trunc | std::fstream::binary);
	ofsTotal.write((const char *)(doseStd.data()), doseStd.size() * sizeof(cl_float) / sizeof(char));
	ofsTotal.close();

	delete[] T;
	delete[] pos;
	delete[] dir;
	delete[] weight;
	return 0;
}

