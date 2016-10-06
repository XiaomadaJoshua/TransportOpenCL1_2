/*
 *
 * file  example.cpp
 * brief example program for using goPMC.
 *
 * author Nan Qin
 *
 * last update on 9/21/2016
 *
 */
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

// A function to initialize source protons. Should be replaced by real beams.
void initSource(cl_float * T, cl_float3 * pos, cl_float3 * dir, cl_float * weight){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	std::minstd_rand0 g1(seed1);  // minstd_rand0 is a standard linear_congruential_engine
	std::fill_n(T, N, 120.0f);
	std::fill_n(weight, N, 1.0f);
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
	// Get OpenCL platform and device.
	cl::Platform platform;
	cl::Platform::get(&platform);
	std::vector<cl::Device> devs;
	platform.getDevices(CL_DEVICE_TYPE_CPU, &devs);
	cl::Device device = devs[0];
	
	// Initialize simulation engine.
	goPMC::MCEngine mcEngine;
	mcEngine.initializeComputation(platform, device);
	
	// Read and process physics data.
	mcEngine.initializePhysics("input");
	
	// Read and process patient Dicom CT data.
	mcEngine.initializePhantom("directoryToDicomData");
	
	// Initialize source protons with arrays of energy (T), position (pos), direction (dir) and weight (weight) of each proton.
	// Position and direction should be defined in Dicom CT coordinate.
	cl_float * T = new cl_float[N];
	cl_float3 * pos = new cl_float3[N];
	cl_float3 * dir = new cl_float3[N];
	cl_float * weight = new cl_float[N];
	initSource(T, pos, dir, weight);
	
	// Choose a physics quantity to score for this simulation run.
	// Scoring quantity could be one of {DOSE2MEDIUM, DOSE2WATER, FLUENCE, LETD}.
	// LETD is dose weighted LET, to get dose averaged LET, divide it by DOSE2MEDIUM from another simulation run.
	std::string quantity("DOSE2WATER"); 
	
	// Run simulation.
	mcEngine.simulate(T, pos, dir, weight, N, quantity);
	
	// Get simulation results.
	std::vector<cl_float> doseMean, doseStd;
	mcEngine.getResult(doseMean, doseStd);
	
	// Clear the scoring counters in previous simulation runs.
	mcEngine.clearCounter();


	delete[] T;
	delete[] pos;
	delete[] dir;
	delete[] weight;
	return 0;
}

