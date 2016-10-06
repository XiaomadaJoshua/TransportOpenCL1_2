#include "stdafx.h"

#include "ParticleStatus.h"
#include "OpenCLStuff.h"
#include "Secondary.h"
#include "Phantom.h"
#include "MacroCrossSection.h"
#include "RSPW.h"
#include "MSPR.h"
#include "Macro.h"
#include "OpenCLStuff.h"
#include <time.h>
#include <iostream>
#include <fstream>

const char *Particle_ocl =
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
"#define M_RAN_INVM32 2.32830643653869628906e-010\n"
"\n"
"float inverseCumulativeNormal(float u){\n"
"	float a[4] = { 2.50662823884f, -18.61500062529f, 41.39119773534f, -25.44106049637f };\n"
"	float b[4] = { -8.47351093090f, 23.08336743743f, -21.06224101826f, 3.13082909833f };\n"
"	float c[9] = { 0.3374754822726147f, 0.9761690190917186f, 0.1607979714918209f, 0.0276438810333863f, 0.0038405729373609f, 0.0003951896511919f,\n"
"		0.0000321767881768f, 0.0000002888167364f, 0.0000003960315187f };\n"
"	float x = u - 0.5f;\n"
"	float r;\n"
"	if (x < 0.42f && x > -0.42f)  // Beasley-Springer\n"
"	{\n"
"		float y = x*x;\n"
"		r = x*(((a[3] * y + a[2])*y + a[1])*y + a[0]) / ((((b[3] * y + b[2])*y + b[1])*y + b[0])*y + 1.0f);\n"
"	}\n"
"	else // Moro\n"
"	{\n"
"		r = u;\n"
"		if (x>0.0f)\n"
"			r = 1.0f - u;\n"
"		r = log(-log(r));\n"
"		r = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] + r*(c[5] +\n"
"			r*(c[6] + r*(c[7] + r*c[8])))))));\n"
"		if (x<0.0f)\n"
"			r = -r;\n"
"	}\n"
"	return r;\n"
"}\n"
"\n"
"float MTrng(int * iseed){\n"
"	int I1 = iseed[0] / 53668;\n"
"	iseed[0] = 40014 * (iseed[0] - I1 * 53668) - I1 * 12211;\n"
"	if (iseed[0] < 0) iseed[0] = iseed[0] + 2147483563;\n"
"\n"
"	int I2 = iseed[1] / 52774;\n"
"	iseed[1] = 40692 * (iseed[1] - I2 * 52774) - I2 * 3791;\n"
"	if (iseed[1] < 0) iseed[1] = iseed[1] + 2147483399;\n"
"\n"
"	int IZ = iseed[0] - iseed[1];\n"
"	if (IZ < 1) IZ = IZ + 2147483562;\n"
"	return (float)(IZ*4.656612873077392578125e-10);\n"
"}\n"
"\n"
"float MTGaussian(int * iseed){\n"
"	float u1 = MTrng(iseed);\n"
"	float u2 = MTrng(iseed);\n"
"	return sqrt(-2.0f*log(u1))*cos(2.0f*PI*u2);\n"
"}\n"
"\n"
"/*\n"
"float MTGaussian(int * iseed){\n"
"return inverseCumulativeNormal(MTrng(iseed));\n"
"}\n"
"*/\n"
"\n"
"float MTExp(int * iseed, float lambda){\n"
"\n"
"	return -log(1 - MTrng(iseed))*lambda;\n"
"}\n"
"\n"
"\n"
"#define M1 2147483647\n"
"#define M2 2147483399\n"
"#define A1 40015\n"
"#define A2 40692\n"
"#define Q1 ( M1 / A1 )\n"
"#define Q2 ( M2 / A2 )\n"
"#define R1 ( M1 % A1 )\n"
"#define R2 ( M2 % A2 )\n"
"\n"
"\n"
"/* Dual-Phase Linear Congruential Generator */\n"
"int jswRand(int * seed)\n"
"{\n"
"	int result;\n"
"\n"
"	seed[0] = A1 * (seed[0] % Q1) - R1 * (seed[0] / Q1);\n"
"	seed[1] = A2 * (seed[1] % Q2) - R2 * (seed[1] / Q2);\n"
"\n"
"	if (seed[0] <= 0)\n"
"	{\n"
"		seed[0] += M1;\n"
"	}\n"
"\n"
"	if (seed[1] <= 0)\n"
"	{\n"
"		seed[1] += M2;\n"
"	}\n"
"\n"
"	result = seed[0] - seed[1];\n"
"\n"
"	if (result < 1)\n"
"	{\n"
"		result += M1 - 1;\n"
"	}\n"
"\n"
"	return result;\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"\n"
"/*\n"
"\n"
"void getUniform(threefry2x32_ctr_t * c, float * random){\n"
"	threefry2x32_key_t k = { {0} };\n"
"\n"
"	*c = threefry2x32(*c, k);\n"
"	\n"
"	random[0] = (int)c->v[0] * M_RAN_INVM32 + 0.5f;\n"
"	random[1] = (int)c->v[1] * M_RAN_INVM32 + 0.5f;\n"
"}\n"
"\n"
"\n"
"void getGaussian(threefry2x32_ctr_t * c, float * variates){\n"
"	getUniform(c, variates);\n"
"	for (unsigned int i = 0; i<2; i++){\n"
"		float x = *(variates + i);\n"
"		*(variates + i) = inverseCumulativeNormal(x);\n"
"	}\n"
"}\n"
"\n"
"void getExp(threefry2x32_ctr_t * c, float * variates, float p){\n"
"	getUniform(c, variates);\n"
"	for (unsigned int i = 0; i<2; i++){\n"
"		*(variates + i) = -log(1 - *(variates + i)) / p;\n"
"	}\n"
"}*/\n"
"\n"
"\n"
"typedef struct __attribute__ ((aligned)) ParticleStatus{\n"
"	float3 pos, dir;\n"
"	float energy, maxSigma, mass, charge, weight;\n"
"}PS;\n"
"\n"
"\n"
"__kernel void initParticles(__global PS * particle, float T, float2 width, float3 sourceCenter, float m, float c, int randSeed){\n"
"	size_t gid = get_global_id(0);\n"
"	\n"
"//	if(gid == 1){\n"
"//		int size = sizeof(PS);\n"
"//		printf(\"size of PS: %d\\n\", size);\n"
"//	}\n"
"\n"
"	particle[gid].pos.z = sourceCenter.z;\n"
"	int iseed[2];\n"
"	iseed[0] = randSeed;\n"
"	iseed[1] = gid;\n"
"	jswRand(iseed);\n"
"	particle[gid].pos.x = (MTrng(iseed) - 0.5f) * width.s0;\n"
"	particle[gid].pos.y = (MTrng(iseed) - 0.5f) * width.s1;\n"
"	\n"
"	particle[gid].dir = normalize((float3)(0.0f, 0.0f, 0.0f) - sourceCenter);\n"
"	\n"
"//	particle[gid].dir.x = -1.0f;\n"
"//	particle[gid].dir.y = 0.0f;\n"
"//	particle[gid].dir.z = 0.0f;\n"
"\n"
"	particle[gid].energy = T;\n"
"	particle[gid].maxSigma = 0.0f;\n"
"	particle[gid].mass = m;\n"
"	particle[gid].charge = c;\n"
"	particle[gid].weight = 1.0f;\n"
"}\n"
"\n"
"__kernel void loadSource(__global PS * particle, __global float * energyBuffer, __global float3 * posBuffer, __global float3 * dirBuffer, __global float * weightBuffer, float3 translation){\n"
"	size_t gid = get_global_id(0);\n"
"	particle[gid].energy = energyBuffer[gid];\n"
"	particle[gid].pos = posBuffer[gid] + translation;\n"
"	particle[gid].dir = dirBuffer[gid];\n"
"	\n"
"\n"
"	particle[gid].maxSigma = 0.0f;\n"
"	particle[gid].mass = MP;\n"
"	particle[gid].charge = CP;\n"
"	particle[gid].weight = weightBuffer[gid];\n"
"}\n"
"\n"
"\n"
"bool ifInsidePhantom(float3 pos, float3 voxSize, int3 phantomSize, int3 * voxIndex, int * absIndex){\n"
"	int3 ifInside = isgreaterequal(pos, -convert_float3(phantomSize)*voxSize/2.0f) * islessequal(pos, convert_float3(phantomSize)*voxSize/2.0f);\n"
"//	printf(\"voxSize = %f %f %f, phantomSize = %d %d %d, pos = %f %f %f, ifInside = %d %d %d\\n\",\n"
"//			voxSize.x, voxSize.y, voxSize.z, phantomSize.x, phantomSize.y, phantomSize.z, pos.x, pos.y, pos.z, ifInside.x, ifInside.y, ifInside.z);\n"
"	if(ifInside.x == 0 || ifInside.y == 0 || ifInside.z == 0)\n"
"		return false;\n"
"	*voxIndex = convert_int3_rtn(pos / voxSize + convert_float3(phantomSize) / 2.0f);\n"
"	*absIndex = (*voxIndex).x + (*voxIndex).y*phantomSize.x + (*voxIndex).z*phantomSize.x*phantomSize.y;\n"
"	if(*absIndex >= phantomSize.x*phantomSize.y*phantomSize.z)\n"
"		return false;\n"
"	return true;\n"
"}\n"
"\n"
"\n"
"float step2VoxBoundary(float3 pos, float3 dir, float3 voxSize, int * cb, int3 phantomSize, int3 voxIndex) {\n"
"//	pos = (float3)(0.0, 0.0, 10.0);\n"
"//	dir = (float3)(0.01, -0.01, 0.999);\n"
"	float stepX, stepY, stepZ;\n"
"\n"
"	float3 phantomBoundary = convert_float3(phantomSize) * voxSize;\n"
"\n"
"//	printf(\"floor(0.0) = %f, ceil(0.0) = %f\\n\", floor(0.0f), ceil(0.0f));\n"
"	if(fabs(dir.x) < EPSILON)\n"
"		stepX = INF;\n"
"	else if(dir.x > 0)\n"
"		stepX = ((voxIndex.x + 1)*voxSize.x - phantomBoundary.x * 0.5f - pos.x)/dir.x;\n"
"	else\n"
"		stepX = (voxIndex.x * voxSize.x - phantomBoundary.x * 0.5f  - pos.x)/dir.x;\n"
"\n"
"	if(fabs(dir.y) < EPSILON)\n"
"		stepY = INF;\n"
"	else if(dir.y > 0)\n"
"		stepY = ((voxIndex.y + 1)*voxSize.y - phantomBoundary.y * 0.5f - pos.y)/dir.y;\n"
"	else\n"
"		stepY = (voxIndex.y * voxSize.y - phantomBoundary.y * 0.5f - pos.y)/dir.y;\n"
"\n"
"	if(fabs(dir.z) < EPSILON)\n"
"		stepZ = INF;\n"
"	else if(dir.z > 0)\n"
"		stepZ = ((voxIndex.z + 1)*voxSize.z - phantomBoundary.z * 0.5f - pos.z)/dir.z;\n"
"	else\n"
"		stepZ = (voxIndex.z * voxSize.z - phantomBoundary.z * 0.5f - pos.z)/dir.z;\n"
"\n"
"	float minStep;\n"
"	if(stepX < stepY){\n"
"		minStep = stepX;\n"
"		if(minStep < stepZ)\n"
"			*cb = 1;\n"
"		else{\n"
"			minStep = stepZ;\n"
"			*cb = 3;\n"
"		}\n"
"	}\n"
"	else{\n"
"		minStep = stepY;\n"
"		if(minStep < stepZ)\n"
"			*cb = 2;\n"
"		else{\n"
"			minStep = stepZ;\n"
"			*cb = 3;\n"
"		}\n"
"	}\n"
"//	printf(\"pos = %v3f, dir = %v3f, voxSize = %v3f, stepX = %f, stepY = %f, stepZ = %f, cb = %d\\n\",\n"
"//			pos, dir, voxSize, stepX, stepY, stepZ, *cb);\n"
"//	if(minStep < 0)\n"
"//		printf(\"vox = %v3d, dir = %v3f, pos = %v3f, stepX = %f, stepY = %f, stepZ = %f, minStep = %f\\n\", voxIndex, dir, pos, stepX, stepY, stepZ, minStep);\n"
"	return fabs(minStep) + 5e-6 ;\n"
"}\n"
"\n"
"float energyInOneStep(float4 vox, PS * particle, read_only image2d_t RSPW, read_only image2d_t MSPR, float stepLength) {\n"
"	//calculate equivalent step in water\n"
"\n"
"	float stepInWater;\n"
"	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;\n"
"	float4 mspr = read_imagef(MSPR, sampler, (float2)(particle->energy - 0.5f, vox.s1 + 0.5f));\n"
"	stepInWater = mspr.s0*stepLength*vox.s2/WATERDENSITY;\n"
"\n"
"	//calculate energy transfer\n"
"	float4 rspw = read_imagef(RSPW, sampler, (float2)(particle->energy/0.5f - 0.5f, 0.5f));\n"
"\n"
"\n"
"	float de1 = stepInWater*rspw.s0;\n"
"	float b = rspw.s1;\n"
"	float temp = particle->energy/particle->mass;\n"
"	float eps = de1/particle->energy;\n"
"	return de1*(1.0f + eps/(1.0f+temp)/(2.0f+temp) + eps*eps*(2.0f+2.0f*temp+temp*temp)/(1.0f+temp)/(1.0f+temp)/(2.0f+temp)/(2.0f+temp)\n"
"			- b*eps*(0.5f+2.0f*eps/3.0f/(1.0f+temp)/(2.0f+temp) + (1.0f-b)*eps/6.0f) );\n"
"}\n"
"\n"
"inline float totalLinearSigma(float4 vox, read_only image2d_t MCS, float e) {\n"
"	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;\n"
"	float4 mcs = read_imagef(MCS, sampler, (float2)(e/0.5f - 0.5f, 0.5f));\n"
"\n"
"	return mcs.s0*vox.s3 + (mcs.s1 + mcs.s2 + mcs.s3)*vox.s2;\n"
"}\n"
"\n"
"inline float gamma(PS * particle) {\n"
"	return (particle->energy + particle->mass) / particle->mass;\n"
"}\n"
"inline float beta(PS * particle) {\n"
"	return sqrt(1 - particle->mass*particle->mass/((particle->energy + particle->mass)*(particle->energy + particle->mass)));\n"
"}\n"
"\n"
"inline float maxDeltaElectronEnergy(PS * particle) {\n"
"	return (2 * ME*beta(particle)*beta(particle)*gamma(particle)*gamma(particle)) / (1 + 2 * gamma(particle)*ME / particle->mass + ME*ME / (particle->mass*particle->mass));\n"
"}\n"
"\n"
"inline float maxOxygenEnergy(PS * particle){\n"
"	return (2 * MO*beta(particle)*beta(particle)*gamma(particle)*gamma(particle)) / (1 + 2 * gamma(particle)*MO / particle->mass + MO*MO / (particle->mass*particle->mass));\n"
"}\n"
"\n"
"inline float momentumSquare(PS * particle) {\n"
"	return particle->energy*particle->energy + 2 * particle->energy*particle->mass;\n"
"}\n"
"\n"
"float radiationLength(float4 vox)  \n"
"//	calculate the radiation length ratio \\rho_wX_w/(\\rhoX_0(\\rho)\n"
"{\n"
"	float ratio;\n"
"	if (vox.s2 >= 0.9)\n"
"	{\n"
"		ratio = 1.19f + 0.44f*log(vox.s2 - 0.44f);\n"
"	}\n"
"	else if (vox.s2 >= 0.26)\n"
"	{\n"
"		ratio = 1.0446f - 0.2180f*vox.s2;\n"
"	}\n"
"	else\n"
"	{\n"
"		ratio = 0.9857f + 0.0085f*vox.s2;\n"
"	}\n"
"	return WATERDENSITY*XW / (ratio*vox.s2);\n"
"}\n"
"\n"
"float3 transform(float3 dir, float theta, float phi){\n"
"	// if original direction is along z-axis\n"
"	float temp = 1.0 - ZERO;\n"
"	if (dir.z*dir.z >= temp){\n"
"		if (dir.z > 0){\n"
"			dir.x = sin(theta)*cos(phi);\n"
"			dir.y = sin(theta)*sin(phi);\n"
"			dir.z = cos(theta);\n"
"		}\n"
"		else{\n"
"			dir.x = -sin(theta)*cos(phi);\n"
"			dir.y = -sin(theta)*sin(phi);\n"
"			dir.z = -cos(theta);\n"
"		}\n"
"	}\n"
"	else{\n"
"		float u, v, w;\n"
"		u = dir.x*cos(theta) + sin(theta)*(dir.x*dir.z*cos(phi) - dir.y*sin(phi)) / sqrt(1.0 - dir.z*dir.z);\n"
"		v = dir.y*cos(theta) + sin(theta)*(dir.y*dir.z*cos(phi) + dir.x*sin(phi)) / sqrt(1.0 - dir.z*dir.z);\n"
"		w = dir.z*cos(theta) - sqrt(1.0f - dir.z*dir.z)*sin(theta)*cos(phi);\n"
"\n"
"		dir.x = u;\n"
"		dir.y = v;\n"
"		dir.z = w;\n"
"	}\n"
"	\n"
"	return normalize(dir);\n"
"\n"
"\n"
"}\n"
"\n"
"\n"
"float3 getMovement(float3 value, int crossBound, float3 pos){\n"
"	float3 zero = ZERO*fabs(pos);\n"
"	zero = ZERO;\n"
"//	printf(\"value = %v3f, zero = %v3f\\n\", value, zero);\n"
"	switch(crossBound){\n"
"	case(1):\n"
"		if(fabs(value.x) < zero.x)\n"
"			value.x = value.x >= 0 ? zero.x : -zero.x;\n"
"		break;\n"
"\n"
"	case(2):\n"
"		if(fabs(value.y) < zero.y)\n"
"			value.y = value.y >= 0 ? zero.y : -zero.y;\n"
"		break;\n"
"\n"
"	case(3):\n"
"		if(fabs(value.z) < zero.z)\n"
"			value.z = value.z >= 0 ? zero.z : -zero.z;\n"
"		break;\n"
"\n"
"	case(0):\n"
"		break;\n"
"	}\n"
"//	printf(\"value = %v3f\\n\", value);\n"
"	return value;\n"
"}\n"
"\n"
"inline void update(PS * thisOne, float stepLength, float energyTransfer, float theta, float phi, int crossBound, float deflection){	\n"
"	\n"
"	float3 movement = getMovement(thisOne->dir*stepLength, crossBound, thisOne->pos);	\n"
"//	printf(\"dir = %v3f, pos = %v3f, movement = %v3f\\n\", thisOne->dir, thisOne->pos, movement);\n"
"	thisOne->pos += movement;\n"
"	thisOne->dir = transform(thisOne->dir, theta, phi);\n"
"	thisOne->energy -= energyTransfer;\n"
"\n"
"}\n"
"\n"
"inline void atomicAdd(volatile global float * source, const float operand) {\n"
"	union {\n"
"		unsigned int intVal;\n"
"		float floatVal;\n"
"	} newVal;\n"
"	union {\n"
"		unsigned int intVal;\n"
"		float floatVal;\n"
"	} prevVal;\n"
"\n"
"	do {\n"
"		prevVal.floatVal = *source;\n"
"		newVal.floatVal = prevVal.floatVal + operand;\n"
"	} while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);\n"
"\n"
"}\n"
"\n"
"\n"
"void score(global float * doseCounter, int absIndex, int nVoxels, float energyTransfer, float stepLength, int * iseed, \n"
"			read_only image2d_t MSPR, PS * thisOne, float material, int scoringQuantity){\n"
"\n"
"\n"
"	// choose a dose counter\n"
"	int doseCounterId = convert_int_rtn(MTrng(iseed)*(float)(NDOSECOUNTERS));\n"
"\n"
"	volatile global float * counter = &doseCounter[absIndex + doseCounterId * nVoxels];\n"
"	sampler_t dataSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;\n"
"\n"
"\n"
"	switch(scoringQuantity){\n"
"	case(0): //dose2medium\n"
"		atomicAdd(counter, energyTransfer*thisOne->weight);\n"
"		break;\n"
"	case(1): //dose2water\n"
"		energyTransfer = energyTransfer*thisOne->weight/read_imagef(MSPR, dataSampler, (float2)(thisOne->energy - 0.5f, material + 0.5f)).s0;\n"
"		atomicAdd(counter, energyTransfer);\n"
"		break;\n"
"	case(3): //LET\n"
"		stepLength = stepLength > ZERO ? stepLength : ZERO;\n"
"		atomicAdd(counter, energyTransfer*energyTransfer*thisOne->weight*thisOne->weight/stepLength);\n"
"		break;\n"
"	}\n"
"}\n"
"\n"
"void scoreFluence(global float * doseCounter, int absIndex, int nVoxels, float fluence, global int * mutex, int * iseed, int scoringQuantity){\n"
"	if(scoringQuantity != 2) return;\n"
"	// choose a dose counter\n"
"	int doseCounterId = convert_int_rtn(MTrng(iseed)*(float)(NDOSECOUNTERS));\n"
"	volatile global float * counter = &doseCounter[absIndex + doseCounterId * nVoxels];\n"
"	atomicAdd(counter, fluence);\n"
"}\n"
"\n"
"\n"
"void store(PS * newOne, __global PS * secondary, volatile __global uint * nSecondary, global int * mutex2Secondary){\n"
"	if(*nSecondary == 0){\n"
"//		printf(\"\\n secondary particle overflow!!!\\n\");\n"
"		return;	\n"
"	}\n"
"	\n"
"	uint ind = atomic_dec(nSecondary);\n"
"	secondary[ind-1] = *newOne;\n"
"\n"
"//	printf(\"store to # %d\\n\", *nSecondary);\n"
"//	printf(\"stored proton status: energy %f, pos %v3f, dir %v3f, ifPrimary %d, mass %f, charge %f, maxSigma %f\\n\", newOne->energy, newOne->pos, newOne->dir, newOne->ifPrimary, newOne->mass, newOne->charge, newOne->maxSigma);\n"
"\n"
"}\n"
"\n"
"\n"
"\n"
"void ionization(PS * thisOne, global float * doseCounter, int absIndex, int nVoxels, \n"
"				int * iseed, float stepLength, read_only image2d_t MSPR, float material, int scoringQuantity){\n"
"\n"
"	float E = thisOne->energy + thisOne->mass;\n"
"	float Te;\n"
"	float rand;\n"
"	while(true){\n"
"		rand = MTrng(iseed);\n"
"		Te = MINELECTRONENERGY*maxDeltaElectronEnergy(thisOne)\n"
"			/((1-rand)*maxDeltaElectronEnergy(thisOne)+rand*MINELECTRONENERGY);\n"
"		if(MTrng(iseed) < 1-beta(thisOne)*beta(thisOne)*Te/maxDeltaElectronEnergy(thisOne)+Te*Te/(2*E*E))\n"
"			break;\n"
"	}\n"
"\n"
"\n"
"	update(thisOne, 0.0f, Te, 0.0f, 0.0f, 0, 0.0f);\n"
"	score(doseCounter, absIndex, nVoxels, Te, stepLength, iseed, MSPR, thisOne, material, scoringQuantity);\n"
"\n"
"}\n"
"\n"
"void PPElastic(PS * thisOne, __global PS * secondary, volatile __global uint * nSecondary, int * iseed, global int * mutex2Secondary){\n"
"	// new sampling method used \n"
"	// dsigma/dt = exp(14.5t) + 1.4exp(10t)\n"
"	// first sample invariant momentum transfer t\n"
"	float m = MP;\n"
"	float E1 = thisOne->energy + m;\n"
"	float p1 = sqrt(E1*E1 - m*m);\n"
"	float betaCMS = p1/(E1 + MP);//lorentz factor to CMS frame\n"
"	float gammaCMS = 1.0f/sqrt(1.0f - betaCMS*betaCMS);\n"
"	float p1CMS = p1*gammaCMS - betaCMS*gammaCMS*E1;\n"
"	float t, xi;\n"
"	do{\n"
"		t = MTrng(iseed)*(-4.0f)*p1CMS*p1CMS;\n"
"		xi = MTrng(iseed)*2.4f;\n"
"	}while(xi > exp(14.5f*t*1e-6) + 1.4*exp(10.0f*t*1e-6));\n"
"	//calculate theta and energy transfer to lab system\n"
"	float E4 = (2.0f*m*m - t)/(2.0f*m);\n"
"	float energyTransfer = E4 - m;\n"
"	float E3 = E1 + m - E4;\n"
"	float p4 = sqrt(E4*E4 - m*m);\n"
"	float p3 = sqrt(E3*E3 - m*m);\n"
"	float costhe = (t - 2.0f*m*m + 2.0f*E1*E3)/(2*p1*p3);\n"
"\n"
"//	if(costhe > 1.0f + ZERO || costhe < -1.0f - ZERO){\n"
"//		printf(\"nan from PPE, cos(theta) = %f\\n\", costhe);\n"
"//	}\n"
"	costhe = costhe > 1.0f ? 1.0f : costhe;\n"
"	costhe = costhe < -1.0f ? -1.0f : costhe;\n"
"	\n"
"	float theta = acos(costhe);\n"
"	float phi = 2*PI*MTrng(iseed);\n"
"	update(thisOne, 0.0f, energyTransfer, theta, phi, 0, 0.0f);\n"
"\n"
"	// compute angular deflection of recoil proton and store in secondary particle container\n"
"	float cosalpha = (p1 - p3*costhe)/p4;\n"
"	cosalpha = cosalpha > 1.0f ? 1.0f : cosalpha;\n"
"	cosalpha = cosalpha < -1.0f ? -1.0f : cosalpha;\n"
"	float alpha = acos(cosalpha);\n"
"	PS newOne = *thisOne;\n"
"	newOne.energy = energyTransfer;\n"
"\n"
"//	if(isnan(alpha))\n"
"//		printf(\"nan from PPE with secondary proton, cos(alpha) = %f, p1 = %f, p3 = %f, costhe = %f, p4 = %f, E4 = %f, E3 = %f\\n\", \n"
"//		(p1 - p3*costhe)/p4, p1, p3, costhe, p4, E4, E3);\n"
"\n"
"	update(&newOne, 0.0f, 0.0f, alpha, phi+PI, 0, 0.0f);\n"
"	store(&newOne, secondary, nSecondary, mutex2Secondary);\n"
"}\n"
"\n"
"\n"
"void POElastic(PS * thisOne, global float * doseCounter, int absIndex, int nVoxels, int * iseed, read_only image2d_t MSPR, float material, int scoringQuantity){\n"
"	// sample energy transferred to oxygen\n"
"	float energyTransfer;\n"
"	\n"
"	if(thisOne->energy < 7.2f){\n"
"		energyTransfer = thisOne->energy;\n"
"		score(doseCounter, absIndex, nVoxels, energyTransfer, INF,  iseed, \n"
"			MSPR, thisOne, material, scoringQuantity);\n"
"		thisOne->energy = 0;\n"
"		return;\n"
"	}\n"
"	float meanEnergy = 0.65f*exp(-0.0013f*thisOne->energy) - 0.71f*exp(-0.0177f*thisOne->energy);\n"
"	\n"
"	meanEnergy = meanEnergy < 0.0f ? 0.0f : meanEnergy;\n"
"\n"
"	do{\n"
"		energyTransfer = MTExp(iseed, meanEnergy);\n"
"	}while(energyTransfer > maxOxygenEnergy(thisOne));\n"
"\n"
"	// calculate theta, sample phi\n"
"	float temp1 = thisOne->energy*(thisOne->energy + 2.0f*thisOne->mass);\n"
"	float temp2 = (thisOne->energy- energyTransfer)*(thisOne->energy - energyTransfer + 2.0f*thisOne->mass);\n"
"	float costhe = (temp1 + temp2 - energyTransfer*(energyTransfer + 2.0f*MO))/ 2.0f /sqrt(temp1*temp2 );\n"
"	\n"
"//	if(costhe > 1.0f || costhe < -1.0f){\n"
"//		printf(\"nan from POE\\n\");\n"
"//		printf(\"costhe %f\\n\", costhe);\n"
"//	}\n"
"	costhe = costhe > 1.0f ? 1.0f : costhe;\n"
"	costhe = costhe < -1.0f ? -1.0f : costhe;\n"
"	float theta = acos(costhe);\n"
"\n"
"	float phi;\n"
"	phi = MTrng(iseed)*2.0f*PI;\n"
"\n"
"	update(thisOne, 0.0f, energyTransfer, theta, phi, 0, 0.0f);\n"
"\n"
"	score(doseCounter, absIndex, nVoxels, energyTransfer, INF,  iseed, \n"
"			MSPR, thisOne, material, scoringQuantity);	\n"
"}\n"
"\n"
"void POInelastic(PS * thisOne, __global PS * secondary, volatile __global uint * nSecondary, global float * doseCounter, \n"
"				 int absIndex, int nVoxels, int * iseed, global int * mutex2Secondary, read_only image2d_t MSPR, float material, int scoringQuantity){\n"
"	float rand = MTrng(iseed);\n"
"\n"
"	float bindEnergy = EBIND;\n"
"	float minEnergy = EMINPOI;\n"
"	float remainEnergy = thisOne->energy;\n"
"	float energyDeposit = 0.0f;\n"
"\n"
"	// simulate POI\n"
"	while(true){\n"
"		if(remainEnergy - bindEnergy <= minEnergy){\n"
"			energyDeposit += remainEnergy;\n"
"			break;\n"
"		}\n"
"//		energyDeposit += bindEnergy;\n"
"		remainEnergy -= bindEnergy;\n"
"\n"
"		float energy2SecondParticle = (1.0f - pow(MTrng(iseed), 2.5f))*(remainEnergy - minEnergy) + minEnergy;\n"
"\n"
"		if(rand < 0.65f){ //proton\n"
"			PS newOne = *thisOne;\n"
"			newOne.energy = energy2SecondParticle;\n"
"//			float costhe = MTrng(iseed)*(2.0f - 2.0f*energy2SecondParticle/remainEnergy) + 2.0f*minEnergy/remainEnergy - 1.0f;\n"
"			float xi = 4.0f*energy2SecondParticle*energy2SecondParticle/remainEnergy/remainEnergy;\n"
"			float costhe = log(MTrng(iseed)*(exp(xi) - exp(-xi)) + exp(-xi))/xi;\n"
"//			if(isnan( acos(costhe)))\n"
"//				printf(\"nan from POI, cosine theta is %f, xi = %f\\n\", costhe, xi);\n"
"			costhe = costhe > 1.0f ? 1.0f : costhe;\n"
"			costhe = costhe < -1.0f ? -1.0f : costhe;\n"
"			float theta = acos(costhe);\n"
"			float phi = 2.0f*PI*MTrng(iseed);\n"
"			update(&newOne, 0.0f, 0.0f, theta, phi, 0, 0.0f);\n"
"\n"
"			store(&newOne, secondary, nSecondary, mutex2Secondary);\n"
"		}\n"
"\n"
"		else if(rand < 0.69f)//short range energy\n"
"			energyDeposit += energy2SecondParticle;\n"
"\n"
"		bindEnergy *= 0.5f;\n"
"		remainEnergy -= energy2SecondParticle;\n"
"	}\n"
"\n"
"	score(doseCounter, absIndex, nVoxels, energyDeposit, INF,  iseed, \n"
"			MSPR, thisOne, material, scoringQuantity);		\n"
"	update(thisOne, 0.0f, thisOne->energy, 0.0f, 0.0f, 0, 0.0f);\n"
"}\n"
"\n"
"\n"
"void hardEvent(PS * thisOne, float stepLength, float4 vox, read_only image2d_t MCS, read_only image2d_t MSPR, float material, global float * doseCounter,\n"
"			 int absIndex, int nVoxels, global PS * secondary, volatile __global uint * nSecondary, int * iseed, global int * mutex2Secondary, int scoringQuantity){\n"
"	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;\n"
"	float4 mcs = read_imagef(MCS, sampler, (float2)(thisOne->energy/0.5f - 0.5f, 0.5f));\n"
"	float sigIon = mcs.s0*vox.s3;\n"
"	float sigPPE = mcs.s1*vox.s2;\n"
"	float sigPOE = mcs.s2*vox.s2;\n"
"	float sigPOI = mcs.s3*vox.s2;\n"
"	float sig = sigIon + sigPPE + sigPOE + sigPOI;\n"
"\n"
"	float rand = MTrng(iseed);\n"
"	rand *= thisOne->maxSigma > sig ? thisOne->maxSigma : sig;\n"
"	if(rand < sigIon){\n"
"		ionization(thisOne, doseCounter, absIndex, nVoxels, iseed, stepLength, MSPR, material, scoringQuantity);\n"
"		return;\n"
"	}\n"
"	else if(rand < sigIon + sigPPE){\n"
"		if(thisOne->energy > PPETHRESHOLD)\n"
"			PPElastic(thisOne, secondary, nSecondary, iseed, mutex2Secondary);\n"
"		return;\n"
"	}\n"
"	else if(rand < sigIon + sigPPE + sigPOE){\n"
"		if(thisOne->energy > POETHRESHOLD)\n"
"			POElastic(thisOne, doseCounter, absIndex, nVoxels, iseed, MSPR, material, scoringQuantity);\n"
"		return;\n"
"	} \n"
"	else if(rand < sigIon+sigPPE + sigPOE + sigPOI){\n"
"		if(thisOne->energy > POITHRESHOLD)\n"
"			POInelastic(thisOne, secondary, nSecondary, doseCounter, absIndex, nVoxels, iseed, mutex2Secondary, MSPR, material, scoringQuantity);\n"
"		return;\n"
"	}\n"
"}\n"
"\n"
"void rayTrace(PS * particle, int3 phantomSize, float3 voxSize){\n"
"	float3 phantomBoundary1, phantomBoundary2;\n"
"	phantomBoundary1 = -convert_float3(phantomSize)*voxSize/2.0f;\n"
"	phantomBoundary2 = convert_float3(phantomSize)*voxSize/2.0f;\n"
"\n"
"	float3 delta1, delta2, delta;\n"
"	delta1 = (phantomBoundary1 - particle->pos)/particle->dir;\n"
"	delta2 = (phantomBoundary2 - particle->pos)/particle->dir;\n"
"	delta =	fmin(delta1, delta2);\n"
"\n"
"	float translation = fmax(fmax(delta.x, delta.y), delta.z);\n"
"//	printf(\"particle pos = %f, %f, %f, dir = %f, %f, %f, delta = %v3f, translation = %f\\n\", \n"
"//		particle->pos.x, particle->pos.y, particle->pos.z, particle->dir.x, particle->dir.y, particle->dir.z, delta, translation);	\n"
"	update(particle, translation + 1e-5, 0.0, 0.0, 0.0, 0, 0.0);\n"
"//	printf(\"particle pos = %f, %f, %f, dir = %f, %f, %f, delta = %v3f, translation = %f\\n\", \n"
"//		particle->pos.x, particle->pos.y, particle->pos.z, particle->dir.x, particle->dir.y, particle->dir.z, delta, translation);\n"
"}\n"
"\n"
"\n"
"__kernel void propagate(__global PS * particle, __global float * doseCounter,\n"
"		__read_only image3d_t voxels, float3 voxSize, __read_only image2d_t MCS, __read_only image2d_t RSPW, \n"
"		__read_only image2d_t MSPR, __global PS * secondary, volatile __global uint * nSecondary, int randSeed, \n"
"		__global int * mutex, int scoringQuantity){\n"
"	size_t gid = get_global_id(0);\n"
"	PS thisOne = particle[gid];\n"
"	\n"
"\n"
"//	printf(\"size of PS in kernel = %d\\n\", sizeof(PS));\n"
"	\n"
"//	if(thisOne.ifPrimary == 0){\n"
"//		printf(\"simulate secondary proton\\n\");\n"
"//		printf(\"simulated proton status: energy %f, pos %v3f, dir %v3f, ifPrimary %d, mass %f, charge %f, maxSigma %f\\n\", thisOne.energy, thisOne.pos, thisOne.dir, thisOne.ifPrimary, thisOne.mass, thisOne.charge, thisOne.maxSigma);\n"
"//	}\n"
"\n"
"\n"
"\n"
"	int3 phantomSize = (int3)(get_image_width(voxels), get_image_height(voxels), get_image_depth(voxels));\n"
"//	printf(\"phantomSize = %v3d\\n\", phantomSize);\n"
"	int nVoxels = phantomSize.x * phantomSize.y * phantomSize.z;\n"
"\n"
"	float stepLength, stepInWater, thisMaxStep, step2bound, energyTransfer, sigma1, sigma2, sigma, sampledStep, variance, theta0, theta, phi, es;\n"
"	es = 17.5f;\n"
"	int3 voxIndex;\n"
"	int absIndex, crossBound = 0;\n"
"	int iseed[2];\n"
"	iseed[0] = gid;\n"
"	iseed[1] = randSeed;\n"
"	jswRand(iseed);\n"
"	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
"	sampler_t dataSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;\n"
"	float4 vox;\n"
"	bool ifHard;\n"
"	int step = 0;\n"
"	\n"
"	if(!ifInsidePhantom(thisOne.pos, voxSize, phantomSize, &voxIndex, &absIndex))\n"
"		rayTrace(&thisOne, phantomSize, voxSize);\n"
"\n"
"	while (ifInsidePhantom(thisOne.pos, voxSize, phantomSize, &voxIndex, &absIndex)){\n"
"		step++;\n"
"\n"
"\n"
"		vox = read_imagef(voxels, voxSampler, (float4)(convert_float3(voxIndex), 0.0f));\n"
"		if(vox.s0 < -800.0f){\n"
"			update(&thisOne, step2VoxBoundary(thisOne.pos, thisOne.dir, voxSize, &crossBound, phantomSize, voxIndex), 0, 0, 0, crossBound, 0.0f);\n"
"			continue;\n"
"		}\n"
"\n"
"\n"
"		if (thisOne.energy <= MINPROTONENERGY){\n"
"			stepInWater = thisOne.energy / read_imagef(RSPW, dataSampler, (float2)(thisOne.energy/0.5f - 0.5f, 0.5f)).s0;\n"
"			stepLength = stepInWater*WATERDENSITY / vox.s2 / read_imagef(MSPR, dataSampler, (float2)(thisOne.energy - 0.5f, vox.s1 + 0.5f)).s0;\n"
"			energyTransfer = thisOne.energy;			\n"
"			score(doseCounter, absIndex, nVoxels, energyTransfer, stepLength, iseed, MSPR, &thisOne, vox.s1, scoringQuantity);\n"
"			scoreFluence(doseCounter, absIndex, nVoxels, stepLength*thisOne.weight, mutex, iseed, scoringQuantity);\n"
"			return;\n"
"		}\n"
"\n"
"		// rescale maxStep to let energy transferred in one step < MAXENERGYRATIO\n"
"		thisMaxStep = MAXSTEP;\n"
"		step2bound = step2VoxBoundary(thisOne.pos, thisOne.dir, voxSize, &crossBound, phantomSize, voxIndex);\n"
"		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, thisMaxStep);\n"
"		if (energyTransfer > MAXENERGYRATIO*thisOne.energy){\n"
"			stepInWater = MAXENERGYRATIO*thisOne.energy / read_imagef(RSPW, dataSampler, (float2)((1 - 0.5f*MAXENERGYRATIO)*thisOne.energy/0.5f - 0.5f, 0.5f)).s0;\n"
"			thisMaxStep = stepInWater*WATERDENSITY / vox.s2 / read_imagef(MSPR, dataSampler, (float2)((1 - 0.5f*MAXENERGYRATIO)*thisOne.energy - 0.5f, vox.s1 + 0.5f)).s0;\n"
"		}\n"
"\n"
"\n"
"\n"
"		// get linear macro cross section to sample a step\n"
"		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, thisMaxStep);\n"
"		sigma1 = totalLinearSigma(vox, MCS, thisOne.energy);\n"
"		sigma2 = totalLinearSigma(vox, MCS, thisOne.energy - energyTransfer);\n"
"		sigma = sigma1 > sigma2 ? sigma1 : sigma2;\n"
"\n"
"\n"
"		// sample one step\n"
"		sampledStep = -log(MTrng(iseed)) / sigma;\n"
"		stepLength = sampledStep < thisMaxStep ? sampledStep : thisMaxStep;\n"
"		if (stepLength >= step2bound){\n"
"			ifHard = false;\n"
"			stepLength = step2bound;\n"
"		}\n"
"		else{\n"
"			ifHard = true;\n"
"			crossBound = 0;\n"
"		}\n"
"\n"
"//		if(step > 2000)\n"
"//			printf(\"sample step = %f, maxStep = %f, if crossBound = %d, step2Bound = %f, steplength: %f\\n\", sampledStep, thisMaxStep, crossBound, step2bound, stepLength);\n"
"\n"
"		// get energy transferred (plus energy straggling) in this sampled step\n"
"		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, stepLength);\n"
"		variance = TWOPIRE2MENEW*vox.s3*stepLength*thisOne.charge*thisOne.charge*fmin(MINELECTRONENERGY, maxDeltaElectronEnergy(&thisOne))*(1.0f / (beta(&thisOne)*beta(&thisOne)) - 0.5f);\n"
"		energyTransfer += MTGaussian(iseed) * sqrt(variance);\n"
"		energyTransfer = energyTransfer > 0 ? energyTransfer : -energyTransfer;\n"
"		energyTransfer = energyTransfer < thisOne.energy ? energyTransfer : thisOne.energy;\n"
"\n"
"\n"
"		// deflection\n"
"		if(thisOne.energy < 70.0f)\n"
"			es = 15.5f;\n"
"\n"
"\n"
"		theta0 = es*thisOne.charge*sqrt(stepLength/radiationLength(vox))/beta(&thisOne)/sqrt(momentumSquare(&thisOne));\n"
"		theta = MTGaussian(iseed) * theta0;\n"
"		phi = 2.0f*PI*MTrng(iseed);\n"
"		thisOne.maxSigma = sigma;\n"
"		update(&thisOne, stepLength, energyTransfer, theta, phi, crossBound, 0.0f);		\n"
"		score(doseCounter, absIndex, nVoxels, energyTransfer, stepLength, iseed, MSPR, &thisOne, vox.s1, scoringQuantity);\n"
"		scoreFluence(doseCounter, absIndex, nVoxels, stepLength*thisOne.weight, mutex, iseed, scoringQuantity);\n"
"\n"
"		//hard event\n"
"		if(!ifHard)\n"
"			continue;\n"
"\n"
"		hardEvent(&thisOne, stepLength, vox, MCS, MSPR, vox.s1, doseCounter, absIndex, nVoxels, secondary, nSecondary, iseed, mutex, scoringQuantity);\n"
"\n"
"	}\n"
"	\n"
"}\n"
;





ParticleStatus::ParticleStatus(){
}

ParticleStatus::ParticleStatus(OpenCLStuff & stuff, cl_float T, cl_float2 width_, cl_float3 sourceCenter_, cl_ulong nParticles_)
	:energy(T), width(width_), sourceCenter(sourceCenter_), nParticles(nParticles_) {
	int err;
	buildProgram(stuff);
}

void ParticleStatus::buildProgram(OpenCLStuff & stuff){

	std::string source(Particle_ocl);
/*	OpenCLStuff::convertToString("obfuscation_output/ReplacementFor_ParticleStatus.c", source);
	std::string include1, include2;
	OpenCLStuff::convertToString("obfuscation_output/ReplacementFor_Macro.h", include1);
	OpenCLStuff::convertToString("obfuscation_output/ReplacementFor_randomKernel.h", include2);
	source = include1 + include2 + source;
*/


	int err;
	program = cl::Program(stuff.context, source);
	err = program.build("-cl-single-precision-constant -I.");
	std::string info;
	info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(stuff.device);
	if (err != 0){
		std::cout << "build result: " << err << std::endl;
		std::cout << info << std::endl;
		exit(-1);
	}

	loadSourceKernel.push_back(cl::make_kernel < cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl_float3 >(program, "loadSource", &err));
	propagateKernel.push_back(cl::make_kernel< cl::Buffer &, cl::Buffer &, cl::Image3D &, cl_float3, cl::Image2D &, cl::Image2D &, cl::Image2D &, cl::Buffer &, cl::Buffer &, cl_int, cl::Buffer &, cl_int>(program, "propagate", &err));

}

int ParticleStatus::reload(OpenCLStuff & stuff, cl_float * T, cl_float3 * pos, cl_float3 * dir, cl_float * weight, cl_uint nHistory, cl_float3 translation){
	if (nParticles > stuff.nBatch()){
		load(stuff, stuff.nBatch(), nHistory - nParticles, T, pos, dir, weight, translation);
		nParticles -= stuff.nBatch();
	}
	else{
		load(stuff, nParticles, nHistory - nParticles, T, pos, dir, weight, translation);
		nParticles = 0;
	}
	return 1;
}

void ParticleStatus::load(OpenCLStuff & stuff, cl_uint nParticles_, cl_uint offset, cl_float * T, cl_float3 * pos, cl_float3 * dir, cl_float * weight, cl_float3 translation){
	particleStatus.clear();
	int err;
	cl::Buffer energyBuffer(stuff.context, &T[offset], &T[offset + nParticles_ - 1], true, true, &err);
	cl::Buffer posBuffer(stuff.context, &pos[offset], &pos[offset + nParticles_ - 1], true, true, &err);
	cl::Buffer dirBuffer(stuff.context, &dir[offset], &dir[offset + nParticles_ - 1], true, true, &err);
	cl::Buffer weightBuffer(stuff.context, &weight[offset], &weight[offset + nParticles_ - 1], true, true, &err);
	particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS) * nParticles_, NULL, &err));


	globalRange = cl::NDRange(nParticles_);
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	loadSourceKernel.back()(arg, particleStatus[0], energyBuffer, posBuffer, dirBuffer, weightBuffer, translation);
}


void ParticleStatus::load(OpenCLStuff & stuff, cl_ulong nParticles_, cl_float T, cl_float2 width, cl_float3 sourceCenter_, cl_float mass, cl_float charge)
{
	particleStatus.clear();
	int err;
	int tempSize = sizeof(PS);
	particleStatus.push_back(cl::Buffer(stuff.context, CL_MEM_READ_WRITE, sizeof(PS) * nParticles_, NULL, &err));

	cl::make_kernel<cl::Buffer &, cl_float, cl_float2, cl_float3, cl_float, cl_float, cl_int> initParticlesKernel(program, "initParticles");

	globalRange = cl::NDRange(nParticles_);
	cl::EnqueueArgs arg (stuff.queue, globalRange);
	cl_int randSeed = rand();
	initParticlesKernel(arg, particleStatus[0], T, width, sourceCenter_, mass, charge, randSeed);


//	PS * particleTest = new PS[nParticles_]();
//	err = stuff.queue.enqueueReadBuffer(particleStatus[0], CL_TRUE, 0, sizeof(PS) * nParticles_, particleTest);
}


ParticleStatus::~ParticleStatus(){
}

int ParticleStatus::reload(OpenCLStuff & stuff){
	if (nParticles == 0)
		return 0;
	if (nParticles > stuff.nBatch()){
		nParticles -= stuff.nBatch();
		load(stuff, stuff.nBatch(), energy, width, sourceCenter, MP, CP);
	}
	else{
		load(stuff, nParticles, energy, width, sourceCenter, MP, CP);
		nParticles = 0;
	}
	return 1;
}


void ParticleStatus::propagate(OpenCLStuff & stuff, Phantom * phantom, MacroCrossSection * macroSigma, 
	RSPW * resStpPowWater, MSPR * massStpPowRatio, ParticleStatus * secondary, cl_int scoringQuantity){
	int err;
	cl::EnqueueArgs arg(stuff.queue, globalRange);
	
	
	cl_int randSeed = rand();
//	std::cout << randSeed << std::endl;
	cl::Buffer mutex(stuff.context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
	cl_int initialMutext = 0;
	stuff.queue.enqueueWriteBuffer(mutex, CL_TRUE, 0, sizeof(cl_int), &initialMutext);


	stuff.queue.finish();
	propagateKernel.back()(arg, particleStatus.back(), phantom->doseCounterGPU(), phantom->voxelGPU(), phantom->voxelSize(), macroSigma->gpu(),
		resStpPowWater->gpu(), massStpPowRatio->gpu(), secondary->particleStatus[0], secondary->nSecondBuffer(), randSeed, mutex, scoringQuantity);
	//std::cout << "number of protons in this batch: " << *globalRange << std::endl;
	stuff.queue.finish();

}