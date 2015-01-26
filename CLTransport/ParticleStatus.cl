#include "randomKernel.h"
#include "Macro.h"

typedef struct ParticleStatus{
	float3 pos, dir;
	float energy, maxSigma, mass, charge;
}PS;


__kernel void initParticles(__global PS * particle, float T, float2 width, float3 sourceCenter, float m, float c, int randSeed){
	size_t gid = get_global_id(0);
	particle[gid].pos.y = -distance(sourceCenter, (float3)(0.0f, 0.0f, 0.0f));
	int iseed[2];
	iseed[0] = randSeed;
	iseed[1] = gid;
	MTrng(iseed);
	particle[gid].pos.x = (MTrng(iseed) - 0.5f) * width.x;
	particle[gid].pos.z = (MTrng(iseed) - 0.5f) * width.y;

	particle[gid].dir.x = 0.0f;
	particle[gid].dir.y = 1.0f;
	particle[gid].dir.z = 0.0f;

	particle[gid].energy = T;
	particle[gid].maxSigma = 0.0f;
	particle[gid].mass = m;
	particle[gid].charge = c;
}

bool ifInsidePhantom(float3 pos, float3 voxSize, int3 phantomSize){
	int3 ifInside = isgreaterequal(pos, -convert_float3(phantomSize)*voxSize/2.0f) * isless(pos, convert_float3(phantomSize)*voxSize/2.0f);
	if(ifInside.x == 0 || ifInside.y == 0 || ifInside.z == 0)
		return false;
	float3 voxIndex = pos / voxSize + convert_float3(phantomSize) / 2.0f;
	int absIndex = convert_int_rtn(voxIndex.x) + convert_int_rtn(voxIndex.y)*phantomSize.x + convert_int_rtn(voxIndex.z)*phantomSize.x*phantomSize.y;
	if(absIndex >= phantomSize.x*phantomSize.y*phantomSize.z)
		return false;
	return true;
}

float step2VoxBoundary(float3 pos, float3 dir, float3 voxSize, int * cb) {
	float stepX, stepY, stepZ;
	if(dir.x < ZERO && -1.0f*dir.x > -1.0f*ZERO)
		stepX = INF;
	else if(dir.x > 0)
		stepX = (ceil(pos.x/voxSize.x)*voxSize.x - pos.x)/dir.x;
	else
		stepX = (floor(pos.x/voxSize.x)*voxSize.x - pos.x)/dir.x;

	if(dir.y < ZERO && -1.0f*dir.y > -1.0f*ZERO)
		stepY = INF;
	else if(dir.y > 0)
		stepY = (ceil(pos.y/voxSize.y)*voxSize.y - pos.y)/dir.y;
	else
		stepY = (floor(pos.y/voxSize.y)*voxSize.y - pos.y)/dir.y;

	if(dir.z < ZERO && -1.0f*dir.z > -1.0f*ZERO)
		stepZ = INF;
	else if(dir.z > 0)
		stepZ = (ceil(pos.z/voxSize.z)*voxSize.z - pos.z)/dir.z;
	else
		stepZ = (floor(pos.z/voxSize.z)*voxSize.z - pos.z)/dir.z;

	float step;
	if(stepX < stepY){
		step = stepX;
		if(step < stepZ)
			*cb = 1;
		else{
			step = stepZ;
			*cb = 3;
		}
	}
	else{
		step = stepY;
		if(step < stepZ)
			*cb = 2;
		else{
			step = stepZ;
			*cb = 3;
		}
	}
	

	return step < ZERO ? ZERO : step;
}

float energyInOneStep(float4 vox, PS * particle, read_only image2d_t RSPW, read_only image2d_t MSPR, float stepLength) {
	//calculate equivalent step in water

	float stepInWater;
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	float4 mspr = read_imagef(MSPR, sampler, (float2)(particle->energy - 0.5f, vox.s1 + 0.5f));
	stepInWater = mspr.s0*stepLength*vox.s2/WATERDENSITY;

	//calculate energy transfer
	float4 rspw = read_imagef(RSPW, sampler, (float2)(particle->energy/0.5f - 0.5f, 0.5f));


	float de1 = stepInWater*rspw.s0;
	float b = rspw.s1;
	float temp = particle->energy/particle->mass;
	float eps = de1/particle->energy;
	return de1*(1.0f + eps/(1.0f+temp)/(2.0f+temp) + eps*eps*(2.0f+2.0f*temp+temp*temp)/(1.0f+temp)/(1.0f+temp)/(2.0f+temp)/(2.0f+temp)
			- b*eps*(0.5f+2.0f*eps/3.0f/(1.0f+temp)/(2.0f+temp) + (1.0f-b)*eps/6.0f) );
}

inline float totalLinearSigma(float4 vox, read_only image2d_t MCS, float e) {
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	float4 mcs = read_imagef(MCS, sampler, (float2)(e/0.5f - 0.5f, 0.5f));
	return mcs.s0*vox.s3 + (mcs.s1 + mcs.s2 + mcs.s3)*vox.s2;
}

inline float gamma(PS * particle) {
	return (particle->energy + particle->mass) / particle->mass;
}
inline float beta(PS * particle) {
	return sqrt(1 - particle->mass*particle->mass/((particle->energy + particle->mass)*(particle->energy + particle->mass)));
}

inline float maxDeltaElectronEnergy(PS * particle) {
	return (2 * ME*beta(particle)*beta(particle)*gamma(particle)*gamma(particle)) / (1 + 2 * gamma(particle)*ME / particle->mass + ME*ME / (particle->mass*particle->mass));
}

inline float maxOxygenEnergy(PS * particle){
	return (2 * MO*beta(particle)*beta(particle)*gamma(particle)*gamma(particle)) / (1 + 2 * gamma(particle)*MO / particle->mass + MO*MO / (particle->mass*particle->mass));
}

inline float momentumSquare(PS * particle) {
	return particle->energy*particle->energy + 2 * particle->energy*particle->mass;
}

float radiationLength(float4 vox)  
//	calculate the radiation length ratio \rho_wX_w/(\rhoX_0(\rho)
{
	float ratio;
	if (vox.s2 >= 0.9)
	{
		ratio = 1.19f + 0.44f*log(vox.s2 - 0.44f);
	}
	else if (vox.s2 >= 0.26)
	{
		ratio = 1.0446f - 0.2180f*vox.s2;
	}
	else
	{
		ratio = 0.9857f + 0.0085f*vox.s2;
	}
	return WATERDENSITY*XW / (ratio*vox.s2);
}

float3 transform(float3 dir, float theta, float phi){
	// if original direction is along z-axis
	float temp = 1.0 - ZERO;
	if (dir.z*dir.z >= temp){
		if (dir.z > 0){
			dir.x = sin(theta)*cos(phi);
			dir.y = sin(theta)*sin(phi);
			dir.z = cos(theta);
		}
		else{
			dir.x = -sin(theta)*cos(phi);
			dir.y = -sin(theta)*sin(phi);
			dir.z = -cos(theta);
		}
	}
	else{
		float u, v, w;
		u = dir.x*cos(theta) + sin(theta)*(dir.x*dir.z*cos(phi) - dir.y*sin(phi)) / sqrt(1.0 - dir.z*dir.z);
		v = dir.y*cos(theta) + sin(theta)*(dir.y*dir.z*cos(phi) + dir.x*sin(phi)) / sqrt(1.0 - dir.z*dir.z);
		w = dir.z*cos(theta) - sqrt(1 - dir.z*dir.z)*sin(theta)*cos(phi);
		dir.x = u;
		dir.y = v;
		dir.z = w;
	}
	// if norm does not equal to 1
	return normalize(dir);
}

float3 getMovement(float3 value, int crossBound){
	switch(crossBound){
	case(1):
		if(value.x < ZERO && -1.0f*value.x > -1.0f*ZERO)
			value.x = value.x >= 0 ? ZERO : -ZERO;
		break;

	case(2):
		if(value.y < ZERO && -1.0f*value.y > -1.0f*ZERO)
			value.y = value.y >= 0 ? ZERO : -ZERO;
		break;

	case(3):
		if(value.z < ZERO && -1.0f*value.z > -1.0f*ZERO)
			value.z = value.z >= 0 ? ZERO : -ZERO;
		break;

	case(0):
		break;
	}

	return value;
}

inline void update(PS * thisOne, float stepLength, float energyTransfer, float theta, float phi, int crossBound){
	float3 movement = getMovement(thisOne->dir*stepLength, crossBound);
	thisOne->pos += movement;
	thisOne->dir = transform(thisOne->dir, theta, phi);
	thisOne->energy -= energyTransfer;
}

inline void atomicAdd(volatile global float * source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;

	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
	
	/*
	prevVal.floatVal = *source;
	newVal.floatVal = prevVal.floatVal + operand;
	atomic_xchg((volatile global unsigned int *)source, newVal.intVal);*/
}

void score(global float * doseCounter, int absIndex, float energyTransfer, float stepLength){
	atomicAdd(&doseCounter[absIndex], energyTransfer);
}

void store(PS * newOne, __global PS * secondary, volatile __global uint * nSecondary){
	if(*nSecondary == 0){
//		printf("\n secondary particle overflow!!!\n");
		return;	
	}
	atomic_dec(nSecondary);
//	printf("store to # %d\n", *nSecondary);
	secondary[*nSecondary] = *newOne;
}


void ionization(PS * thisOne, global float * doseCounter, int absIndex, int * iseed){
	

	float E = thisOne->energy + thisOne->mass;
	float Te;
	float rand;
	while(true){
		rand = MTrng(iseed);
		Te = MINELECTRONENERGY*maxDeltaElectronEnergy(thisOne)
			/((1-rand)*maxDeltaElectronEnergy(thisOne)+rand*MINELECTRONENERGY);
		if(MTrng(iseed) < 1-beta(thisOne)*beta(thisOne)*Te/maxDeltaElectronEnergy(thisOne)+Te*Te/(2*E*E))
			break;
	}


	update(thisOne, 0.0f, Te, 0.0f, 0.0f, 0);
	score(doseCounter, absIndex, Te, 0.0f);

}

void PPElastic(PS * thisOne, __global PS * secondary, volatile __global uint * nSecondary, int * iseed){

	// sample energy transferred
	float energyTransfer = 0.5 * thisOne->energy * MTrng(iseed);
	// sample incident proton deflection and update incident proton
	float temp1 = thisOne->energy*(thisOne->energy + 2.0*thisOne->mass);
	float temp2 = (thisOne->energy - energyTransfer)*(thisOne->energy - energyTransfer + 2.0*thisOne->mass);
	float costhe = (temp1 + temp2 - energyTransfer*(energyTransfer + 2.0*thisOne->mass))/2.0/sqrt(temp1*temp2);
	float theta = acos(costhe);
	float phi = 2*PI*MTrng(iseed);
	update(thisOne, 0.0f, energyTransfer, theta, phi, 0);

	// compute angular deflection of recoil proton and store in secondary particle container
	temp1 = (sqrt(temp1) - sqrt(temp2)*(costhe))/sqrt(energyTransfer*(energyTransfer + 2.0f*thisOne->mass));

	PS newOne = *thisOne;
	newOne.energy = energyTransfer;
	update(&newOne, 0.0f, 0.0f, acos(temp1), phi+PI, 0);
	store(&newOne, secondary, nSecondary);
}

void POElastic(PS * thisOne, global float * doseCounter, int absIndex, int * iseed){
	// sample energy transferred to oxygen
	float meanEnergy = 0.65f*exp(-0.0013f*thisOne->energy) - 0.71f*exp(-0.0177f*thisOne->energy);
	meanEnergy = meanEnergy < 1.0f ? 1.0f : meanEnergy;
	float energyTransfer;
	do{
		energyTransfer = MTExp(iseed, 1.0f/meanEnergy);
	}while(energyTransfer > maxOxygenEnergy(thisOne));

	// calculate theta, sample phi
	float temp1 = thisOne->energy*(thisOne->energy + 2.0f*thisOne->mass);
	float temp2 = (thisOne->energy- energyTransfer)*(thisOne->energy - energyTransfer + 2.0f*thisOne->mass);
	float costhe = (temp1 + temp2 - energyTransfer*(energyTransfer + 2.0f*MO))/ 2.0f /sqrt(temp1*temp2 );
	float theta = acos(costhe);
	float phi;
	phi = MTrng(iseed)*2.0f*PI;

	update(thisOne, 0.0f, energyTransfer, theta, phi, 0);
	score(doseCounter, absIndex, energyTransfer, 0.0f);
}

void POInelastic(PS * thisOne, __global PS * secondary, volatile __global uint * nSecondary, global float * doseCounter, int absIndex, int * iseed){
	float rand = MTrng(iseed);

	float bindEnergy = EBIND;
	float minEnergy = EMINPOI;
	float remainEnergy = thisOne->energy;
	float energyDeposit = 0.0f;

	// simulate POI
	while(true){
		if(remainEnergy - bindEnergy < minEnergy){
			energyDeposit += remainEnergy;
			break;
		}
		energyDeposit += bindEnergy;
		remainEnergy -= bindEnergy;

		float energy2SecondParticle = MTrng(iseed)*(remainEnergy - minEnergy) + minEnergy;
		remainEnergy -= energy2SecondParticle;

		if(rand < 0.650f){ //proton
			PS newOne = *thisOne;
			newOne.energy = energy2SecondParticle;
			float costhe = (2.0f - 2.0f*energy2SecondParticle/remainEnergy)*MTrng(iseed) + 2.0f*energy2SecondParticle/remainEnergy - 1.0f;
			float theta = acos(costhe);
			float phi = 2.0f*PI*MTrng(iseed);
			update(&newOne, 0.0f, 0.0f, theta, phi, 0);
			store(&newOne, secondary, nSecondary);
		}

		else if(rand < 0.67450f)//short range energy
			energyDeposit += energy2SecondParticle;

		bindEnergy *= 0.85f;
	}
	score(doseCounter, absIndex, energyDeposit, 0.0f);
	update(thisOne, 0.0f, thisOne->energy, 0.0f, 0.0f, 0);
}

void hardEvent(PS * thisOne, float4 vox, read_only image2d_t MCS, global float * doseCounter, int absIndex, global PS * secondary, volatile __global uint * nSecondary, int * iseed){
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	float4 mcs = read_imagef(MCS, sampler, (float2)(thisOne->energy/0.5f - 0.5f, 0.5f));
	float sigIon = mcs.s0*vox.s3;
	float sigPPE = mcs.s1*vox.s2;
	float sigPOE = mcs.s2*vox.s2;
	float sigPOI = mcs.s3*vox.s2;

	float rand = MTrng(iseed);
	rand *= thisOne->maxSigma;
	if(rand < sigIon){
		ionization(thisOne, doseCounter, absIndex, iseed);
		return;
	}
	else if(rand < sigIon + sigPPE){
		PPElastic(thisOne, secondary, nSecondary, iseed);
		return;
	}
	else if(rand < sigIon + sigPPE + sigPOE){
		POElastic(thisOne, doseCounter, absIndex, iseed);
		return;
	}
	else if(rand < sigIon+sigPPE + sigPOE + sigPOI){
		POInelastic(thisOne, secondary, nSecondary, doseCounter, absIndex, iseed);
		return;
	}
}


__kernel void propagate(__global PS * particle, __global float * doseCounter, 
		__read_only image3d_t voxels, float3 voxSize, __read_only image2d_t MCS, __read_only image2d_t RSPW, 
		__read_only image2d_t MSPR, __global PS * secondary, volatile __global uint * nSecondary, int randSeed){

	size_t gid = get_global_id(0);
	PS thisOne = particle[gid];
	int3 phantomSize = (int3)(get_image_width(voxels), get_image_height(voxels), get_image_depth(voxels));

	float stepLength, stepInWater, thisMaxStep, step2bound, energyTransfer, sigma1, sigma2, sigma, sampledStep, variance, theta, phi;
	float3 voxIndex;
	int absIndex, crossBound;
	int iseed[2];
	iseed[0] = gid;
	iseed[1] = randSeed;
	MTrng(iseed);
	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	sampler_t dataSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	float4 vox;
	bool ifHard;
	int step = 0;

	while (ifInsidePhantom(thisOne.pos, voxSize, phantomSize)){
		step++;
		voxIndex = thisOne.pos / voxSize + convert_float3(phantomSize) / 2.0f;
		absIndex = convert_int_rtn(voxIndex.x) + convert_int_rtn(voxIndex.y)*phantomSize.x + convert_int_rtn(voxIndex.z)*phantomSize.x*phantomSize.y;
		vox = read_imagef(voxels, voxSampler, (float4)(voxIndex, 0.0f));

		if (thisOne.energy <= MINPROTONENERGY){
			stepInWater = thisOne.energy / read_imagef(RSPW, dataSampler, (float2)(thisOne.energy/0.5f - 0.5f, 0.5f)).s0;
			stepLength = stepInWater*WATERDENSITY / vox.s2 / read_imagef(MSPR, dataSampler, (float2)(thisOne.energy - 0.5f, vox.s1 + 0.5f)).s0;
			score(doseCounter, absIndex, thisOne.energy, stepLength);
			return;
		}

		// rescale maxStep to let energy transferred in one step < MAXENERGYRATIO
		thisMaxStep = MAXSTEP;
		step2bound = step2VoxBoundary(thisOne.pos, thisOne.dir, voxSize, &crossBound);
		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, thisMaxStep);
		if (energyTransfer > MAXENERGYRATIO*thisOne.energy){
			stepInWater = MAXENERGYRATIO*thisOne.energy / read_imagef(RSPW, dataSampler, (float2)((1 - 0.5f*MAXENERGYRATIO)*thisOne.energy/0.5f - 0.5f, 0.5f)).s0;
			thisMaxStep = stepInWater*WATERDENSITY / vox.s2 / read_imagef(MSPR, dataSampler, (float2)((1 - 0.5f*MAXENERGYRATIO)*thisOne.energy - 0.5f, vox.s1 + 0.5f)).s0;
		}


		// get linear macro cross section to sample a step
		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, thisMaxStep);
		sigma1 = totalLinearSigma(vox, MCS, thisOne.energy);
		sigma2 = totalLinearSigma(vox, MCS, thisOne.energy - energyTransfer);
		sigma = sigma1 > sigma2 ? sigma1 : sigma2;


		// sample one step
		sampledStep = -log(MTrng(iseed)) / sigma;
		stepLength = sampledStep < thisMaxStep ? sampledStep : thisMaxStep;
		if (stepLength >= step2bound){
			ifHard = false;
			stepLength = step2bound;
		}
		else{
			ifHard = true;
			crossBound = 0;
		}

		// get energy transferred (plus energy straggling) in this sampled step
		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, stepLength);
		variance = TWOPIRE2MENEW*vox.s3*stepLength*thisOne.charge*thisOne.charge*fmin(MINELECTRONENERGY, maxDeltaElectronEnergy(&thisOne))*(1.0f / (beta(&thisOne)*beta(&thisOne)) - 0.5f);
		energyTransfer += MTGaussian(iseed) * sqrt(variance);
		energyTransfer = energyTransfer > 0 ? energyTransfer : -energyTransfer;
		energyTransfer = energyTransfer < thisOne.energy ? energyTransfer : thisOne.energy;


		// deflection
		variance = ES*ES*thisOne.charge*thisOne.charge*stepLength / (beta(&thisOne)*beta(&thisOne)*momentumSquare(&thisOne)*radiationLength(vox));
		
		theta = MTGaussian(iseed) * sqrt(variance);
		phi = 2.0f*PI*MTrng(iseed);
		thisOne.maxSigma = sigma;
		update(&thisOne, stepLength, energyTransfer, theta, phi, crossBound);
		score(doseCounter, absIndex, energyTransfer, stepLength);

		//hard event
		if(!ifHard)
			continue;

		hardEvent(&thisOne, vox, MCS, doseCounter, absIndex, secondary, nSecondary, iseed);

	}
	
}