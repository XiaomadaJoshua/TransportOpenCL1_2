#include "randomKernel.h"
#include "Macro.h"

typedef struct __attribute__ ((packed)) ParticleStatus{
	float3 pos, dir;
	float energy, maxSigma, mass, charge;
	int ifPrimary;
}PS;


__kernel void initParticles(__global PS * particle, float T, float2 width, float3 sourceCenter, float m, float c, int randSeed){
	size_t gid = get_global_id(0);
	
//	if(gid == 1){
//		int size = sizeof(PS);
//		printf("size of PS: %d\n", size);
//	}
	particle[gid].pos.z = -distance(sourceCenter, (float3)(0.0f, 0.0f, 0.0f));
	int iseed[2];
	iseed[0] = randSeed;
	iseed[1] = gid;
	MTrng(iseed);
	particle[gid].pos.x = (MTrng(iseed) - 0.5f) * width.x;
	particle[gid].pos.y = (MTrng(iseed) - 0.5f) * width.y;

	particle[gid].dir.x = 0.0f;
	particle[gid].dir.y = 0.0f;
	particle[gid].dir.z = 1.0f;

	particle[gid].energy = T;
	particle[gid].maxSigma = 0.0f;
	particle[gid].mass = m;
	particle[gid].charge = c;
	particle[gid].ifPrimary = 1;
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

float energyInOneStep(float4 vox, PS * particle, read_only image1d_t RSPW, read_only image2d_t MSPR, float stepLength) {
	//calculate equivalent step in water

	float stepInWater;
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	float4 mspr = read_imagef(MSPR, sampler, (float2)(particle->energy - 0.5f, vox.s1 + 0.5f));
	stepInWater = mspr.s0*stepLength*vox.s2/WATERDENSITY;

	//calculate energy transfer
	float4 rspw = read_imagef(RSPW, sampler, particle->energy/0.5f - 0.5f);


	float de1 = stepInWater*rspw.s0;
	float b = rspw.s1;
	float temp = particle->energy/particle->mass;
	float eps = de1/particle->energy;
	return de1*(1.0f + eps/(1.0f+temp)/(2.0f+temp) + eps*eps*(2.0f+2.0f*temp+temp*temp)/(1.0f+temp)/(1.0f+temp)/(2.0f+temp)/(2.0f+temp)
			- b*eps*(0.5f+2.0f*eps/3.0f/(1.0f+temp)/(2.0f+temp) + (1.0f-b)*eps/6.0f) );
}

inline float totalLinearSigma(float4 vox, read_only image1d_t MCS, float e) {
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	float4 mcs = read_imagef(MCS, sampler, e/0.5f - 0.5f);
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

/*float3 transform(float3 dir, float theta, float phi){
	// if original direction is along z-axis
	printf("direction before transform %v3f\n", dir);
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
		w = dir.z*cos(theta) - sqrt(1.0f - dir.z*dir.z)*sin(theta)*cos(phi);

		dir.x = u;
		dir.y = v;
		dir.z = w;
	}
	
	if(any(isnan(dir))){
		printf("transform result: %v3f\n", dir);
		printf("theta %f, phi %f\n", theta, phi);
	}
	printf("direction after transform %v3f\n", dir);

	dir = normalize(dir);
	printf("direction after normalization %v3f\n", dir);
	// if norm does not equal to 1
	return normalize(dir);

}*/

void transform(float3 * dir, float theta, float phi){
	// if original direction is along z-axis
//	printf("direction before transform %f %f %f\n", (*dir).x, (*dir).y, (*dir).z);
	float temp = 1.0 - ZERO;
	if ((*dir).z*(*dir).z >= temp){
		if ((*dir).z > 0){
			(*dir).x = sin(theta)*cos(phi);
			(*dir).y = sin(theta)*sin(phi);
			(*dir).z = cos(theta);
		}
		else{
			(*dir).x = -sin(theta)*cos(phi);
			(*dir).y = -sin(theta)*sin(phi);
			(*dir).z = -cos(theta);
		}
	}
	else{
		float u, v, w;
		u = (*dir).x*cos(theta) + sin(theta)*((*dir).x*(*dir).z*cos(phi) - (*dir).y*sin(phi)) / sqrt(1.0 - (*dir).z*(*dir).z);
		v = (*dir).y*cos(theta) + sin(theta)*((*dir).y*(*dir).z*cos(phi) + (*dir).x*sin(phi)) / sqrt(1.0 - (*dir).z*(*dir).z);
		w = (*dir).z*cos(theta) - sqrt(1.0f - (*dir).z*(*dir).z)*sin(theta)*cos(phi);

		(*dir).x = u;
		(*dir).y = v;
		(*dir).z = w;
	}
	
	if(any(isnan(*dir))){
		printf("transform result: %v3f\n", *dir);
		printf("theta %f, phi %f\n", theta, phi);
	}
//	printf("direction after transform %f %f %f\n", (*dir).x, (*dir).y, (*dir).z);

	*dir = normalize(*dir);
//	printf("direction after normalization %f %f %f\n", (*dir).x, (*dir).y, (*dir).z);

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

inline void update(PS * thisOne, float stepLength, float energyTransfer, float theta, float phi, int crossBound, float deflection){	
//	printf("before update dir %f %f %f, pos %f %f %f\n", thisOne->dir.x, thisOne->dir.y, thisOne->dir.z, thisOne->pos.x, thisOne->pos.y, thisOne->pos.z);
	
	float3 movement = getMovement(thisOne->dir*stepLength, crossBound);	
/*	if(ABS(deflection) > 0.0){
		float3 n = thisOne->dir;
		transform(&n, PI/2, 0.0f);
//		if(ABS(dot(n, thisOne->dir)) > 100.0*ZERO)
//			printf("error in deflection n: %f %f %f, dir: %f %f %f\n", n.x, n.y, n.z, thisOne->dir.x, thisOne->dir.y, thisOne->dir.z);
		n = normalize(n)*deflection;
		movement += n;
	}*/
	thisOne->pos += movement;
	transform(&(thisOne->dir), theta, phi);
	thisOne->energy -= energyTransfer;
//	printf("after update dir %f %f %f, pos %f %f %f\n", thisOne->dir.x, thisOne->dir.y, thisOne->dir.z, thisOne->pos.x, thisOne->pos.y, thisOne->pos.z);

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

}

void getMutex(global int * mutex){
	int occupied = atomic_xchg(mutex, 1);
	while(occupied > 0){
		occupied = atomic_xchg(mutex, 1);
	}
}

void releaseMutex(global int * mutex){
	int prevVal = atomic_xchg(mutex, 0);
}

	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 5 in float8 is primary dose,
	// 6 in float8 is secondary dose, 7 in float8 is heavy dose.
void score(global float8 * doseCounter, int absIndex, int nVoxels, float energyTransfer, float stepLength, int ifPrimary, int * iseed){
	// choose a dose counter
	int doseCounterId = convert_int_rtn(MTrng(iseed)*NDOSECOUNTERS);

	volatile global float * counter = &doseCounter[absIndex + doseCounterId * nVoxels];
	atomicAdd(counter, energyTransfer);
	if(ifPrimary == 1){
		atomicAdd(counter + 5, energyTransfer);
		if(stepLength > 0)
			atomicAdd(counter + 3, energyTransfer*energyTransfer/stepLength);
	}
	else{
		atomicAdd(counter + 6, energyTransfer);
		if(stepLength > 0)
			atomicAdd(counter + 4, energyTransfer*energyTransfer/stepLength);
	}
}

void scoreFluence(global float8 * doseCounter, int absIndex, int nVoxels, int ifPrimary, float fluence, global int * mutex, int * iseed){
	// choose a dose counter
	int doseCounterId = convert_int_rtn(MTrng(iseed)*NDOSECOUNTERS);

//	getMutex(mutex);
	volatile global float * counter = &doseCounter[absIndex + doseCounterId * nVoxels];
	if(ifPrimary == 1)
		atomicAdd(counter + 1, fluence);

	else
		atomicAdd(counter + 2, fluence);

//	if(absIndex == 25*51 + 25 && ifPrimary == 1)
//		printf("entrance fluence %f\n", fluence);
//	releaseMutex(mutex);
}

void scoreHeavy(global float8 * doseCounter, int absIndex, int nVoxels, float energyTransfer, int * iseed){
	// choose a dose counter
	int doseCounterId = convert_int_rtn(MTrng(iseed)*NDOSECOUNTERS);
	volatile global float * counter = &doseCounter[absIndex + doseCounterId * nVoxels];
	atomicAdd(counter, energyTransfer);
	atomicAdd(counter + 7, energyTransfer);
}

void store(PS * newOne, __global PS * secondary, volatile __global uint * nSecondary, global int * mutex2Secondary){
	if(*nSecondary == 0){
		printf("\n secondary particle overflow!!!\n");
		return;	
	}
	
//	getMutex(mutex2Secondary);
	uint ind = atomic_dec(nSecondary);
	secondary[ind-1] = *newOne;
//	releaseMutex(mutex2Secondary);

//	printf("store to # %d\n", *nSecondary);
//	printf("stored proton status: energy %f, pos %v3f, dir %v3f, ifPrimary %d, mass %f, charge %f, maxSigma %f\n", newOne->energy, newOne->pos, newOne->dir, newOne->ifPrimary, newOne->mass, newOne->charge, newOne->maxSigma);

}


void ionization(PS * thisOne, global float8 * doseCounter, int absIndex, int nVoxels, int * iseed, float stepLength){
	

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


	update(thisOne, 0.0f, Te, 0.0f, 0.0f, 0, 0.0f);
	score(doseCounter, absIndex, nVoxels, Te, stepLength, thisOne->ifPrimary, iseed);

}

void PPElastic(PS * thisOne, __global PS * secondary, volatile __global uint * nSecondary, int * iseed, global int * mutex2Secondary){
	// new sampling method used 
	// dsigma/dt = exp(14.5t) + 1.4exp(10t)
	// first sample invariant momentum transfer t
	float m = MP;
	float E1 = thisOne->energy + m;
	float p1 = sqrt(E1*E1 - m*m);
	float betaCMS = p1/(E1 + MP);//lorentz factor to CMS frame
	float gammaCMS = 1.0f/sqrt(1 - betaCMS*betaCMS);
	float p1CMS = p1*gammaCMS - betaCMS*gammaCMS*E1;
	float t, xi;
	do{
		t = MTrng(iseed)*(-4.0f)*p1CMS*p1CMS;
		xi = MTrng(iseed)*2.4f;
	}while(xi > exp(14.5f*t*1e-6) + 1.4*exp(10.0f*t*1e-6));
	//calculate theta and energy transfer in lab system
	float E4 = (2.0f*m*m - t)/(2.0f*m);
	float energyTransfer = E4 - m;
	float E3 = E1 + m - E4;
	float p4 = sqrt(E4*E4 - m*m);
	float p3 = sqrt(E3*E3 - m*m);
	float costhe = (t - 2.0f*m*m + 2.0f*E1*E3)/(2*p1*p3);

	if(costhe > 1.0f + ZERO || costhe < -1.0f - ZERO){
		printf("nan from PPE, cos(theta) = %f\n", costhe);
	}
	costhe = costhe > 1.0f ? 1.0f : costhe;
	costhe = costhe < -1.0f ? -1.0f : costhe;
	
	float theta = acos(costhe);
	float phi = 2*PI*MTrng(iseed);
	update(thisOne, 0.0f, energyTransfer, theta, phi, 0, 0.0f);

	// compute angular deflection of recoil proton and store in secondary particle container
	float alpha = acos((p1 - p3*costhe)/p4);
	PS newOne = *thisOne;
	newOne.energy = energyTransfer;
	newOne.ifPrimary = 0;

	if(isnan(alpha))
		printf("nan from PPE with secondary proton, cos(alpha) = %f, p1 = %f, p3 = %f, costhe = %f, p4 = %f, E4 = %f, E3 = %f\n", 
		(p1 - p3*costhe)/p4, p1, p3, costhe, p4, E4, E3);

	update(&newOne, 0.0f, 0.0f, alpha, phi+PI, 0, 0.0f);

	if(thisOne->energy < newOne.energy){
		thisOne->ifPrimary = 0;
		newOne.ifPrimary = 1;
	}
	
	store(&newOne, secondary, nSecondary, mutex2Secondary);
}

	// let the proton with higher energy be primary proton


void POElastic(PS * thisOne, global float8 * doseCounter, int absIndex, int nVoxels, int * iseed){
	// sample energy transferred to oxygen
	float meanEnergy = 0.65f*exp(-0.0013f*thisOne->energy) - 0.71f*exp(-0.0177f*thisOne->energy);
//	meanEnergy = meanEnergy < 1.0f ? 1.0f : meanEnergy;
	float energyTransfer;
	do{
		energyTransfer = MTExp(iseed, meanEnergy);
	}while(energyTransfer > maxOxygenEnergy(thisOne));

	// calculate theta, sample phi
	float temp1 = thisOne->energy*(thisOne->energy + 2.0f*thisOne->mass);
	float temp2 = (thisOne->energy- energyTransfer)*(thisOne->energy - energyTransfer + 2.0f*thisOne->mass);
	float costhe = (temp1 + temp2 - energyTransfer*(energyTransfer + 2.0f*MO))/ 2.0f /sqrt(temp1*temp2 );
	
	if(costhe > 1.0f || costhe < -1.0f){
		printf("nan from POE\n");
		printf("costhe %f\n", costhe);
	}
	costhe = costhe > 1.0f ? 1.0f : costhe;
	costhe = costhe < -1.0f ? -1.0f : costhe;
	float theta = acos(costhe);

	float phi;
	phi = MTrng(iseed)*2.0f*PI;

	update(thisOne, 0.0f, energyTransfer, theta, phi, 0, 0.0f);
	scoreHeavy(doseCounter, absIndex, nVoxels, energyTransfer, iseed);
}

void POInelastic(PS * thisOne, __global PS * secondary, volatile __global uint * nSecondary, global float8 * doseCounter, int absIndex, int nVoxels, int * iseed, global int * mutex2Secondary){

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

		if(rand < 0.5f){ //proton
			PS newOne = *thisOne;
			newOne.energy = energy2SecondParticle;
			newOne.ifPrimary = 0;
//			float costhe = MTrng(iseed)*(2.0f - 2.0f*energy2SecondParticle/remainEnergy) + 2.0f*minEnergy/remainEnergy - 1.0f;
			float xi = 2.5f*energy2SecondParticle/remainEnergy;
			float costhe = log(MTrng(iseed)*(exp(xi) - exp(-xi)) + exp(-xi))/xi;
			float theta = acos(costhe);
			float phi = 2.0f*PI*MTrng(iseed);
			update(&newOne, 0.0f, 0.0f, theta, phi, 0, 0.0f);

			if(isnan(theta))
				printf("nan from POI, cosine theta is %f, xi = %f\n", costhe, xi);

			store(&newOne, secondary, nSecondary, mutex2Secondary);
		}

		else if(rand < 0.503f)//short range energy
			energyDeposit += energy2SecondParticle;

		bindEnergy *= 0.5f;
		remainEnergy -= energy2SecondParticle;
	}
	scoreHeavy(doseCounter, absIndex, nVoxels, energyDeposit, iseed);
	update(thisOne, 0.0f, thisOne->energy, 0.0f, 0.0f, 0, 0.0f);
}

void hardEvent(PS * thisOne, float stepLength, float4 vox, read_only image1d_t MCS, global float8 * doseCounter, int absIndex, int nVoxels, global PS * secondary, volatile __global uint * nSecondary, 
				int * iseed, global int * mutex2Secondary){
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	float4 mcs = read_imagef(MCS, sampler, thisOne->energy/0.5f - 0.5f);
	float sigIon = mcs.s0*vox.s3;
	float sigPPE = mcs.s1*vox.s2;
	float sigPOE = mcs.s2*vox.s2;
	float sigPOI = mcs.s3*vox.s2;
	float sig = sigIon + sigPPE + sigPOE + sigPOI;

	float rand = MTrng(iseed);
	rand *= thisOne->maxSigma > sig ? thisOne->maxSigma : sig;
	if(rand < sigIon){
		ionization(thisOne, doseCounter, absIndex, nVoxels, iseed, stepLength);
		return;
	}
	else if(rand < sigIon + sigPPE){
		if(thisOne->energy > PPETHRESHOLD)
			PPElastic(thisOne, secondary, nSecondary, iseed, mutex2Secondary);
		return;
	}
	else if(rand < sigIon + sigPPE + sigPOE){
		if(thisOne->energy > POETHRESHOLD)
			POElastic(thisOne, doseCounter, absIndex, nVoxels, iseed);
		return;
	} 
	else if(rand < sigIon+sigPPE + sigPOE + sigPOI){
		if(thisOne->energy > POITHRESHOLD)
			POInelastic(thisOne, secondary, nSecondary, doseCounter, absIndex, nVoxels, iseed, mutex2Secondary);
		return;
	}
}


__kernel void propagate(__global PS * particle, __global float8 * doseCounter, 
		__read_only image3d_t voxels, float3 voxSize, __read_only image1d_t MCS, __read_only image1d_t RSPW, 
		__read_only image2d_t MSPR, __global PS * secondary, volatile __global uint * nSecondary, int randSeed, __global int * mutex){

	size_t gid = get_global_id(0);
	PS thisOne = particle[gid];
	
//	if(thisOne.ifPrimary == 0){
//		printf("simulate secondary proton\n");
//		printf("simulated proton status: energy %f, pos %v3f, dir %v3f, ifPrimary %d, mass %f, charge %f, maxSigma %f\n", thisOne.energy, thisOne.pos, thisOne.dir, thisOne.ifPrimary, thisOne.mass, thisOne.charge, thisOne.maxSigma);
//	}

	int3 phantomSize = (int3)(get_image_width(voxels), get_image_height(voxels), get_image_depth(voxels));
	int nVoxels = phantomSize.x * phantomSize.y * phantomSize.z;

	float stepLength, stepInWater, thisMaxStep, step2bound, energyTransfer, sigma1, sigma2, sigma, sampledStep, variance, theta0, theta, phi, deflection, z1, z2;
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
		voxIndex = thisOne.pos / voxSize + convert_float3(phantomSize)*0.5f;
		absIndex = convert_int_rtn(voxIndex.x) + convert_int_rtn(voxIndex.y)*phantomSize.x + convert_int_rtn(voxIndex.z)*phantomSize.x*phantomSize.y;
		vox = read_imagef(voxels, voxSampler, (float4)(voxIndex, 0.0f));
//		if(absIndex != absIndex2){
//			absIndex = absIndex2;
//			scoreFluence(doseCounter, absIndex, thisOne.ifPrimary);
//		}

//		printf("voxSize %v3f, position %v3f, energy %f, voxel %v3f, abs voxel %d\n", voxSize, thisOne.pos, thisOne.energy, voxIndex, absIndex);


		if (thisOne.energy <= MINPROTONENERGY){
			stepInWater = thisOne.energy / read_imagef(RSPW, dataSampler, thisOne.energy/0.5f - 0.5f).s0;
			stepLength = stepInWater*WATERDENSITY / vox.s2 / read_imagef(MSPR, dataSampler, (float2)(thisOne.energy - 0.5f, vox.s1 + 0.5f)).s0;
			score(doseCounter, absIndex, nVoxels, thisOne.energy, stepLength, thisOne.ifPrimary, iseed);
			scoreFluence(doseCounter, absIndex, nVoxels, thisOne.ifPrimary, stepLength, mutex, iseed);
			return;
		}

		// rescale maxStep to let energy transferred in one step < MAXENERGYRATIO
		thisMaxStep = MAXSTEP;
		step2bound = step2VoxBoundary(thisOne.pos, thisOne.dir, voxSize, &crossBound);
		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, thisMaxStep);
		if (energyTransfer > MAXENERGYRATIO*thisOne.energy){
			stepInWater = MAXENERGYRATIO*thisOne.energy / read_imagef(RSPW, dataSampler, (1 - 0.5f*MAXENERGYRATIO)*thisOne.energy/0.5f - 0.5f).s0;
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

//		printf("sample step: if crossBound %d, steplength %f\n", crossBound, stepLength);

		// get energy transferred (plus energy straggling) in this sampled step
		energyTransfer = energyInOneStep(vox, &thisOne, RSPW, MSPR, stepLength);
		variance = TWOPIRE2MENEW*vox.s3*stepLength*thisOne.charge*thisOne.charge*fmin(MINELECTRONENERGY, maxDeltaElectronEnergy(&thisOne))*(1.0f / (beta(&thisOne)*beta(&thisOne)) - 0.5f);
		energyTransfer += MTGaussian(iseed) * sqrt(variance);
		energyTransfer = energyTransfer > 0 ? energyTransfer : -energyTransfer;
		energyTransfer = energyTransfer < thisOne.energy ? energyTransfer : thisOne.energy;


		// deflection
		z1 = MTGaussian(iseed);
		//z2 = MTGaussian(iseed);
		theta0 = ES*thisOne.charge*sqrt(stepLength/radiationLength(vox))/beta(&thisOne)/sqrt(momentumSquare(&thisOne));
		theta = z1 * theta0;
		//deflection = z1*stepLength*theta0/sqrt(12.0f) + z2*stepLength*theta0/2.0f;
		//deflection = ABS(deflection) < ZERO ? ZERO*deflection/ABS(deflection) : deflection;
//		printf("deflection %f\n", deflection);
		phi = 2.0f*PI*MTrng(iseed);
		thisOne.maxSigma = sigma;
		update(&thisOne, stepLength, energyTransfer, theta, phi, crossBound, 0);
		score(doseCounter, absIndex, nVoxels, energyTransfer, stepLength, thisOne.ifPrimary, iseed);
		scoreFluence(doseCounter, absIndex, nVoxels, thisOne.ifPrimary, stepLength, mutex, iseed);

		//hard event
		if(!ifHard)
			continue;

		hardEvent(&thisOne, stepLength, vox, MCS, doseCounter, absIndex, nVoxels, secondary, nSecondary, iseed, mutex);


	}
	
}