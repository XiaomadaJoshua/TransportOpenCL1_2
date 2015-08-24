#include "Macro.h"

__kernel void initializeDoseCounter(__global float8 * doseCounter){
	size_t gid = get_global_id(0);
	doseCounter[gid] = 0.0f;
}


__kernel void finalize(__global float8 * doseBuff, __global float8 * errorBuff, read_only image3d_t voxels, float3 voxSize, uint nPaths){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);

	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	float4 vox = read_imagef(voxels, voxSampler, (float4)(idx, idy, idz, 0.0f));
	float volume = voxSize.x*voxSize.y*voxSize.z;
	float mass = vox.s2*volume;
	float8 dose = doseBuff[absId];
	float8 error = errorBuff[absId];

	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 
	//5 in float8 is primary dose, 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	dose.s0 = dose.s0/mass;
	dose.s1 = dose.s1/volume;
	dose.s2 = dose.s2/volume;
	
	if(dose.s0 > 0){
		dose.s3 = dose.s3/dose.s0/vox.s2;
		dose.s4 = dose.s4/dose.s0/vox.s2;
	}

	dose.s5 =  dose.s5/mass;
	dose.s6 =  dose.s6/mass;
	dose.s7 =  dose.s7/mass;

	dose = dose/nPaths;

	
	error.s0 = error.s0/mass/mass;
	error.s1 = error.s1/volume/volume;
	error.s2 = error.s2/volume/volume;
	
	if(error.s0 > 0){
		error.s3 = error.s3/dose.s0/vox.s2/dose.s0/vox.s2;
		error.s4 = error.s4/dose.s0/vox.s2/dose.s0/vox.s2;
	}

	error.s5 =  error.s5/mass/mass;
	error.s6 =  error.s6/mass/mass;
	error.s7 =  error.s7/mass/mass;

	error = error/nPaths;
	error = sqrt((error - dose*dose)/nPaths);

	doseBuff[absId] = dose;
	errorBuff[absId] = error;

//	if(doseBuff[absId].s0 > 0.0f)
//		printf("buff, %v8f\n", doseBuff[absId]);
}

__kernel void tempStore(__global float8 * doseCounter, __global float8 * doseBuff, 
					__global float8 * errorCounter, __global float8 * errorBuff){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);
	int nVoxels = get_global_size(0)*get_global_size(1)*get_global_size(2); 

//	printf("idx = %d, idy = %d, idz = %d, absId = %d, nVoxels = %d\n", idx, idy, idz, absId, nVoxels);
	
	for(int i  = 0; i < NDOSECOUNTERS; i++){
		doseBuff[absId] += doseCounter[absId + i*nVoxels];
		errorBuff[absId] += errorCounter[absId + i*nVoxels];
		doseCounter[absId + i*nVoxels] = 0;
		errorCounter[absId + i*nVoxels] = 0;
	}
//	if(doseBuff[absId].s0 > 0.0f)
//		printf("buff, %v8f\n", doseBuff[absId]);

}