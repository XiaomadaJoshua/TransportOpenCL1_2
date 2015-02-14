#include "Macro.h"

__kernel void initializeDoseCounter(__global float8 * doseCounter){
	size_t gid = get_global_id(0);
	doseCounter[gid] = (float8)(0, 0, 0, 0, 0, 0, 0, 0);
}


__kernel void finalize(__global float8 * doseBuff, read_only image3d_t voxels, float3 voxSize){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);

	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	float4 vox = read_imagef(voxels, voxSampler, (float4)(idx, idy, idz, 0.0f));
	float volume = voxSize.x*voxSize.y*voxSize.z;
	float mass = vox.s2*volume;
	
	// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is secondary LET, 5 in float8 is primary dose,
	// 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	doseBuff[absId].s1 = doseBuff[absId].s1/volume;
	doseBuff[absId].s2 = doseBuff[absId].s2/volume;
	
	if(doseBuff[absId].s0 > 0){
		doseBuff[absId].s3 = doseBuff[absId].s3/doseBuff[absId].s0/vox.s2;
		doseBuff[absId].s4 = doseBuff[absId].s4/doseBuff[absId].s0/vox.s2;
	}

//	doseCounter[absId].s5 =  doseCounter[absId].s5/mass;
//	doseCounter[absId].s6 =  doseCounter[absId].s6/mass;
//	doseCounter[absId].s7 =  doseCounter[absId].s7/mass;
//	doseCounter[absId].s0 =  doseCounter[absId].s0/mass;

//	printf("mass = %f\n", mass);
}

__kernel void tempStore(__global float8 * doseCounter, __global float8 * doseBuff){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);
	int nVoxels = get_global_size(0)*get_global_size(1)*get_global_size(2);
	for(int i  = 0; i < NDOSECOUNTERS; i++){
		doseBuff[absId] += doseCounter[absId + i*nVoxels];
		doseCounter[absId + i*nVoxels] = 0;
	}
}