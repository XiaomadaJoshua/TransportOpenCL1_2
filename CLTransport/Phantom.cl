#include "Macro.h"

__kernel void initializeDoseCounter(__global float * doseCounter){
	size_t gid = get_global_id(0);
	doseCounter[gid] = 0.0f;
}


__kernel void finalize(__global float * doseBuff, read_only image3d_t voxels, float3 voxSize, uint nPaths){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);

	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	float4 vox = read_imagef(voxels, voxSampler, (float4)(idx, idy, idz, 0.0f));
	float volume = voxSize.x*voxSize.y*voxSize.z;
	float mass = vox.s2*volume;

	doseBuff[absId] =  doseBuff[absId]/mass/nPaths;
//	if(vox.s0 < -800.0f)
//		doseBuff[absId] = 0.0f;

}

__kernel void tempStore(__global float * doseCounter, __global float * doseBuff){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);
	int nVoxels = get_global_size(0)*get_global_size(1)*get_global_size(2);
	for(int i  = 0; i < NDOSECOUNTERS; i++){
		doseBuff[absId] += doseCounter[absId + i*nVoxels];
//		if(doseCounter[absId + i*nVoxels] != 0)
//			printf("%f\t", doseCounter[absId + i*nVoxels]);
		doseCounter[absId + i*nVoxels] = 0;
	}
//	if(doseBuff[absId] != 0)
//		printf("buff\t%f\n", doseBuff[absId]);

}