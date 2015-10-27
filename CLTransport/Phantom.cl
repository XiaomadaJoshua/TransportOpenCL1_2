//#include "Macro.h"

__kernel void initializeDoseCounter(__global float8 * doseCounter){
	size_t gid = get_global_id(0);
	doseCounter[gid] = 0.0f;
}


__kernel void finalize(__global float8 * doseCounter, __global float8 * doseBuff, __global float8 * errorBuff, read_only image3d_t voxels, float3 voxSize, uint nPaths){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t sizeX = get_global_size(0), sizeY = get_global_size(1), sizeZ = get_global_size(2);
	size_t absId = idx + idy*sizeX + idz*sizeX*sizeY;
	size_t nVoxels = sizeX*sizeY*sizeZ;
	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	float4 vox = read_imagef(voxels, voxSampler, (float4)(idx, idy, idz, 0.0f));
	float volume = voxSize.x*voxSize.y*voxSize.z;
	float mass = vox.s2*volume;
	float8 mean = 0.0f, var = 0.0f, std;

	for(int i = 0; i < NDOSECOUNTERS; i++){
		mean += doseCounter[absId + i*nVoxels];
		var += doseCounter[absId + i*nVoxels]*doseCounter[absId + i*nVoxels];
//		if(idx == 25 && idy == 25 && idz == 0)
//			printf("%v8f\n", doseCounter[absId + i*nVoxels]);
	}

	std = sqrt(NDOSECOUNTERS*var - mean*mean);

// 0 in float8 is total dose, 1 in float8 is primary fluence, 2 in float8 is secondary fluence, 3 in float8 is primary LET, 4 in float8 is nothing, 
	//5 in float8 is primary dose, 6 in float8 is secondary dose, 7 in float8 is heavy dose.
	mean.s0 = mean.s0/mass/nPaths;
	mean.s5 = mean.s5/mass/nPaths;
	mean.s6 = mean.s6/mass/nPaths;
	mean.s7 = mean.s7/mass/nPaths;
	mean.s1 = mean.s1/volume/nPaths;
	mean.s2 = mean.s2/volume/nPaths;
	if(mean.s0 > ZERO)
		mean.s3 = mean.s3/vox.s2/mean.s0/nPaths;
	
	
	std.s0 = std.s0/mass/nPaths;
	std.s5 = std.s5/mass/nPaths;
	std.s6 = std.s6/mass/nPaths;
	std.s7 = std.s7/mass/nPaths;
	std.s1 = std.s1/volume/nPaths;
	std.s2 = std.s2/volume/nPaths;
	if(mean.s0 > ZERO)
		std.s3 = std.s3/vox.s2/mean.s0/nPaths;

	doseBuff[absId] = mean;
//	if(idx == 25 && idy == 25 && idz == 0)
//		printf("%f\n", doseBuff[absId].s1);
	errorBuff[absId] = std;
//	if(idx == 25 && idy == 25 && idz == 0)
//		printf("%f\n", errorBuff[absId].s1);

//	if(doseBuff[absId].s0 > 0.0f)
//		printf("buff, %v8f\n", doseBuff[absId]);
}

__kernel void tempStore(__global float8 * doseCounter, __global float8 * batchBuff){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);
	int nVoxels = get_global_size(0)*get_global_size(1)*get_global_size(2); 

//	printf("idx = %d, idy = %d, idz = %d, absId = %d, nVoxels = %d\n", idx, idy, idz, absId, nVoxels);
	
	for(int i  = 0; i < NDOSECOUNTERS; i++){
//		if(idx == 25 && idy == 25 && idz == 0)
//			printf("%f\n", doseCounter[absId + i*nVoxels].s1);
		batchBuff[absId + i*nVoxels] += doseCounter[absId + i*nVoxels];
		doseCounter[absId + i*nVoxels] = 0;

//		if(idx == 25 && idy == 25 && idz == 2)
//			printf("%v8f\n", doseCounter[absId + i*nVoxels]);
	}
//	if(doseBuff[absId].s0 > 0.0f)
//		printf("buff, %v8f\n", doseBuff[absId]);

}