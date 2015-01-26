__kernel void initializeDoseCounter(__global float * doseCounter){
	size_t gid = get_global_id(0);
	doseCounter[gid] = 0.0f;
}


__kernel void finalize(__global float * doseCounter, read_only image3d_t voxels, float3 voxSize){
	size_t idx = get_global_id(0);
	size_t idy = get_global_id(1);
	size_t idz = get_global_id(2);
	size_t absId = idx + idy*get_global_size(0) + idz*get_global_size(0)*get_global_size(1);

	sampler_t voxSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	float4 vox = read_imagef(voxels, voxSampler, (float4)(idx, idy, idz, 0.0f));
	float mass = vox.s2*voxSize.x*voxSize.y*voxSize.z;
	
	doseCounter[absId] =  doseCounter[absId]/mass;
}