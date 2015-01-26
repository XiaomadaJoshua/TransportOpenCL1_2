#include "randomKernel.h"

__kernel void test(read_only image1d_t data, __global float4 * value, __global float * cord){
	size_t gid = get_global_id(0);
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	value[gid] = read_imagef(data, sampler, cord[gid]);
}

__kernel void test2D(read_only image2d_t data, __global float4 * value, __global float2 * cord){
	size_t gid = get_global_id(0);
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	value[gid] = read_imagef(data, sampler, cord[gid]);
}

__kernel void testPhantom(read_only image3d_t vox, __global float * dose, float4 cord, __global float16 * value){
	size_t gid = get_global_id(0);
	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	value[gid].s0123 = read_imagef(vox, sampler, cord);
	value[gid].s456789ab = dose[101];
}



__kernel void testPi(unsigned n, __global uint2 * hitsp){
	size_t gid = get_global_id(0);
	unsigned hits = 0, tries = 0;

	float x[2];
	while(tries < n){
		getUniform(gid, x);
		if((x[0]*x[0] + x[1]*x[1]) < 1.0f)
			hits++;
		tries++;
	}

	hitsp[gid].s0 = hits;
	hitsp[gid].s1 = tries;
}

