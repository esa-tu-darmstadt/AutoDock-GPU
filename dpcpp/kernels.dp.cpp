/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.
Copyright (C) 2022 Intel Corporation

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cassert>
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"
#include "dpcpp_migration.h"

inline uint64_t llitoulli(int64_t l)
{
	uint64_t u;
	/*
	DPCT1053:0: Migration of device assembly code is not supported.
	*/
	// asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
	u = l;
	return u;
}

inline int64_t ullitolli(uint64_t u)
{
	int64_t l;
	/*
	DPCT1053:1: Migration of device assembly code is not supported.
	*/
	// asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
	l = u;
	return l;
}

#define ATOMICADDI32(pAccumulator, value) \
	sycl::atomic_ref<int, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) += ((int)(value))

#define ATOMICSUBI32(pAccumulator, value) \
	sycl::atomic_ref<int, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) -= ((int)(value))

#define ATOMICADDF32(pAccumulator, value) \
	sycl::atomic_ref<float, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) += ((float)(value))

#define ATOMICSUBF32(pAccumulator, value) \
	sycl::atomic_ref<float, SYCL_ATOMICS_MEMORY_ORDER, SYCL_ATOMICS_MEM_SCOPE, sycl::access::address_space::local_space>(*pAccumulator) -= ((float)(value))

#ifdef USE_XMX
/* Reduction using matrix units */

// Implementation based on M.Sc. thesis by Gabin Schieffer at KTH:
// "Accelerating a Molecular Docking Application by Leveraging Modern Heterogeneous Computing Systemx"
// https://www.diva-portal.org/smash/get/diva2:1786161/FULLTEXT01.pdf

// We consider that a CUDA fragment is equivalent to a SYCL submatrix
//
// Compilation: make DEVICE=XeGPU PLATFORM=NvGPU XMX=ON TESTLS=ad NUMWI=64 test

// If enabled, then using hardcoded inputs
//#define DEBUG_XMX_INPUTS
//#define DEBUG_XMX_INPUTS_INDEX_MAP
#define XMX_EC
#define XMX_EC_DEBUG

// Number of rows/cols of a submatrix: tM, tN, tK
constexpr int tM = 8;
constexpr int tN = 16;
constexpr int tK = 8;

using tf32 = sycl::ext::oneapi::experimental::matrix::precision::tf32;
using TA = tf32;
using TB = tf32;
using TC = float;

// Number of elements of input matrix (to be reduced)
constexpr int TILE_NELEMS = tM * tK;

using namespace sycl::ext::oneapi::experimental::matrix;

// Printing submatrices contents,
// which have to be previously copied into an array in local memory.
// Enclosing the implementation of print_submatrix_sg()
// within barriers (as initially thought) produces wrong results.
// Such mistake makes sense since print_submatrix_sg() is called if(wi_Id_Wg <= 31).
// Extra sync before printing is not needed as long as
// print_submatrix_sg() is called after joint_matrix functions,
// which are executed by the entire sub_group (i.e., wi_Id_Wg <= 31)

// Printing within sub-group
template <typename T, uint NROWS, uint NCOLS, enum layout LAYOUT>
void print_submatrix_sg (
	sycl::nd_item<3> item,
	const char *msg,
	T *data_to_print
) {
	int wg_Id_ND = item.get_group(2);

	sycl::sub_group sg = item.get_sub_group();
	int wi_Id_sg = sg.get_local_id();

	// Only a single work-item within a sub-group prints
	if (wg_Id_ND == 0 && wi_Id_sg == 0) {
		sycl::ext::oneapi::experimental::printf("\n%s", msg);
		for (uint i = 0; i < NROWS; i++) {
			sycl::ext::oneapi::experimental::printf("\n[Row %2u]: ", i);
			for (uint j = 0; j < NCOLS; j++) {
				if (LAYOUT == layout::row_major) {
					sycl::ext::oneapi::experimental::printf(" %5.3f ", float(data_to_print[i*NCOLS+j]));
				}
				else if (LAYOUT == layout::col_major) {
					sycl::ext::oneapi::experimental::printf(" %5.3f ", float(data_to_print[j*NROWS+i]));
				}
			}
		}
		sycl::ext::oneapi::experimental::printf("\n");
	}
}

// Printing within work-group
template <typename T, uint NROWS, uint NCOLS, enum layout LAYOUT>
void print_submatrix_WG (
	sycl::nd_item<3> item,
	const char *msg,
	T *data_to_print
) {
	int wi_Id_Wg = item.get_local_id(2);
	int wg_Id_ND = item.get_group(2);

	// Only a single work-item within a work-group prints
	if (wg_Id_ND == 0 && wi_Id_Wg == 0) {
		sycl::ext::oneapi::experimental::printf("\n%s", msg);
		for (uint i = 0; i < NROWS; i++) {
			sycl::ext::oneapi::experimental::printf("\n[Row %2u]: ", i);
			for (uint j = 0; j < NCOLS; j++) {
				if (LAYOUT == layout::row_major) {
					sycl::ext::oneapi::experimental::printf(" %5.3f ", float(data_to_print[i*NCOLS+j]));
				}
				else if (LAYOUT == layout::col_major) {
					sycl::ext::oneapi::experimental::printf(" %5.3f ", float(data_to_print[j*NROWS+i]));
				}
			}
		}
		sycl::ext::oneapi::experimental::printf("\n");
    }
}

void print_wi_indexes (
	sycl::nd_item<3> item
) {
	// Identifying global, local, and work-group ids
	int wi_Id_ND = item.get_global_id(2); // Returns the wi's position in the NDRange (in dimension 2)
	int wi_Id_Wg = item.get_local_id(2); // Returns the wi's position within the current wg (in dimension 2)
	int wg_Id_ND = item.get_group(2); // Returns the wg's position within the overal NDRange (in dimension 2)
	int wg_Size = item.get_local_range(2); // Returns the number of wis per wg (in dimension 2)

	// Identifying sub-groups
	sycl::sub_group sg = item.get_sub_group();
	int sg_Range = sg.get_group_range().get(0); // Returns the number of subgroups within the wg
	int sg_Id_Wg = sg.get_group_id().get(0); // Returns the index of the subgroup within the wg
	int sg_Size = sg.get_local_range().get(0); // Returns the number of wis per subgroup
	int wi_Id_sg = sg.get_local_id(); // Returns the index of the work-item within its subgroup

	sycl::ext::oneapi::experimental::printf(
		"wi_Id_ND: %i, \twi_Id_Wg: %i, \twg_Id_ND: %i,\twg_Size: %i, \tsg_Range: %i, \tsg_Id_Wg: %i, \tsg_Size: %i, \twi_Id_sg: %i\n",
		wi_Id_ND, wi_Id_Wg, wg_Id_ND, wg_Size, sg_Range, sg_Id_Wg, sg_Size, wi_Id_sg);
}

// Q_data points to an array to be loaded to sub_Q.
// sub_Q is a submatrix configured as
// "use::a" (1st matrix mult operand) and "tM x tK" (shape).
// Hence, Q_data holds the data of the sub_Q submatrix
void fill_Q (
	sycl::nd_item<3> item,
	float *Q_data
) {
	sycl::sub_group sg = item.get_sub_group();
	int wi_Id_sg = sg.get_local_id();
	int sg_Size = sg.get_local_range().get(0);

	// Slightly improved multi-threaded implementation.
	// IMPORTANT: this is computed by a sub-group,
	// and thus, the stride MUST be "sg_Size" instead of "wg_Size"
	for (uint i = wi_Id_sg; i < tM/4; i+=sg_Size) {	// Row counter: how many rows (of 4x4 blocks) are there in the matrix?
		for (uint j = 0; j < tK/4; j++) {	// Col counter: how many cols (of 4x4 blocks) are there in the matrix?
			for (uint ii = 0; ii < 4; ii++) {
				for (uint jj = 0; jj < 4; jj++) {
					Q_data[4 * (tM*i + j) + tM*ii + jj] = (ii == jj)? 1.0f: 0.0f; // Row-major
					//Q_data[4 * (tK*j + i) + tK*jj + ii] = (ii == jj)? 1.0f: 0.0f; // Col-major
				}
			}
		}
	}

	/*
	print_submatrix_sg<float, tM, tK, layout::row_major>(item, "Q_data [inside fill_Q()]", Q_data);
	*/
}

// Reordering arrays for correctly reducing input data.
// Reason:
// For PVC GPUs, the chosen "tM x tN x tK" (i.e., 8 x 16 x 8) matrix configuration
// only works for some layouts (but not for both row- and col-major)
void map_input_array (
	sycl::nd_item<3> item,
	float *data_to_be_reduced,
	float *data_to_be_reduced_arranged
	#ifdef DEBUG_XMX_INPUTS_INDEX_MAP
	,
	uint *in_indexes,
	uint *out_indexes
	#endif
) {
	int wi_Id_Wg = item.get_local_id(2);
	int wg_Size = item.get_local_range(2);

	item.barrier(SYCL_MEMORY_SPACE);

	for (uint i = wi_Id_Wg; i < (4 * NUM_OF_THREADS_PER_BLOCK); i+=wg_Size) {
		uint j = 24*(i/32) + (i/4) + 8*(i%4);

		// Storing values of initial and final indexes
		#ifdef DEBUG_XMX_INPUTS_INDEX_MAP
		in_indexes[i] = i;
		out_indexes[i] = j;
		#endif

		data_to_be_reduced_arranged[j] = data_to_be_reduced[i];
		//sycl::ext::oneapi::experimental::printf("i = %i, j = %i\n", i, j);
	}

	item.barrier(SYCL_MEMORY_SPACE);

	// Comparing initial and final indexes.
	// These help us verifying the correct index mapping.
	// Only a single work-item within a work-group prints
	#ifdef DEBUG_XMX_INPUTS_INDEX_MAP
	int wg_Id_ND = item.get_group(2);
	if (wg_Id_ND == 0 && wi_Id_Wg == 0) {
		sycl::ext::oneapi::experimental::printf("\n\nInitial indexes (data_to_be_reduced)");
		for (uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK); i++) {
			if(i % 16 == 0) {
				sycl::ext::oneapi::experimental::printf("\n");
			}
			sycl::ext::oneapi::experimental::printf("\t%3i", in_indexes[i]);
		}

		sycl::ext::oneapi::experimental::printf("\n\nFinal indexes (data_to_be_reduced_arranged)");
		for (uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK); i++) {
			if(i % 16 == 0) {
				sycl::ext::oneapi::experimental::printf("\n");
			}
			sycl::ext::oneapi::experimental::printf("\t%3i", out_indexes[i]);
		}
	}
	#endif
}

void print_reduced_values (
	sycl::nd_item<3> item,
	const char *msg,
	float *data_to_be_reduced_arranged
){
	int wi_Id_Wg = item.get_local_id(2);
	int wg_Id_ND = item.get_group(2);

	if (wg_Id_ND == 0 && wi_Id_Wg == 0) {
		sycl::ext::oneapi::experimental::printf("\n%s: \t%5.3f \t%5.3f \t%5.3f \t%5.3f\n", msg,
			data_to_be_reduced_arranged[0], data_to_be_reduced_arranged[1], data_to_be_reduced_arranged[2], data_to_be_reduced_arranged[3]);
	}
}

// Col_major for T_JM_A: not supported for current matrix shape (8 x 16 x 8) and data type (tf32)
using T_JM_A = joint_matrix<sycl::sub_group, TA, use::a, tM, tK, layout::row_major>;
using T_JM_B = joint_matrix<sycl::sub_group, TB, use::b, tK, tN, layout::col_major>;
using T_JM_C = joint_matrix<sycl::sub_group, TC, use::accumulator, tM, tN>;


#ifdef XMX_EC
// Implementation based on paper by Ootomo et al.:
// "Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance"
// https://doi.org/10.1177/10943420221090256

// The probabilities of underflow and gradual underflow
// in subtracting two values are high when their absolute values are close.
// For reducing these probabilities,
// Ootomo et al. introduce a factor (here as FACTOR_RED_UF)
// based on the mantissa length of the matrix operand.
// The mantissa length depends on the data format.
// Notice that for both FP16 and TF32, the mantissa length is 10
constexpr int length_mantissa_tf32 = 10;
constexpr int FACTOR_RED_UF = 1 << (length_mantissa_tf32 + 1); // 2 ^ 11 = 2048

void custom_matrix_mad_ec (
	sycl::nd_item<3> item,
	float *A_fp32,
	float *B_fp32,
	T_JM_C &sub_C,
	// Arrays for intermediate storage
	float *A_tf32,
	float *B_tf32,
	float *dA_tf32,
	float *dB_tf32
) {
	sycl::sub_group sg = item.get_sub_group();
	int wi_Id_sg = sg.get_local_id();
	int sg_Size = sg.get_local_range().get(0);

	T_JM_A sub_A;
	T_JM_B sub_B;

	T_JM_A sub_dA;
	T_JM_B sub_dB;
	T_JM_C sub_dC;

	// Initializing accumulator submatrices
	//joint_matrix_fill(sg, sub_C, 0.0f); // MUST be commented out as it is initialized outside
	joint_matrix_fill(sg, sub_dC, 0.0f);

	// Computing [19] (Ootomo et al.)
	for (uint i = wi_Id_sg; i < tM * tK; i+=sg_Size) { A_tf32[i] = round_to_tf32(A_fp32[i]); }
	joint_matrix_load(sg, sub_A, sycl::local_ptr<float>(A_tf32), tK); // Row-major -> stride is tK

	// Computing [21] (Ootomo et al.)
	for (uint i = wi_Id_sg; i < tK * tN; i+=sg_Size) { B_tf32[i] = round_to_tf32(B_fp32[i]); }

	// TODO: it seems that loading directly into matrix B is not supported.
	// Thus, we load instead into a matrix C, which has the same sizes (by coincidence),
	// and then, we copy from the matrix C to the matrix B.
	// Remember that when loading into matrix C, the matrix layout must be explicitly declared
	//joint_matrix_load(sg, sub_B, sycl::local_ptr<float>(B_tf32), tK); // Col-major -> stride is tK
	T_JM_C sub_C_tmp;
	joint_matrix_load(sg, sub_C_tmp, sycl::local_ptr<float>(B_tf32), tK, layout::col_major); // Col-major -> stride is tK
	// Replacemented suggested by Intel
	//joint_matrix_copy(sg, sub_C_tmp, sub_B);
	auto wi_data_C = sycl::ext::oneapi::detail::get_wi_data(sg, sub_C_tmp);
	auto wi_data_B = sycl::ext::oneapi::detail::get_wi_data(sg, sub_B);
	for (int i = 0; i < /*wi_data_C.length()*/4; i++) { wi_data_B[i] = (float)(wi_data_C[i]); }

	// Computing [20] (Ootomo et al.)
	for (uint i = wi_Id_sg; i < tM * tK; i+=sg_Size) { dA_tf32[i] = round_to_tf32( (A_fp32[i] - (float)(A_tf32[i])) * FACTOR_RED_UF ); }
		#ifdef XMX_EC_DEBUG
		print_submatrix_sg<float, tM, tK, layout::row_major>(item, "A_fp32", A_fp32);
		print_submatrix_sg<float, tM, tK, layout::row_major>(item, "A_tf32", A_tf32);
		print_submatrix_sg<float, tM, tK, layout::row_major>(item, "dA_tf32", dA_tf32);
		#endif
	joint_matrix_load(sg, sub_dA, sycl::local_ptr<float>(dA_tf32), tK); // Row-major -> stride is tK

	// Computing [22] (Ootomo et al.)
	for (uint i = wi_Id_sg; i < tK * tN; i+=sg_Size) { dB_tf32[i] = round_to_tf32( (B_fp32[i] - (float)(B_tf32[i])) * FACTOR_RED_UF ); }
		#ifdef XMX_EC_DEBUG
		print_submatrix_sg<float, tK, tN, layout::col_major>(item, "B_fp32", B_fp32);
		print_submatrix_sg<float, tK, tN, layout::col_major>(item, "B_tf32", B_tf32);
		print_submatrix_sg<float, tK, tN, layout::col_major>(item, "dB_tf32", dB_tf32);
		#endif

	// TODO: it seems that loading directly into matrix B is not supported.
	// Thus, we load instead into a matrix C, which has the same sizes (by coincidence),
	// and then, we copy from the matrix C to the matrix B.
	// Remember that when loading into matrix C, the matrix layout must be explicitly declared
	//joint_matrix_load(sg, sub_dB, sycl::local_ptr<float>(dB_tf32), tK); // Col-major -> stride is tK
	T_JM_C sub_dC_tmp;
	joint_matrix_load(sg, sub_dC_tmp, sycl::local_ptr<float>(dB_tf32), tK, layout::col_major); // Col-major -> stride is tK
	// Replacemented suggested by Intel
	//joint_matrix_copy(sg, sub_dC_tmp, sub_dB);
	auto wi_data_dC = sycl::ext::oneapi::detail::get_wi_data(sg, sub_dC_tmp);
	auto wi_data_dB = sycl::ext::oneapi::detail::get_wi_data(sg, sub_dB);
	for (int i = 0; i < /*wi_data_C.length()*/4; i++) { wi_data_dB[i] = (float)(wi_data_dC[i]); }

	// Computing part of [24] (Ootomo et al.)
	joint_matrix_mad(sg, sub_dC, sub_dA, sub_B, sub_dC);
	joint_matrix_mad(sg, sub_dC, sub_A, sub_dB, sub_dC);

	// Initializing temporal accumulator
	T_JM_C sub_tmp;
	joint_matrix_fill(sg, sub_tmp, 0.0f);
	joint_matrix_mad(sg, sub_tmp, sub_A, sub_B, sub_tmp);

	// Accumulation using FP32 SIMT cores
/*
	//  Doesn't compile
	joint_matrix_apply(sg, sub_tmp, sub_C, [=](const TC &x, TC &y) {
		y = y + x;
	});
*/

	// Compiles, but produces incorrect results
	joint_matrix_apply(sg, sub_C, sub_tmp, [=](TC &y, const TC &x) {
		y = y + x;
	});

	// Computing part of [24] (Ootomo et al.)
/*
	// Doesn't compile
	joint_matrix_apply(sg, sub_dC, sub_C, [=](const TC &x, TC &y) {
		y = y + (x / FACTOR_RED_UF);
	});
*/
	// Compiles , but produces incorrect results
	joint_matrix_apply(sg, sub_C, sub_dC, [=](TC &y, const TC &x) {
		y = y + (x / FACTOR_RED_UF);
	});

}
#endif

void reduce_via_matrix_units (
	sycl::nd_item<3> item,
	float *data_to_be_reduced,
	float *Q_data
	#ifdef XMX_EC
	,
	float *in_A,
	float *in_B,
	float *in_A_tf32,
	float *in_B_tf32,
	float *in_dA_tf32,
	float *in_dB_tf32
		#ifdef XMX_EC_DEBUG
		,
		float *debug_B
		#endif
	#endif
) {
	sycl::sub_group sg = item.get_sub_group();
	int sg_Id_Wg = sg.get_group_id().get(0);

	int sg_Size = sg.get_local_range().get(0);
	int wi_Id_sg = sg.get_local_id();

	item.barrier(SYCL_MEMORY_SPACE);

	/*
	print_wi_indexes(item);
	*/

	// Only a single sub-group per work-group performs reduction
	if (sg_Id_Wg == 0) {
		// Declaring and filling submatrices
		#ifdef XMX_EC
		for (uint i = wi_Id_sg; i < tK * tN; i+=sg_Size) { in_B [i] = 1.0f; } // Instead of sub_P filled with 1.0f
		#else
		T_JM_B sub_P;
		joint_matrix_fill(sg, sub_P, 1.0f); // P: only ones
		#endif

		T_JM_C sub_V;
		joint_matrix_fill(sg, sub_V, 0.0f); // Output: initialize to zeros

		// 1. Accumulate the values: V <- AP + V
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK)/(TILE_NELEMS);  i++) {
			const uint offset = i * TILE_NELEMS; // Moving to next input block

			/*
			int wg_Id_ND = item.get_group(2);
			int wi_Id_sg = sg.get_local_id();
			if (wg_Id_ND == 0 && wi_Id_sg == 0) {
				sycl::ext::oneapi::experimental::printf("\nLoop: tripcount = %d | iteration = %d | offset = %d", (4 * NUM_OF_THREADS_PER_BLOCK) / TILE_NELEMS, i, offset);
			}
			*/

			#ifdef XMX_EC
			in_A = (data_to_be_reduced + offset);
			custom_matrix_mad_ec(item, in_A, in_B, sub_V, in_A_tf32, in_B_tf32, in_dA_tf32, in_dB_tf32);
				#ifdef XMX_EC_DEBUG
				joint_matrix_store(sg, sub_V, sycl::local_ptr<float>(debug_B), tM, layout::col_major);
				print_submatrix_sg<float, tK, tN, layout::col_major>(item, "sub_V (partial)", debug_B);
				#endif
			#else
			T_JM_A sub_A;
			joint_matrix_load(sg, sub_A, sycl::local_ptr<float>(data_to_be_reduced + offset), tK); // Row-major -> stride is tK
			joint_matrix_mad(sg, sub_V, sub_A, sub_P, sub_V);
			#endif
		}

		T_JM_C sub_C;
		joint_matrix_fill(sg, sub_C, 0.0f); // Final result

		fill_Q(item, Q_data);

		// 2. Perform line sum: C <- QW + C (zero)
		#ifdef XMX_EC
		in_A = Q_data;
		joint_matrix_store(sg, sub_V, sycl::local_ptr<float>(in_B), tM, layout::col_major);
		custom_matrix_mad_ec(item, in_A, in_B, sub_C, in_A_tf32, in_B_tf32, in_dA_tf32, in_dB_tf32);
				#ifdef XMX_EC_DEBUG
				joint_matrix_store(sg, sub_C, sycl::local_ptr<float>(debug_B), tM, layout::col_major);
				print_submatrix_sg<float, tK, tN, layout::col_major>(item, "sub_C", debug_B);
				#endif
		#else
		T_JM_A sub_Q;
		joint_matrix_load(sg, sub_Q, sycl::local_ptr<float>(Q_data), tK);	// Row-major -> stride is tK

		// W <- V (required since V must be transformed to "use::b")
		T_JM_B sub_W;
		joint_matrix_copy(sg, sub_V, sub_W);

		joint_matrix_mad(sg, sub_C, sub_Q, sub_W, sub_C);
		#endif

		// 3. Store result in shared memory
		joint_matrix_store(sg, sub_C, sycl::local_ptr<float>(data_to_be_reduced), tM, layout::col_major);
	}

	item.barrier(SYCL_MEMORY_SPACE);
}

/* Reduction using matrix units */
#endif

static dpct::constant_memory<GpuData, 0> cData;
static GpuData cpuData;

void SetKernelsGpuData(GpuData *pData) try
{
	int status;
	status = (dpct::get_default_queue().memcpy(cData.get_ptr(), pData, sizeof(GpuData)).wait(), 0);
	RTERROR(status, "SetKernelsGpuData copy to cData failed");
	memcpy(&cpuData, pData, sizeof(GpuData));
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
			  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

void GetKernelsGpuData(GpuData *pData) try
{
	int status;
	status = (dpct::get_default_queue().memcpy(pData, cData.get_ptr(), sizeof(GpuData)).wait(), 0);
	RTERROR(status, "GetKernelsGpuData copy From cData failed");
}
catch (sycl::exception const &exc)
{
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
			  << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}

// Kernel files
#include "calcenergy.dp.cpp"
#include "calcMergeEneGra.dp.cpp"
#include "auxiliary_genetic.dp.cpp"
#include "kernel1.dp.cpp"
#include "kernel2.dp.cpp"
#include "kernel3.dp.cpp"
#include "kernel4.dp.cpp"
#include "kernel_ad.dp.cpp"
#include "kernel_adam.dp.cpp"
