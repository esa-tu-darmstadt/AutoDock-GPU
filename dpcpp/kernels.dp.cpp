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

namespace syclexp = sycl::ext::oneapi::experimental;

#define syclprintf syclexp::printf

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

// If enabled, then using global memory instead of SLM for holding data-to-be-reduced
#define USE_GLOB_SPACE_XMX_INPUTS

// Number of rows/cols of a submatrix: tM, tN, tK
constexpr int tM = 16;
constexpr int tN = 16;
constexpr int tK = 16;

using bf16 = sycl::ext::oneapi::bfloat16;
using TA = bf16;
using TB = bf16;
using TC = float;

// Number of elements of input matrix (to be reduced)
constexpr int TILE_NELEMS = tM * tK;

using namespace syclexp::matrix;

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
		syclprintf("\n%s", msg);
		for (uint i = 0; i < NROWS; i++) {
			syclprintf("\n[Row %2u]: ", i);
			for (uint j = 0; j < NCOLS; j++) {
				if (LAYOUT == layout::row_major) {
					syclprintf(" %5.3f ", float(data_to_print[i*NCOLS+j]));
				}
				else if (LAYOUT == layout::col_major) {
					syclprintf(" %5.3f ", float(data_to_print[j*NROWS+i]));
				}
			}
		}
		syclprintf("\n");
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
		syclprintf("\n%s", msg);
		for (uint i = 0; i < NROWS; i++) {
			syclprintf("\n[Row %2u]: ", i);
			for (uint j = 0; j < NCOLS; j++) {
				if (LAYOUT == layout::row_major) {
					syclprintf(" %5.3f ", float(data_to_print[i*NCOLS+j]));
				}
				else if (LAYOUT == layout::col_major) {
					syclprintf(" %5.3f ", float(data_to_print[j*NROWS+i]));
				}
			}
		}
		syclprintf("\n");
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

	syclprintf(
		"wi_Id_ND: %i, \twi_Id_Wg: %i, \twg_Id_ND: %i,\twg_Size: %i, \tsg_Range: %i, \tsg_Id_Wg: %i, \tsg_Size: %i, \twi_Id_sg: %i\n",
		wi_Id_ND, wi_Id_Wg, wg_Id_ND, wg_Size, sg_Range, sg_Id_Wg, sg_Size, wi_Id_sg);
}

// Q_data points to an array to be loaded to sub_Q.
// sub_Q is a submatrix configured as
// "use::a" (1st matrix mult operand) and "tM x tK" (shape).
// Hence, Q_data holds the data of the sub_Q submatrix
void fill_Q (
	sycl::nd_item<3> item,
	TA *Q_data
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
					//Q_data[4 * (tM*i + j) + 16*ii + jj] = (ii == jj)? 1.0f: 0.0f; // Row-major
					Q_data[4 * (tK*j + i) + 16*jj + ii] = (ii == jj)? 1.0f: 0.0f; // Col-major
				}
			}
		}
	}

	/*
	print_submatrix_sg<TA, tM, tK, layout::col_major>(item, "Q_data [inside fill_Q()]", Q_data);
	*/
}

template <typename T>
void print_reduced_values (
	sycl::nd_item<3> item,
	const char *msg,
	T *data_to_be_reduced_arranged
){
	int wi_Id_Wg = item.get_local_id(2);
	int wg_Id_ND = item.get_group(2);

	if (wg_Id_ND == 0 && wi_Id_Wg == 0) {
		syclprintf("\n%s: \t%5.3f \t%5.3f \t%5.3f \t%5.3f\n", msg,
			float(data_to_be_reduced_arranged[0]), float(data_to_be_reduced_arranged[1]), float(data_to_be_reduced_arranged[2]), float(data_to_be_reduced_arranged[3]));
	}
}

// Col_major for T_JM_A: is supported in the RTX3050Ti for current matrix shape (16 x 16 x 16) and data type (bfloat16)
using T_JM_A = joint_matrix<sycl::sub_group, TA, use::a, tM, tK, layout::col_major>;
using T_JM_B = joint_matrix<sycl::sub_group, TB, use::b, tK, tN, layout::col_major>;
using T_JM_C = joint_matrix<sycl::sub_group, TC, use::accumulator, tM, tN>;

void reduce_via_matrix_units (
	sycl::nd_item<3> item,
	#ifdef USE_GLOB_SPACE_XMX_INPUTS
	bf16 *data_to_be_reduced_global,
	#else
	bf16 *data_to_be_reduced,
	#endif
	bf16 *Q_data,
	float *tmp
) {
	sycl::sub_group sg = item.get_sub_group();
	int sg_Id_Wg = sg.get_group_id().get(0);

	#ifdef USE_GLOB_SPACE_XMX_INPUTS
	// Defining multi-ptr from raw pointer
	auto data_to_be_reduced_mptr = sycl::multi_ptr<bf16, sycl::access::address_space::global_space>(data_to_be_reduced_global);
	#endif

	item.barrier(SYCL_MEMORY_SPACE);

	/*
	print_wi_indexes(item);
	*/

	// Only a single sub-group per work-group performs reduction
	if (sg_Id_Wg == 0) {
		// Declaring and filling submatrices
		T_JM_B sub_P;
		joint_matrix_fill(sg, sub_P, 1.0f); // P: only ones

		T_JM_C sub_V;
		joint_matrix_fill(sg, sub_V, 0.0f); // Output: initialize to zeros

		// 1. Accumulate the values: V <- AP + V
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK)/(TILE_NELEMS);  i++) {
			const uint offset = i * TILE_NELEMS; // Moving to next input block

			/*
			int wg_Id_ND = item.get_group(2);
			int wi_Id_sg = sg.get_local_id();
			if (wg_Id_ND == 0 && wi_Id_sg == 0) {
				syclprintf("\nLoop: tripcount = %d | iteration = %d | offset = %d", (4 * NUM_OF_THREADS_PER_BLOCK) / TILE_NELEMS, i, offset);
			}
			*/

			T_JM_A sub_A;
			joint_matrix_load(
				sg,
				sub_A,
				#ifdef USE_GLOB_SPACE_XMX_INPUTS
				(data_to_be_reduced_mptr + offset),
				#else
				sycl::local_ptr<TA>(data_to_be_reduced + offset),
				#endif
				tM); // Col-major -> stride is tM

			joint_matrix_mad(sg, sub_V, sub_A, sub_P, sub_V);
		}

		// W <- V (required since V must be transformed to "use::b")
		T_JM_B sub_W;
		joint_matrix_copy(sg, sub_V, sub_W);

		T_JM_C sub_C;
		joint_matrix_fill(sg, sub_C, 0.0f); // Final result

		T_JM_A sub_Q;
		fill_Q(item, Q_data);
		joint_matrix_load(sg, sub_Q, sycl::local_ptr<TA>(Q_data), tM);	// Col-major -> stride is tM

		// 2. Perform line sum: C <- QW + C (zero)
		joint_matrix_mad(sg, sub_C, sub_Q, sub_W, sub_C);

		// 3. Store result in shared memory
		joint_matrix_store(sg, sub_C, sycl::local_ptr<TC>(tmp), tM, layout::row_major);
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
	sycl::queue queue;
	status = (queue.memcpy(cData.get_ptr(), pData, sizeof(GpuData)).wait(), 0);
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
	sycl::queue queue;
	status = (queue.memcpy(pData, cData.get_ptr(), sizeof(GpuData)).wait(), 0);
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
