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

// Number of rows/cols of a submatrix: tM, tN, tK
constexpr int tM = 16;
constexpr int tN = 16;
constexpr int tK = 16;

using TA = sycl::half;
using TB = sycl::half;
using TC = sycl::half;

// Number of elements of input matrix (to be reduced)
constexpr int TILE_NELEMS = tM * tN;

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
		for (uint i = 0; i < NROWS; i++) { // Row counter
			sycl::ext::oneapi::experimental::printf("\n[Row %2u]: ", i);
			for (uint j = 0; j < NCOLS; j++) { // Col counter
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
	// Identifying global, local, and group ids
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

// Q_data points to an array to be loaded to sub_Q
// sub_Q is submatrix with "use::a" use
// Hence, Q_data holds the data of a submatrix with "tM x tK" shape
void fill_Q (
	sycl::nd_item<3> item,
	TA *Q_data
) {
	sycl::sub_group sg = item.get_sub_group();
	int wi_Id_sg = sg.get_local_id();
	int sg_Size = sg.get_local_range().get(0);

	// Slightly improved multi-threaded implementation
	// IMPORTANT: this is computed by a sub-group,
	// and thus, MUST use "sg_Size" instead of "wg_Size"
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

using T_JM_A = joint_matrix<sycl::sub_group, TA, use::a, tM, tK, layout::col_major>;
using T_JM_B = joint_matrix<sycl::sub_group, TB, use::b, tK, tN, layout::col_major>;
using T_JM_ACC = joint_matrix<sycl::sub_group, TC, use::accumulator, tM, tN>;

// Implementation based on MSc thesis at KTH:
// "Accelerating a Molecular Docking Application by Leveraging Modern Heterogeneous Computing Systemx"
// https://www.diva-portal.org/smash/get/diva2:1786161/FULLTEXT01.pdf
//
// We consider that a CUDA fragment is equivalent to a SYCL submatrix
//
// Compilation: make DEVICE=XeGPU PLATFORM=NvGPU XMX=ON TESTLS=ad NUMWI=64 test
void reduce_via_matrix_units (
	sycl::nd_item<3> item,
	sycl::half *data_to_be_reduced,
	sycl::half *Q_data,
	sycl::half *tmp
) {
	int wg_Id_ND = item.get_group(2);

	sycl::sub_group sg = item.get_sub_group();
	int wi_Id_sg = sg.get_local_id();
	int sg_Id_Wg = sg.get_group_id().get(0);

	item.barrier(SYCL_MEMORY_SPACE);

	/*
	print_wi_indexes(item);
	*/

	// Only one sub-group performs reduction
	if (sg_Id_Wg == 0) {
		fill_Q(item, Q_data);

		// Declaring and filling submatrices
		T_JM_B sub_P;
		T_JM_ACC sub_V;
		T_JM_A sub_Q;
		T_JM_B sub_W;
		T_JM_ACC sub_C;
		joint_matrix_fill(sg, sub_P, 1.0f); // P: only ones
		joint_matrix_fill(sg, sub_V, 0.0f); // Output: initialize to zeros
		joint_matrix_fill(sg, sub_C, 0.0f); // Final result
		joint_matrix_load(sg, sub_Q, sycl::local_ptr<TA>(Q_data), tM);	// Load use::a -> stride is tM

		// 1. Accumulate the values: V <- AP + V
		for(uint i = 0; i < (4 * NUM_OF_THREADS_PER_BLOCK) / TILE_NELEMS;  i++) {
			const uint offset = i * TILE_NELEMS;

			/*
			if (wg_Id_ND == 0 && wi_Id_sg == 0) {
				sycl::ext::oneapi::experimental::printf("\nLoop: tripcount = %d | iteration = %d | offset = %d", (4 * NUM_OF_THREADS_PER_BLOCK) / TILE_NELEMS, i, offset);
			}
			*/

			T_JM_A sub_A;
			joint_matrix_load(sg, sub_A, sycl::local_ptr<TA>(data_to_be_reduced + offset), tM); // Load use::a -> stride is tM
			joint_matrix_mad(sg, sub_V, sub_A, sub_P, sub_V);
		}

		// W <- V (required since we need V as a "use::b")
		joint_matrix_store(sg, sub_V, sycl::local_ptr<TC>(tmp), tM, layout::col_major);
		joint_matrix_load(sg, sub_W, sycl::local_ptr<TC>(tmp), tK); // Load use::b -> stride is tK

		// 2. Perform line sum: C <- QW + C (zero)
		joint_matrix_mad(sg, sub_C, sub_Q, sub_W, sub_C);

		// 3. Store result in shared memory
		joint_matrix_store(sg, sub_C, sycl::local_ptr<TA>(data_to_be_reduced), tM, layout::col_major);
	}

	item.barrier(SYCL_MEMORY_SPACE);
}

// Validating inputs
using namespace sycl::ext::oneapi::experimental;

// Compile-Time Query of matrix sizes and types configurations

// References in oneAPI 2024.2.1
//
// ---------------------------------------------
// Intel
// Architectures recognized in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/experimental/device_architecture.hpp
// intel_gpu_bdw // Intel(R) microarchitecture code name Broadwell
// intel_gpu_skl // Intel(R) microarchitecture code name Skylake
// intel_gpu_kbl // Kaby Lake
// intel_gpu_cfl // Coffee Lake
// intel_gpu_apl // Apollo Lake
// intel_gpu_bxt = intel_gpu_apl // Broxton
// intel_gpu_glk // Gemini Lake
// intel_gpu_whl // Whiskey Lake
// intel_gpu_aml // Amber Lake
// intel_gpu_cml // Comet Lake
// intel_gpu_icllp // Ice Lake
// intel_gpu_icl = intel_gpu_icllp // Ice Lake
// intel_gpu_ehl // Elkhart Lake
// intel_gpu_jsl = intel_gpu_ehl // Jasper Lake
// intel_gpu_tgllp // Tiger Lake
// intel_gpu_tgl = intel_gpu_tgllp // Tiger Lake
// intel_gpu_rkl // Rocket Lake
// intel_gpu_adl_s // Alder Lake S
// intel_gpu_rpl_s = intel_gpu_adl_s // Raptor Lake
// intel_gpu_adl_p // Alder Lake P
// intel_gpu_adl_n // Alder Lake N
// intel_gpu_dg1 // DG1
// intel_gpu_acm_g10 // Alchemist G10
// intel_gpu_dg2_g10 = intel_gpu_acm_g10 // Alchemist G10
// intel_gpu_acm_g11 // Alchemist G11
// intel_gpu_dg2_g11 = intel_gpu_acm_g11 // Alchemist G11
// intel_gpu_acm_g12 // Alchemist G12
// intel_gpu_dg2_g12 = intel_gpu_acm_g12 // Alchemist G12
// intel_gpu_pvc // Ponte Vecchio
// intel_gpu_pvc_vg // Ponte Vecchio VG
// intel_gpu_mtl_u // Meteor Lake U
// intel_gpu_mtl_s = intel_gpu_mtl_u // Meteor Lake S
// intel_gpu_arl_u = intel_gpu_mtl_u // Arrow Lake U
// intel_gpu_arl_s = intel_gpu_mtl_u // Arrow Lake S
// intel_gpu_mtl_h // Meteor Lake H
// intel_gpu_arl_h // Arrow Lake H
// intel_gpu_bmg_g21 // Battlemage G21
// intel_gpu_lnl_m // Lunar Lake
//
// Architectures with support for static queries in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/matrix/static-query-use.hpp
// intel_gpu_dg2_g10
// intel_gpu_dg2_g11
// intel_gpu_dg2_g12
// intel_gpu_pvc
//
// ---------------------------------------------
// NVIDIA
// Architectures recognized in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/experimental/device_architecture.hpp
// nvidia_gpu_sm_50
// nvidia_gpu_sm_52
// nvidia_gpu_sm_53
// nvidia_gpu_sm_60
// nvidia_gpu_sm_61
// nvidia_gpu_sm_62
// nvidia_gpu_sm_70
// nvidia_gpu_sm_72
// nvidia_gpu_sm_75
// nvidia_gpu_sm_80
// nvidia_gpu_sm_86
// nvidia_gpu_sm_87
// nvidia_gpu_sm_89
// nvidia_gpu_sm_90
//
// Architectures with support for static queries in oneAPI 2024.2.1:
// /opt/intel/oneapi/compiler/2024.2/bin/compiler/../../include/sycl/ext/oneapi/matrix/static-query-use.hpp
// nvidia_gpu_sm_70
// nvidia_gpu_sm_72
// nvidia_gpu_sm_80
//
// ---------------------------------------------
// Custom check
/*
constexpr architecture nvidia_gpu_family[3] = {
	architecture::nvidia_gpu_sm_70,
	architecture::nvidia_gpu_sm_72,
	architecture::nvidia_gpu_sm_80
};
*/
// Intel (all configuration below are invalid -> compilation failing)
/*
using myparams_intel_gpu_dg2_g10 = matrix_params<architecture::intel_gpu_dg2_g10, sycl::half, sycl::half, sycl::half, sycl::half, tM, tN, tK>;
myparams_intel_gpu_dg2_g10 test_params_intel_gpu_dg2_g10; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_intel_gpu_dg2_g11 = matrix_params<architecture::intel_gpu_dg2_g11, sycl::half, sycl::half, sycl::half, sycl::half, tM, tN, tK>;
myparams_intel_gpu_dg2_g11 test_params_intel_gpu_dg2_g11; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_intel_gpu_dg2_g12 = matrix_params<architecture::intel_gpu_dg2_g12, sycl::half, sycl::half, sycl::half, sycl::half, tM, tN, tK>;
myparams_intel_gpu_dg2_g12 test_params_intel_gpu_dg2_g12; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_intel_gpu_pvc = matrix_params<architecture::intel_gpu_pvc, sycl::half, sycl::half, sycl::half, sycl::half, tM, tN, tK>;
myparams_intel_gpu_pvc test_params_intel_gpu_pvc; // Checking with object definition because internal asserts happen at struct instantiation!
*/

// NVIDIA (all configurations below pass -> these are replaced with runtime query)
/*
using myparams_nvidia_gpu_sm_70 = matrix_params<architecture::nvidia_gpu_sm_70, sycl::half, sycl::half, sycl::half, sycl::half, tM, tN, tK>;
myparams_nvidia_gpu_sm_70 test_params_nvidia_gpu_sm_70; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_nvidia_gpu_sm_72 = matrix_params<architecture::nvidia_gpu_sm_72, sycl::half, sycl::half, sycl::half, sycl::half, tM, tN, tK>;
myparams_nvidia_gpu_sm_72 test_params_nvidia_gpu_sm_72; // Checking with object definition because internal asserts happen at struct instantiation!

using myparams_nvidia_gpu_sm_80 = matrix_params<architecture::nvidia_gpu_sm_80, sycl::half, sycl::half, sycl::half, sycl::half, tM, tN, tK>;
myparams_nvidia_gpu_sm_80 test_params_nvidia_gpu_sm_80; // Checking with object definition because internal asserts happen at struct instantiation!
*/
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
