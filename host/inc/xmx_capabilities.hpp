#ifndef XMX_CAPABILITIES
#define XMX_CAPABILITIES

using namespace sycl::ext::oneapi::experimental;

// Run-Time Query of matrix sizes and types configurations

void query_xmx_capabilities (sycl::queue Q){
	// Device architecture
	architecture arch = Q.get_device().get_info<info::device::architecture>();
	printf("\nDetected architecture: ");
	switch (arch) {
		case architecture::x86_64:
			printf("Any CPU device with the x86_64 instruction set");
		break;
		case architecture::intel_gpu_pvc:
			printf("Ponte Vecchio Intel graphics");
		break;
		case architecture::intel_gpu_pvc_vg:
			printf("Ponte Vecchio VG Intel graphics");
		break;
		case architecture::nvidia_gpu_sm_70:
			printf("NVIDIA Volta (compute capability 7.0)");
		break;
		case architecture::nvidia_gpu_sm_72:
			printf("NVIDIA Volta (compute capability 7.2)");
		break;
		case architecture::nvidia_gpu_sm_75:
			printf("NVIDIA Turing (compute capability 7.5)");
		break;
		case architecture::nvidia_gpu_sm_80:
			printf("NVIDIA Ampere (compute capability 8.0)");
		break;
		case architecture::nvidia_gpu_sm_86:
			printf("NVIDIA Ampere (compute capability 8.6)");
		break;
		case architecture::nvidia_gpu_sm_89:
			printf("NVIDIA Ada Lovelace (compute capability 8.9)");
		break;
		case architecture::nvidia_gpu_sm_90:
			printf("NVIDIA Hopper (compute capability 9.0)");
		break;
//		case architecture::nvidia_gpu_sm_90a:
//			printf("NVIDIA Hopper (compute capability 9.0a)");
//		break;
		default:
			printf("Unknown");
		break;
	}

	// Runtime Query of matrix sizes and types configurations
	std::vector<matrix::combination> combinations = Q.get_device().get_info<info::device::matrix_combinations>();
	printf("\nSupported joint-matrix combinations on device: %2i\n", sizeof(combinations));

	for (int i = 0; i < sizeof(combinations); i++) {
		uint invalid = 0;

		printf("\tComb. #%2i: \t", i);

		if (combinations[i].max_msize == 0) {
			printf("m =  %2zu", combinations[i].msize);

			// *********************************************************
			if (combinations[i].max_nsize == 0) {
				printf("\t\tn =  %2zu", combinations[i].nsize);

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			else {
				if (combinations[i].max_nsize <= 64) {
					printf("\t\tn <= %2zu", combinations[i].max_nsize);
				}
				else {
					//printf("[cont. size > 64 (n)!]");
					invalid++;
					goto invalid_combination;
				}

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			// *********************************************************
		}
		else {
			if (combinations[i].max_msize <= 64) {
				printf("m <= %2zu", combinations[i].max_msize);
			}
			else {
				//printf("[cont. size > 64 (m)!]");
				invalid++;
				goto invalid_combination;
			}

			// *********************************************************
			if (combinations[i].max_nsize == 0) {
				printf("\t\tn =  %2zu", combinations[i].nsize);

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			else {
				if (combinations[i].max_nsize <= 64) {
					printf("\t\tn <= %2zu", combinations[i].max_nsize);
				}
				else {
					//printf("[cont. size > 64 (n)!]");
					invalid++;
					goto invalid_combination;
				}

				// ---------------------------------------------------------
				if (combinations[i].max_ksize == 0) {
					printf("\t\tk =  %2zu", combinations[i].ksize);
				}
				else {
					if (combinations[i].max_ksize <= 64) {
						printf("\t\tk <= %2zu", combinations[i].max_ksize);
					}
					else {
						//printf("[cont. size > 64 (k)!]");
						invalid++;
						goto invalid_combination;
					}
				}
				// ---------------------------------------------------------
			}
			// *********************************************************
		}
		invalid_combination:
		if (invalid > 0) {
			printf("\t\t-> Combination not usable!");
		}

		/*
		// Only for debugging
		printf("\tComb. #%2i: \t continuous-sizes (m,n,k): %2zu %2zu %2zu \t|\t discrete sizes (m,n,k): %2zu %2zu %2zu",
			i,
			combinations[i].max_msize, combinations[i].max_nsize, combinations[i].max_ksize,
			combinations[i].msize, combinations[i].nsize, combinations[i].ksize);
		*/

		// Checking support for specific types
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp32 &&
			combinations[i].dtype == matrix::matrix_type::fp32) {
			printf(" <- A (half), B (half), C (float), D (float)");
		}
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp16 &&
			combinations[i].dtype == matrix::matrix_type::fp32) {
			printf(" <- A (half), B (half), C (half), D (float)");
		}
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp32 &&
			combinations[i].dtype == matrix::matrix_type::fp16) {
			printf(" <- A (half), B (half), C (float), D (half)");
		}
		if (combinations[i].atype == matrix::matrix_type::fp16 &&
			combinations[i].btype == matrix::matrix_type::fp16 &&
			combinations[i].ctype == matrix::matrix_type::fp16 &&
			combinations[i].dtype == matrix::matrix_type::fp16) {
			printf(" <- A (half), B (half), C (half), D (half)");
		}
		if (combinations[i].atype == matrix::matrix_type::tf32 &&
			combinations[i].btype == matrix::matrix_type::tf32 &&
			combinations[i].ctype == matrix::matrix_type::fp32 &&
			combinations[i].dtype == matrix::matrix_type::fp32) {
			printf(" <- A (tf32), B (tf32), C (float), D (float)");
		}

		// Checking types supported for a specific combination
		if (
			(combinations[i].msize == 32) &&
			(combinations[i].nsize == 64) &&
			(combinations[i].ksize == 16)
		) {
			switch (combinations[i].atype)	{
				case matrix::matrix_type::bf16:
					printf(" <- A (bf16)"); // Supported on PVC 11.11.2024
					if (combinations[i].btype == matrix::matrix_type::bf16) {
						printf(", B (bf16)"); // Supported on PVC 11.11.2024
						if (combinations[i].ctype == matrix::matrix_type::fp32) {
							printf(", C (float)"); // Supported on PVC 11.11.2024
							if (combinations[i].dtype == matrix::matrix_type::fp32) {
								printf(", D (float)"); // Supported on PVC 11.11.2024
							} else {
								printf("\t D (other types than fp32)");
							}
						}
						else {
							printf("\t C (other types than fp32)");
						}
					}
					else {
						printf("\t B (other types than bf16)");
					}
					break;
				case matrix::matrix_type::fp16:
					printf("\n\t A (fp16)");
					break;
				case matrix::matrix_type::tf32:
					printf("\n\t A (tf32)");
					break;
				case matrix::matrix_type::fp32:
					printf("\n\t A (fp32)");
					break;
				case matrix::matrix_type::fp64:
					printf("\n\t A (fp64)");
					break;
				case matrix::matrix_type::sint8:
					printf("\n\t A (sint8)");
					break;
				case matrix::matrix_type::sint16:
					printf("\n\t A (sint16)");
					break;
				case matrix::matrix_type::sint32:
					printf("\n\t A (sint32)");
					break;
				case matrix::matrix_type::sint64:
					printf("\n\t A (sint64)");
					break;
				case matrix::matrix_type::uint8:
					printf("\n\t A (uint8)");
					break;
				case matrix::matrix_type::uint16:
					printf("\n\t A (uint16)");
					break;
				case matrix::matrix_type::uint32:
					printf("\n\t A (uint32)");
					break;
				case matrix::matrix_type::uint64:
					printf("\n\t A (uint64)");
					break;
			}

		}


		printf("\n");
	}
}

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

#endif // XMX_CAPABILITIES


