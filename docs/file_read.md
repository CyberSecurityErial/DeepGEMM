(.venv) chen@dedicated-developjob-wtl-t1wjo-7c6d5f4d56-rcdz5:~/workspace/DeepGEMM$ tree
.
|-- CMakeLists.txt
|-- LICENSE
|-- README.md
|-- build.sh
|-- csrc
|   |-- apis
|   |   |-- attention.hpp
|   |   |-- einsum.hpp
|   |   |-- gemm.hpp
|   |   |-- hyperconnection.hpp
|   |   |-- layout.hpp
|   |   |-- mega.hpp
|   |   `-- runtime.hpp
|   |-- indexing
|   |   `-- main.cu
|   |-- jit
|   |   |-- cache.hpp
|   |   |-- compiler.hpp
|   |   |-- device_runtime.hpp
|   |   |-- handle.hpp
|   |   |-- include_parser.hpp
|   |   `-- kernel_runtime.hpp
|   |-- jit_kernels
|   |   |-- heuristics
|   |   |   |-- common.hpp
|   |   |   |-- config.hpp
|   |   |   |-- mega_moe.hpp
|   |   |   |-- runtime.hpp
|   |   |   |-- sm100.hpp
|   |   |   |-- sm90.hpp
|   |   |   `-- utils.hpp
|   |   `-- impls
|   |       |-- epilogue.hpp
|   |       |-- runtime_utils.hpp
|   |       |-- sm100_bf16_gemm.hpp
|   |       |-- sm100_bmk_bnk_mn.hpp
|   |       |-- sm100_fp8_fp4_gemm_1d1d.hpp
|   |       |-- sm100_fp8_fp4_mega_moe.hpp
|   |       |-- sm100_fp8_gemm_1d1d.hpp
|   |       |-- sm100_tf32_hc_prenorm_gemm.hpp
|   |       |-- sm90_bf16_gemm.hpp
|   |       |-- sm90_bmk_bnk_mn.hpp
|   |       |-- sm90_fp8_gemm_1d1d.hpp
|   |       |-- sm90_fp8_gemm_1d2d.hpp
|   |       |-- sm90_tf32_hc_prenorm_gemm.hpp
|   |       |-- smxx_clean_logits.hpp
|   |       |-- smxx_cublaslt.hpp
|   |       |-- smxx_fp8_fp4_mqa_logits.hpp
|   |       |-- smxx_fp8_fp4_paged_mqa_logits.hpp
|   |       |-- smxx_fp8_mqa_logits.hpp
|   |       |-- smxx_fp8_paged_mqa_logits.hpp
|   |       `-- smxx_layout.hpp
|   |-- python_api.cpp
|   `-- utils
|       |-- compatibility.hpp
|       |-- exception.hpp
|       |-- format.hpp
|       |-- hash.hpp
|       |-- layout.hpp
|       |-- lazy_init.hpp
|       |-- math.hpp
|       `-- system.hpp
|-- deep_gemm
|   |-- __init__.py
|   |-- include
|   |   `-- deep_gemm
|   |       |-- comm
|   |       |   `-- barrier.cuh
|   |       |-- common
|   |       |   |-- compile.cuh
|   |       |   |-- cute_tie.cuh
|   |       |   |-- epilogue_utils.cuh
|   |       |   |-- exception.cuh
|   |       |   |-- math.cuh
|   |       |   |-- reduction.cuh
|   |       |   |-- scheduler.cuh
|   |       |   |-- sm100_utils.cuh
|   |       |   |-- sm90_utils.cuh
|   |       |   |-- tma_copy.cuh
|   |       |   |-- tma_utils.cuh
|   |       |   |-- types.cuh
|   |       |   |-- types.hpp
|   |       |   `-- utils.cuh
|   |       |-- epilogue
|   |       |   |-- sm100_store_cd.cuh
|   |       |   |-- sm100_store_cd_swap_ab.cuh
|   |       |   `-- transform.cuh
|   |       |-- impls
|   |       |   |-- sm100_bf16_gemm.cuh
|   |       |   |-- sm100_bmk_bnk_mn.cuh
|   |       |   |-- sm100_fp4_mqa_logits.cuh
|   |       |   |-- sm100_fp4_paged_mqa_logits.cuh
|   |       |   |-- sm100_fp8_fp4_gemm_1d1d.cuh
|   |       |   |-- sm100_fp8_fp4_mega_moe.cuh
|   |       |   |-- sm100_fp8_gemm_1d1d.cuh
|   |       |   |-- sm100_fp8_mqa_logits.cuh
|   |       |   |-- sm100_fp8_paged_mqa_logits.cuh
|   |       |   |-- sm100_tf32_hc_prenorm_gemm.cuh
|   |       |   |-- sm90_bf16_gemm.cuh
|   |       |   |-- sm90_bmk_bnk_mn.cuh
|   |       |   |-- sm90_fp8_gemm_1d1d.cuh
|   |       |   |-- sm90_fp8_gemm_1d2d.cuh
|   |       |   |-- sm90_fp8_mqa_logits.cuh
|   |       |   |-- sm90_fp8_paged_mqa_logits.cuh
|   |       |   |-- sm90_tf32_hc_prenorm_gemm.cuh
|   |       |   |-- smxx_clean_logits.cuh
|   |       |   `-- smxx_layout.cuh
|   |       |-- layout
|   |       |   |-- mega_moe.cuh
|   |       |   `-- sym_buffer.cuh
|   |       |-- mma
|   |       |   |-- sm100.cuh
|   |       |   `-- sm90.cuh
|   |       |-- ptx
|   |       |   |-- ld_st.cuh
|   |       |   |-- tcgen05.cuh
|   |       |   |-- tma.cuh
|   |       |   |-- utils.cuh
|   |       |   `-- wgmma.cuh
|   |       `-- scheduler
|   |           |-- gemm.cuh
|   |           |-- mega_moe.cuh
|   |           `-- paged_mqa_logits.cuh
|   |-- legacy
|   |   |-- __init__.py
|   |   |-- a_fused_k_grouped_gemm.py
|   |   |-- a_fused_m_grouped_gemm.py
|   |   |-- b_fused_k_grouped_gemm.py
|   |   |-- m_grouped_gemm.py
|   |   `-- tune_options.py
|   |-- mega
|   |   `-- __init__.py     [MegaMoE]: python入口，算子+symbuffer
|   |-- testing
|   |   |-- __init__.py
|   |   |-- bench.py
|   |   |-- numeric.py
|   |   `-- utils.py
|   `-- utils
|       |-- __init__.py
|       |-- dist.py
|       |-- layout.py
|       `-- math.py
|-- develop.sh
|-- docs
|   `-- file_read.md
|-- install.sh
|-- myself_readme_dev.md
|-- scripts
|   |-- generate_pyi.py
|   |-- quick_plot_pm.py
|   `-- run_ncu_mega_moe.sh
|-- setup.py
|-- tests
|   |-- generators.py
|   |-- test_attention.py
|   |-- test_bf16.py
|   |-- test_einsum.py
|   |-- test_fp8_fp4.py
|   |-- test_hyperconnection.py
|   |-- test_layout.py
|   |-- test_lazy_init.py
|   |-- test_legacy.py
|   |-- test_mega_moe.py
|   `-- test_sanitizer.py
`-- third-party
    |-- cutlass
    |-- fmt
    `-- tilelang_ops
        |-- __init__.py
        |-- swiglu_apply_weight_to_fp8.py
        `-- utils.py

31 directories, 141 files