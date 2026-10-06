#!/bin/bash
# Short single-launch runs under Nsight Compute to confirm analytic work counts.
cd "$(dirname "$0")"
NCU=$(command -v ncu || ls /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
echo "ncu: $NCU"
[ -z "$NCU" ] && exit 1
M=smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed_op_shared_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum,sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_imma.sum,smsp__inst_executed.sum,gpu__time_duration.sum
for spec in "ffma full 64" "smem_read full 64" "l1_read full 64" "l2_read full 64" "hbm_read full 64" "hbm_copy full 16" "gemm_bf16 full" "gemm_int8 full"; do
  set -- $spec
  echo "=== $spec"
  if [ "${1#gemm}" != "$1" ]; then F="--profile-from-start off"; else F="-k regex:ffma|smem_rd|l1_rd|gld_cg|gcopy"; fi
  "$NCU" --metrics $M $F -c 1 --csv python energy_bench.py once "$@" 2>&1 | grep -v '^==PROF==' | tail -20
done
