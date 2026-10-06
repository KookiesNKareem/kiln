#include <cstdint>

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// nap=0: busy loop on globaltimer (issue + control only); nap=1: __nanosleep between polls.
extern "C" __global__ void spin(uint64_t ns, int nap) {
  uint64_t t0 = gtimer();
  while (gtimer() - t0 < ns) {
    if (nap) __nanosleep(2000);
  }
}

// x <- x*x + c with c=-1.9 is chaotic and bounded, so operands toggle like real data.
extern "C" __global__ void k_ffma(float* out, float c, int iters) {
  float x[8];
#pragma unroll
  for (int k = 0; k < 8; k++) x[k] = 0.11f * k - 0.4f + 1e-4f * threadIdx.x;
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int u = 0; u < 4; u++) {
#pragma unroll
      for (int k = 0; k < 8; k++) x[k] = fmaf(x[k], x[k], c);
    }
  }
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) s += x[k];
  if (s == 1234.5f) out[blockIdx.x] = s;
}

#define LD4(op, v, addr, off)                                                    \
  asm volatile(op " {%0,%1,%2,%3}, [%4+" #off "];"                                \
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)                       \
               : "l"(addr))
#define LDS4(v, addr, off)                                                       \
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4+" #off "];"                   \
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)                       \
               : "r"(addr))
#define XOR2(A, P, Q) A.x ^= P.x ^ Q.x; A.y ^= P.y ^ Q.y; A.z ^= P.z ^ Q.z; A.w ^= P.w ^ Q.w

// Every warp reads the same 8 x 512 B rows (4 KiB, lane-consecutive 16 B: conflict-free). The base moves by a runtime
// zero `z` each iteration so ptxas cannot hoist or merge the loads.
#define ROWS8(L, a, p)                                                           \
  {                                                                              \
    uint4 v0, v1, v2, v3, v4, v5, v6, v7;                                        \
    L(v0, p, 0); L(v1, p, 512); L(v2, p, 1024); L(v3, p, 1536);                   \
    L(v4, p, 2048); L(v5, p, 2560); L(v6, p, 3072); L(v7, p, 3584);               \
    XOR2(a, v0, v1); XOR2(a, v2, v3); XOR2(a, v4, v5); XOR2(a, v6, v7);           \
  }
#define LDSV(v, addr, off) LDS4(v, addr, off)
#define LDCA(v, addr, off) LD4("ld.global.ca.v4.u32", v, addr, off)

extern "C" __global__ void smem_rd(unsigned* out, int iters, unsigned z) {
  extern __shared__ uint4 s[];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    unsigned h = i * 2654435761u;
    s[i] = make_uint4(h, h ^ 0x9e3779b9u, h * 7u + 1u, ~h);
  }
  __syncthreads();
  unsigned p = (unsigned)__cvta_generic_to_shared(s) + (threadIdx.x & 31) * 16;
  uint4 a = make_uint4(0, 0, 0, 0);
  for (int i = 0; i < iters; i++) {
    ROWS8(LDSV, a, p);
    p += z;
  }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x1234567u) out[blockIdx.x] = a.x;
}

extern "C" __global__ void l1_rd(const uint4* g, unsigned* out, int iters, unsigned long long z) {
  const char* p = (const char*)g + (threadIdx.x & 31) * 16;
  uint4 a = make_uint4(0, 0, 0, 0);
  for (int i = 0; i < iters; i++) {
    ROWS8(LDCA, a, p);
    p += z;
  }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x1234567u) out[blockIdx.x] = a.x;
}

// Grid-stride ld.global.cg (L1 bypass) over (mask+1) x 16 B, 8 loads in flight per thread.
extern "C" __global__ void gld_cg(const uint4* g, unsigned* out, int iters, unsigned mask) {
  unsigned T = gridDim.x * blockDim.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint4 a = make_uint4(0, 0, 0, 0);
  for (int i = 0; i < iters; i++) {
    const uint4* q[8];
#pragma unroll
    for (int u = 0; u < 8; u++) { q[u] = g + (idx & mask); idx += T; }
    uint4 v0, v1, v2, v3, v4, v5, v6, v7;
    LD4("ld.global.cg.v4.u32", v0, q[0], 0); LD4("ld.global.cg.v4.u32", v1, q[1], 0);
    LD4("ld.global.cg.v4.u32", v2, q[2], 0); LD4("ld.global.cg.v4.u32", v3, q[3], 0);
    LD4("ld.global.cg.v4.u32", v4, q[4], 0); LD4("ld.global.cg.v4.u32", v5, q[5], 0);
    LD4("ld.global.cg.v4.u32", v6, q[6], 0); LD4("ld.global.cg.v4.u32", v7, q[7], 0);
    XOR2(a, v0, v1); XOR2(a, v2, v3); XOR2(a, v4, v5); XOR2(a, v6, v7);
  }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x1234567u) out[blockIdx.x] = a.x;
}

extern "C" __global__ void gcopy(const uint4* src, uint4* dst, int iters, unsigned mask) {
  unsigned T = gridDim.x * blockDim.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < iters; i++) {
    uint4 v[4];
    unsigned j[4];
#pragma unroll
    for (int u = 0; u < 4; u++) { j[u] = idx & mask; idx += T; v[u] = __ldcg(src + j[u]); }
#pragma unroll
    for (int u = 0; u < 4; u++) __stcg(dst + j[u], v[u]);
  }
}
