#include <c10/util/Half.h>
#include "filtered_lrelu.h"
#include <cstdint>

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>
{
    typedef double scalar_t; typedef double2 vec2_t; typedef double4 vec4_t;
    __device__ __forceinline__ static vec2_t zero_vec2(void) { return make_double2(0, 0); }
    __device__ __forceinline__ static vec4_t zero_vec4(void) { return make_double4(0, 0, 0, 0); }
};
template <> struct InternalType<float>
{
    typedef float scalar_t; typedef float2 vec2_t; typedef float4 vec4_t;
    __device__ __forceinline__ static vec2_t zero_vec2(void) { return make_float2(0, 0); }
    __device__ __forceinline__ static vec4_t zero_vec4(void) { return make_float4(0, 0, 0, 0); }
};
template <> struct InternalType<c10::Half>
{
    typedef float scalar_t; typedef float2 vec2_t; typedef float4 vec4_t;
    __device__ __forceinline__ static vec2_t zero_vec2(void) { return make_float2(0, 0); }
    __device__ __forceinline__ static vec4_t zero_vec4(void) { return make_float4(0, 0, 0, 0); }
};

#define MIN(A, B)       ((A) < (B) ? (A) : (B))
#define MAX(A, B)       ((A) > (B) ? (A) : (B))
//#define FLOOR_DIV(A, B) (((A) - ((A) < 0 ? (B) - 1 : 0)) / (B))
//#define CEIL_DIV(A, B)  (((A) + ((A) > 0 ? (B) - 1 : 0)) / (B))
#define CEIL_DIV(A, B) (((B)==1) ? (A) : \
                        ((B)==2) ? ((int)((A)+1) >> 1) : \
                        ((B)==4) ? ((int)((A)+3) >> 2) : \
                        (((A) + ((A) > 0 ? (B) - 1 : 0)) / (B)))
#define NICE_MODULO(A, B)   (((B)==1) ? 0 : \
                            ((B)==2) ? ((A) & 1) : \
                            ((B)==4) ? ((A) & 3) : \
                            ((A)>=0) ? ((A) % (B)) : ((B) - ((-(A)-1) % (B)) - 1))

// This works only up to blocks of size 256 x 256 and for all N that are powers of two.
template <int N> __device__ __forceinline__ void fast_div_mod(int& x, int& y, unsigned int i)
{
    // Check that N is valid for our formula in compile time.
    //static_assert(N <= 256 || !(N & (N-1)), "invalid divisor for fast_div_mod");

    if ((N & (N-1)) && N <= 256)
        y = (i * ((1<<24)/N + 1)) >> 24; // Assumes N <= 256, i < N*256.
    else
        y = i/N;

    x = i - y*N;
}

// Ensure that N is known at compile time. For debugging.
template <int N> __device__ __forceinline__ void assert_ct_constant(void) {}

//------------------------------------------------------------------------
// Filters.

enum
{
    MODE_SUSD = 0,
    MODE_FUSD = 1,
    MODE_SUFD = 2,
    MODE_FUFD = 3,
    MODE_SKIP = 4,
};

enum
{
    SIGN_WRITE   = 0,
    SIGN_READ    = 1,
    SIGN_UNKNOWN = 2
};

#define MAX_FILTER_SIZE 48
__constant__ float c_fu_small[MAX_FILTER_SIZE * MAX_FILTER_SIZE];
__constant__ float c_fd_small[MAX_FILTER_SIZE * MAX_FILTER_SIZE];
__constant__ float* p_fu_small = c_fu_small;
__constant__ float* p_fd_small = c_fd_small;

#define BIG_FILTER_SIZE 64 // For oversized separable kernels.
__constant__ float c_fu_big[BIG_FILTER_SIZE];
__constant__ float c_fd_big[BIG_FILTER_SIZE];
__constant__ float* p_fu_big = c_fu_big;
__constant__ float* p_fd_big = c_fd_big;

static __global__ void setup_filters_kernel(filtered_lrelu_kernel_params p)
{
    if (p.fuShape.x > MAX_FILTER_SIZE || p.fdShape.x > MAX_FILTER_SIZE)
    {
        // Oversized separable filter.
        for (int idx = threadIdx.x; idx < BIG_FILTER_SIZE; idx += blockDim.x)
        {
            int x = idx;
            int fu_x = p.flip ? p.fuShape.x - 1 - x : x;
            int fd_x = p.flip ? p.fdShape.x - 1 - x : x;
            p_fu_big[idx] = (x >= p.fuShape.x) ? 0.0f : p.fu[fu_x * p.fuStride.x];
            p_fd_big[idx] = (x >= p.fdShape.x) ? 0.0f : p.fd[fd_x * p.fdStride.x];
        }
        return;
    }

    for (int idx = threadIdx.x; idx < MAX_FILTER_SIZE * MAX_FILTER_SIZE; idx += blockDim.x)
    {
        int x, y;
        fast_div_mod<MAX_FILTER_SIZE>(x, y, idx);
        int fu_x = p.flip ? p.fuShape.x - 1 - x : x;
        int fu_y = p.flip ? p.fuShape.y - 1 - y : y;
        int fd_x = p.flip ? p.fdShape.x - 1 - x : x;
        int fd_y = p.flip ? p.fdShape.y - 1 - y : y;
        p_fu_small[idx] = (x >= p.fuShape.x || y >= p.fuShape.y) ? 0.0f : p.fu[fu_x * p.fuStride.x + fu_y * p.fuStride.y];
        p_fd_small[idx] = (x >= p.fdShape.x || y >= p.fdShape.y) ? 0.0f : p.fd[fd_x * p.fdStride.x + fd_y * p.fdStride.y];
    }
}

//------------------------------------------------------------------------
// Coordinate spaces:
// - Relative to input tensor:      inX, inY, tileInX, tileInY
// - Relative to input tile:        relInX, relInY, tileInW, tileInH
// - Relative to upsampled tile:    relUpX, relUpY, tileUpW, tileUpH
// - Relative to output tile:       relOutX, relOutY, tileOutW, tileOutH
// - Relative to output tensor:     outX, outY, tileOutX, tileOutY
//
// Relationships between coordinate spaces:
// - inX = tileInX + relInX
// - inY = tileInY + relInY
// - relUpX = relInX * up + phaseInX
// - relUpY = relInY * up + phaseInY
// - relUpX = relOutX * down
// - relUpY = relOutY * down
// - outX = tileOutX + relOutX
// - outY = tileOutY + relOutY

// When sharedKB == 0, allocate shared memory statically inside the kernel, otherwise use the externally allocated shared memory buffer.
extern __shared__ char s_buf_raw[];
template <class T, int sharedKB, int signMode, int filterMode_, int up, int fuSize, int down, int fdSize, int tileOutW, int tileOutH, int threadsPerBlock, bool enableXrep>
static __global__ void filtered_lrelu_kernel(filtered_lrelu_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    typedef typename InternalType<T>::vec2_t vec2_t;
    typedef typename InternalType<T>::vec4_t vec4_t;
    const int tileUpW_ = tileOutW * down + (fdSize - 1) - (down - 1);
    const int tileUpW = (tileUpW_ + 3) & ~3; // Round up to multiple of 4 for sign byte alignment and horizontal upsampling.
    const int tileUpH = tileOutH * down + (fdSize - 1) - (down - 1);
    const int tileUpH_ = (up == 2 || up == 4) ? (tileUpH + up - 1) & ~(up - 1) : tileUpH; // Round up to a multiple of up. Never actually produce tiles of this size.
    const int tileInW  = CEIL_DIV(tileUpW  + (fuSize - 1), up);
    const int tileInH  = CEIL_DIV(tileUpH  + (fuSize - 1), up);
    const int tileInH_ = CEIL_DIV(tileUpH_ + (fuSize - 1), up); // Use rounded-up size for allocations to avoid overruns with up=2 and up=4 variants.
    const bool transposeInput = !(up == 1 || up == 2 || up == 4); // Transpose only if using general code variants.
    const bool writeSigns = (signMode == SIGN_WRITE) ? true : (signMode == SIGN_READ) ? false : !!p.so;
    const int filterMode = filterMode_;
    //const int filterMode = MODE_SKIP; // Benchmark just the cost of memory accesses.

    const int szIn    = tileInH_ * tileInW;
    const int szUpX   = tileInH_ * tileUpW;
    const int szUpXY  = tileUpH * tileUpW;
    const int szDownX = tileUpH * tileOutW;

    // Access to kernels.
    const bool oversized = (fdSize > MAX_FILTER_SIZE || fuSize > MAX_FILTER_SIZE);
    const float* const c_fu = oversized ? c_fu_big : c_fu_small;
    const float* const c_fd = oversized ? c_fd_big : c_fd_small;

    // Sizes for shared memory arrays.
    const int s_buf0_size_base =
        (filterMode == MODE_SUSD) ? MAX(szIn, szUpXY) :
        (filterMode == MODE_FUSD) ? MAX(szIn, szDownX) :
        (filterMode == MODE_SUFD) ? MAX(szIn, szUpXY) :
        (filterMode == MODE_FUFD) ? szIn :
        (filterMode == MODE_SKIP) ? szIn :
        -1;
    const int s_buf1_size_base =
        (filterMode == MODE_SUSD) ? MAX(szUpX, szDownX) :
        (filterMode == MODE_FUSD) ? szUpXY :
        (filterMode == MODE_SUFD) ? szUpX  :
        (filterMode == MODE_FUFD) ? szUpXY :
        (filterMode == MODE_SKIP) ? 1 :
        -1;

    // Ensure U128 alignment.
    const int s_buf0_size = (s_buf0_size_base + 3) & ~3;
    const int s_buf1_size = (s_buf1_size_base + 3) & ~3;

    // Check at compile time that we don't use too much shared memory.
    static_assert((s_buf0_size + s_buf1_size) * sizeof(scalar_t) <= ((sharedKB ? sharedKB : 48) << 10), "shared memory overflow");

    // Declare shared memory arrays.
    scalar_t* s_buf0;
    scalar_t* s_buf1;
    if (sharedKB == 0)
    {
        // Allocate shared memory arrays here.
        __shared__ scalar_t s_buf0_st[sharedKB ? (1<<24) : s_buf0_size]; // Prevent launching if these aren't optimized away when unused.
        __shared__ scalar_t s_buf1_st[sharedKB ? (1<<24) : s_buf1_size];
        s_buf0 = s_buf0_st;
        s_buf1 = s_buf1_st;
    }
    else
    {
        // Use the dynamically allocated shared memory array.
        s_buf0 = (scalar_t*)s_buf_raw;
        s_buf1 = s_buf0 + s_buf0_size;
    }

    // Pointers to the buffers.
    scalar_t* s_tileIn;       // Input tile:                      [relInX * tileInH + relInY] (possibly transposed)
    scalar_t* s_tileUpX;      // After horizontal upsampling:     [relInY * tileUpW + relUpX]
    scalar_t* s_tileUpXY;     // After upsampling:                [relUpY * tileUpW + relUpX]
    scalar_t* s_tileDownX;    // After horizontal downsampling:   [relUpY * tileOutW + relOutX]
    if (filterMode == MODE_SUSD)
    {
        s_tileIn    = s_buf0;
        s_tileUpX   = s_buf1;
        s_tileUpXY  = s_buf0;
        s_tileDownX = s_buf1;
    }
    else if (filterMode == MODE_FUSD)
    {
        s_tileIn    = s_buf0;
        s_tileUpXY  = s_buf1;
        s_tileDownX = s_buf0;
    }
    else if (filterMode == MODE_SUFD)
    {
        s_tileIn    = s_buf0;
        s_tileUpX   = s_buf1;
        s_tileUpXY  = s_buf0;
    }
    else if (filterMode == MODE_FUFD)
    {
        s_tileIn    = s_buf0;
        s_tileUpXY  = s_buf1;
    }
    else if (filterMode == MODE_SKIP)
    {
        s_tileIn    = s_buf0;
        s_tileUpXY  = s_buf0;
    }

    int channelIdx = blockIdx.z;
    int batchIdx = channelIdx / p.yShape.z;
    channelIdx -= batchIdx * p.yShape.z;

    // Inner tile loop.
    #pragma unroll 1
    for (int tileIdx = 0; !enableXrep || (tileIdx < MIN(p.tilesXrep, p.tilesXdim - p.tilesXrep * blockIdx.y)); tileIdx++)
    {
        // Locate output tile.
        int tileX = enableXrep ? blockIdx.y * p.tilesXrep + tileIdx : blockIdx.x;
        int tileOutX = tileX * tileOutW;
        int tileOutY = (enableXrep ? blockIdx.x : blockIdx.y) * tileOutH;

        // Locate input tile.
        int tmpX = tileOutX * down - p.pad0.x;
        int tmpY = tileOutY * down - p.pad0.y;
        int tileInX = CEIL_DIV(tmpX, up);
        int tileInY = CEIL_DIV(tmpY, up);
        const int phaseInX = tileInX * up - tmpX;
        const int phaseInY = tileInY * up - tmpY;

        // Extra sync if input and output buffers are the same and we are not on first tile.
        if (enableXrep && tileIdx > 0 && (filterMode == MODE_FUSD || filterMode == MODE_SUFD))
            __syncthreads();

        // Load input tile & apply bias. Unrolled.
        scalar_t b = (scalar_t)*(const T*)((const char*)p.b + (channelIdx * p.bStride));
        int mapOfsIn = channelIdx * p.xStride.z + batchIdx * p.xStride.w;
        int idx = threadIdx.x;
        const int loopCountIN = CEIL_DIV(tileInW * tileInH, threadsPerBlock);
        #pragma unroll
        for (int loop = 0; loop < loopCountIN; loop++)
        {
            int relInX, relInY;
            fast_div_mod<tileInW>(relInX, relInY, idx);
            int inX = tileInX + relInX;
            int inY = tileInY + relInY;
            scalar_t v = 0;
            if ((unsigned int)inX < (unsigned int)p.xShape.x & (unsigned int)inY < (unsigned int)p.xShape.y)
                v = (scalar_t)*((const T*)((const char*)p.x + (inX * p.xStride.x + inY * p.xStride.y + mapOfsIn))) + b;

            bool skip = (loop == loopCountIN-1) && (idx >= tileInW * tileInH);
            if (!skip)
                s_tileIn[transposeInput ? relInX * tileInH + relInY : relInX + tileInW * relInY] = v;

            idx += threadsPerBlock;
        }

        if (filterMode == MODE_SUSD || filterMode == MODE_SUFD) // Separable upsampling filter.
        {
            // Horizontal upsampling.
            __syncthreads();
            if (up == 4)
            {
                for (int idx = threadIdx.x*up; idx < tileUpW * tileInH; idx += blockDim.x*up)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileUpW>(relUpX0, relInY, idx);
                    int relInX0 = relUpX0 / up;
                    int src0 = relInX0 + tileInW * relInY;
                    int dst = relInY * tileUpW + relUpX0;
                    vec4_t v = InternalType<T>::zero_vec4();
                    scalar_t x = s_tileIn[src0];
                    if (phaseInX == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 0];
                            x = s_tileIn[src0 + step + 1];
                            v.y += x * (scalar_t)c_fu[step * up + 3];
                            v.z += x * (scalar_t)c_fu[step * up + 2];
                            v.w += x * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else if (phaseInX == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 1];
                            v.y += x * (scalar_t)c_fu[step * up + 0];
                            x = s_tileIn[src0 + step + 1];
                            v.z += x * (scalar_t)c_fu[step * up + 3];
                            v.w += x * (scalar_t)c_fu[step * up + 2];
                        }
                    }
                    else if (phaseInX == 2)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 2];
                            v.y += x * (scalar_t)c_fu[step * up + 1];
                            v.z += x * (scalar_t)c_fu[step * up + 0];
                            x = s_tileIn[src0 + step + 1];
                            v.w += x * (scalar_t)c_fu[step * up + 3];
                        }
                    }
                    else // (phaseInX == 3)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 3];
                            v.y += x * (scalar_t)c_fu[step * up + 2];
                            v.z += x * (scalar_t)c_fu[step * up + 1];
                            v.w += x * (scalar_t)c_fu[step * up + 0];
                            x = s_tileIn[src0 + step + 1];
                        }
                    }
                    s_tileUpX[dst+0] = v.x;
                    s_tileUpX[dst+1] = v.y;
                    s_tileUpX[dst+2] = v.z;
                    s_tileUpX[dst+3] = v.w;
                }
            }
            else if (up == 2)
            {
                bool p0 = (phaseInX == 0);
                for (int idx = threadIdx.x*up; idx < tileUpW * tileInH; idx += blockDim.x*up)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileUpW>(relUpX0, relInY, idx);
                    int relInX0 = relUpX0 / up;
                    int src0 = relInX0 + tileInW * relInY;
                    int dst = relInY * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    scalar_t x = s_tileIn[src0];
                    if (p0) // (phaseInX == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 0];
                            x = s_tileIn[src0 + step + 1];
                            v.y += x * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else // (phaseInX == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 1];
                            v.y += x * (scalar_t)c_fu[step * up + 0];
                            x = s_tileIn[src0 + step + 1];
                        }
                    }
                    s_tileUpX[dst+0] = v.x;
                    s_tileUpX[dst+1] = v.y;
                }
            }
            else if (up == 1)
            {
                // Normal order with non-transposed input.
                for (int idx = threadIdx.x; idx < tileUpW * tileInH; idx += blockDim.x)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileUpW>(relUpX0, relInY, idx);
                    int relInX0 = relUpX0;
                    int src0 = relInX0 + tileInW * relInY;
                    int dst = relInY * tileUpW + relUpX0;
                    scalar_t v = 0;
                    #pragma unroll
                    for (int step = 0; step < fuSize / up; step++)
                        v += s_tileIn[src0 + step] * (scalar_t)c_fu[step * up];
                    s_tileUpX[dst] = v;
                }
            }
            else
            {
                // General variant, assumes transposed input.
                for (int idx = threadIdx.x; idx < tileUpW * tileInH; idx += blockDim.x)
                {
                    int relUpX0, relInY;
                    fast_div_mod<tileInH>(relInY, relUpX0, idx);
                    int relInX0 = CEIL_DIV(relUpX0 - phaseInX, up);
                    int src0 = relInX0 * tileInH + relInY;
                    int tap0 = relInX0 * up + phaseInX - relUpX0;
                    scalar_t v = 0;
                    #pragma unroll
                    for (int step = 0; step < fuSize / up; step++)
                        v += s_tileIn[src0 + step * tileInH] * (scalar_t)c_fu[tap0 + step * up];
                    s_tileUpX[relInY * tileUpW + relUpX0] = v;
                }
            }

            // Vertical upsampling & nonlinearity.

            __syncthreads();
            int groupMask = 15 << ((threadIdx.x & 31) & ~3);
            int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH + p.sOfs.y : 0; // Skip already written signs.
            int sShapeMaxY = MIN(p.sShape.y, tileOutY * down + tileUpH + p.sOfs.y); // Avoid out-of-tile sign writes.
            if (up == 4)
            {
                minY -= 3; // Adjust according to block height.
                for (int idx = threadIdx.x; idx < tileUpW * tileUpH_ / up; idx += blockDim.x)
                {
                    int relUpX, relInY0;
                    fast_div_mod<tileUpW>(relUpX, relInY0, idx);
                    int relUpY0 = relInY0 * up;
                    int src0 = relInY0 * tileUpW + relUpX;
                    int dst = relUpY0 * tileUpW + relUpX;
                    vec4_t v = InternalType<T>::zero_vec4();

                    #if 0
                        // Prefetch data into registers.
                        scalar_t s_in[fuSize / up + 1];
                        #pragma unroll
                        for (int step = 0; step <= fuSize / up; step++)
                            s_in[step] = s_tileUpX[src0 + step * tileUpW];
                        #define S_IN(X) s_in[(X)]
                    #else
                        // Fetch as we go.
                        #define S_IN(STEP) s_tileUpX[src0 + (STEP) * tileUpW]
                    #endif

                    scalar_t x = S_IN(0);
                    if (phaseInY == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 0];
                            x = S_IN(step + 1);
                            v.y += x * (scalar_t)c_fu[step * up + 3];
                            v.z += x * (scalar_t)c_fu[step * up + 2];
                            v.w += x * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else if (phaseInY == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 1];
                            v.y += x * (scalar_t)c_fu[step * up + 0];
                            x = S_IN(step + 1);
                            v.z += x * (scalar_t)c_fu[step * up + 3];
                            v.w += x * (scalar_t)c_fu[step * up + 2];
                        }
                    }
                    else if (phaseInY == 2)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 2];
                            v.y += x * (scalar_t)c_fu[step * up + 1];
                            v.z += x * (scalar_t)c_fu[step * up + 0];
                            x = S_IN(step + 1);
                            v.w += x * (scalar_t)c_fu[step * up + 3];
                        }
                    }
                    else // (phaseInY == 3)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 3];
                            v.y += x * (scalar_t)c_fu[step * up + 2];
                            v.z += x * (scalar_t)c_fu[step * up + 1];
                            v.w += x * (scalar_t)c_fu[step * up + 0];
                            x = S_IN(step + 1);
                        }
                    }
                    #undef S_IN

                    int signX = tileOutX * down + relUpX + p.sOfs.x;
                    int signY = tileOutY * down + relUpY0 + p.sOfs.y;
                    int signXb = (signX + p.sByteOfs) >> 2;
                    int signXo2 = ((signX + p.sByteOfs) & 3) << 1;
                    int signIdx0 = signXb * p.sStride.x + signY * p.sStride.y + channelIdx * p.sStride.z + batchIdx * p.sStride.w;
                    int signIdx1 = signIdx0 + p.sStride.y;
                    int signIdx2 = signIdx1 + p.sStride.y;
                    int signIdx3 = signIdx2 + p.sStride.y;
                    v.x *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y *= (scalar_t)((float)up * (float)up * p.gain);
                    v.z *= (scalar_t)((float)up * (float)up * p.gain);
                    v.w *= (scalar_t)((float)up * (float)up * p.gain);

                    if (writeSigns) // Determine and write signs.
                    {
                        int sx = __float_as_uint(v.x) >> 31 <<  0;
                        int sy = __float_as_uint(v.y) >> 31 <<  8;
                        int sz = __float_as_uint(v.z) >> 31 << 16;
                        int sw = __float_as_uint(v.w) >> 31 << 24;
                        if (sx) v.x *= p.slope;
                        if (sy) v.y *= p.slope;
                        if (sz) v.z *= p.slope;
                        if (sw) v.w *= p.slope;
                        if (fabsf(v.x) > p.clamp) { sx = 2 <<  0; v.x = copysignf(p.clamp, v.x); }
                        if (fabsf(v.y) > p.clamp) { sy = 2 <<  8; v.y = copysignf(p.clamp, v.y); }
                        if (fabsf(v.z) > p.clamp) { sz = 2 << 16; v.z = copysignf(p.clamp, v.z); }
                        if (fabsf(v.w) > p.clamp) { sw = 2 << 24; v.w = copysignf(p.clamp, v.w); }

                        if ((unsigned int)signXb < (unsigned int)p.sShape.x & signY >= minY)
                        {
                            // Combine signs.
                            unsigned int ss = sx + sy + sw + sz;
                            ss <<= signXo2;
                            ss |= __shfl_xor_sync(groupMask, ss, 1);
                            ss |= __shfl_xor_sync(groupMask, ss, 2);

                            // Write signs.
                            if ((unsigned int)(signY + 0) < (unsigned int)sShapeMaxY) { p.so[signIdx0] = (unsigned char)(ss >>  0);
                            if ((unsigned int)(signY + 1) < (unsigned int)sShapeMaxY) { p.so[signIdx1] = (unsigned char)(ss >>  8);
                            if ((unsigned int)(signY + 2) < (unsigned int)sShapeMaxY) { p.so[signIdx2] = (unsigned char)(ss >> 16);
                            if ((unsigned int)(signY + 3) < (unsigned int)sShapeMaxY) { p.so[signIdx3] = (unsigned char)(ss >> 24); } } } } // Cursed.
                        }
                    }
                    else // Read signs and apply.
                    {
                        if ((unsigned int)signX < (unsigned int)p.sWidth)
                        {
                            if ((unsigned int)(signY + 0) < (unsigned int)p.sShape.y) { int s = p.si[signIdx0] >> signXo2; if (s & 1) v.x *= p.slope; if (s & 2) v.x = 0.f;
                            if ((unsigned int)(signY + 1) < (unsigned int)p.sShape.y) { int s = p.si[signIdx1] >> signXo2; if (s & 1) v.y *= p.slope; if (s & 2) v.y = 0.f;
                            if ((unsigned int)(signY + 2) < (unsigned int)p.sShape.y) { int s = p.si[signIdx2] >> signXo2; if (s & 1) v.z *= p.slope; if (s & 2) v.z = 0.f;
                            if ((unsigned int)(signY + 3) < (unsigned int)p.sShape.y) { int s = p.si[signIdx3] >> signXo2; if (s & 1) v.w *= p.slope; if (s & 2) v.w = 0.f; } } } }
                        }
                    }

                    s_tileUpXY[dst + 0 * tileUpW] = v.x;
                    if (relUpY0 < tileUpH - 1) s_tileUpXY[dst + 1 * tileUpW] = v.y;
                    if (relUpY0 < tileUpH - 2) s_tileUpXY[dst + 2 * tileUpW] = v.z;
                    if (relUpY0 < tileUpH - 3) s_tileUpXY[dst + 3 * tileUpW] = v.w;
                }
            }
            else if (up == 2)
            {
                minY -= 1; // Adjust according to block height.
                for (int idx = threadIdx.x; idx < tileUpW * tileUpH_ / up; idx += blockDim.x)
                {
                    int relUpX, relInY0;
                    fast_div_mod<tileUpW>(relUpX, relInY0, idx);
                    int relUpY0 = relInY0 * up;
                    int src0 = relInY0 * tileUpW + relUpX;
                    int dst = relUpY0 * tileUpW + relUpX;
                    vec2_t v = InternalType<T>::zero_vec2();

                    #if 0
                        // Prefetch data into registers.
                        scalar_t s_in[fuSize / up + 1];
                        #pragma unroll
                        for (int step = 0; step <= fuSize / up; step++)
                            s_in[step] = s_tileUpX[src0 + step * tileUpW];
                        #define S_IN(X) s_in[(X)]
                    #else
                        // Fetch as we go.
                        #define S_IN(STEP) s_tileUpX[src0 + (STEP) * tileUpW]
                    #endif

                    scalar_t x = S_IN(0);
                    if (phaseInY == 0)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 0];
                            x = S_IN(step + 1);
                            v.y += x * (scalar_t)c_fu[step * up + 1];
                        }
                    }
                    else // (phaseInY == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                        {
                            v.x += x * (scalar_t)c_fu[step * up + 1];
                            v.y += x * (scalar_t)c_fu[step * up + 0];
                            x = S_IN(step + 1);
                        }
                    }
                    #undef S_IN

                    int signX = tileOutX * down + relUpX + p.sOfs.x;
                    int signY = tileOutY * down + relUpY0 + p.sOfs.y;
                    int signXb = (signX + p.sByteOfs) >> 2;
                    int signXo2 = ((signX + p.sByteOfs) & 3) << 1;
                    int signIdx0 = signXb * p.sStride.x + signY * p.sStride.y + channelIdx * p.sStride.z + batchIdx * p.sStride.w;
                    int signIdx1 = signIdx0 + p.sStride.y;
                    v.x *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y *= (scalar_t)((float)up * (float)up * p.gain);

                    if (writeSigns) // Determine and write signs.
                    {
                        int sx = __float_as_uint(v.x) >> 31 << 0;
                        int sy = __float_as_uint(v.y) >> 31 << 8;
                        if (sx) v.x *= p.slope;
                        if (sy) v.y *= p.slope;
                        if (fabsf(v.x) > p.clamp) { sx = 2 << 0; v.x = copysignf(p.clamp, v.x); }
                        if (fabsf(v.y) > p.clamp) { sy = 2 << 8; v.y = copysignf(p.clamp, v.y); }

                        if ((unsigned int)signXb < (unsigned int)p.sShape.x & signY >= minY)
                        {
                            // Combine signs.
                            int ss = sx + sy;
                            ss <<= signXo2;
                            ss |= __shfl_xor_sync(groupMask, ss, 1);
                            ss |= __shfl_xor_sync(groupMask, ss, 2);

                            // Write signs.
                            if ((unsigned int)(signY + 0) < (unsigned int)sShapeMaxY) { p.so[signIdx0] = (unsigned char)(ss >>  0); }
                            if ((unsigned int)(signY + 1) < (unsigned int)sShapeMaxY) { p.so[signIdx1] = (unsigned char)(ss >>  8); }
                        }
                    }
                    else // Read signs and apply.
                    {
                        if ((unsigned int)signX < (unsigned int)p.sWidth)
                        {
                            if ((unsigned int)(signY + 0) < (unsigned int)p.sShape.y) { int s = p.si[signIdx0] >> signXo2; if (s & 1) v.x *= p.slope; if (s & 2) v.x = 0.f; }
                            if ((unsigned int)(signY + 1) < (unsigned int)p.sShape.y) { int s = p.si[signIdx1] >> signXo2; if (s & 1) v.y *= p.slope; if (s & 2) v.y = 0.f; }
                        }
                    }

                    s_tileUpXY[dst + 0 * tileUpW] = v.x;
                    if (relUpY0 < tileUpH - 1) s_tileUpXY[dst + 1 * tileUpW] = v.y;
                }
            }
            else
            {
                // General case for up=1 and other values.
                minY -= (up - 1); // Adjust according to block height.
                for (int idx = threadIdx.x; idx < tileUpW * tileUpH; idx += blockDim.x)
                {
                    int relUpX, relUpY0;
                    fast_div_mod<tileUpW>(relUpX, relUpY0, idx);
                    int relInY0 = CEIL_DIV(relUpY0 - phaseInY, up);
                    int src0 = relInY0 * tileUpW + relUpX;
                    int tap0 = relInY0 * up + phaseInY - relUpY0;
                    scalar_t v = 0;

                    #if 1
                        // Prefetch data into registers.
                        scalar_t s_in[fuSize / up];
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                            s_in[step] = s_tileUpX[src0 + step * tileUpW];
                        #define S_IN(X) s_in[X]
                    #else
                        // Fetch as we go.
                        #define S_IN(STEP) s_tileUpX[src0 + STEP * tileUpW]
                    #endif

                    if (up == 1)
                    {
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                            v += S_IN(step) * (scalar_t)c_fu[0 + step * up];
                    }
                    else
                    {
                        // General case.
                        #pragma unroll
                        for (int step = 0; step < fuSize / up; step++)
                            v += S_IN(step) * (scalar_t)c_fu[tap0 + step * up];
                    }
                    #undef S_IN

                    int signX = tileOutX * down + relUpX + p.sOfs.x;
                    int signY = tileOutY * down + relUpY0 + p.sOfs.y;
                    int signXb = (signX + p.sByteOfs) >> 2;
                    int signXo2 = ((signX + p.sByteOfs) & 3) << 1;
                    int signIdx = signXb * p.sStride.x + signY * p.sStride.y + channelIdx * p.sStride.z + batchIdx * p.sStride.w;
                    v *= (scalar_t)((float)up * (float)up * p.gain);

                    if (writeSigns) // Determine and write sign.
                    {
                        int s = __float_as_uint(v) >> 31; // Bit 0 = sign.
                        if (s) v *= p.slope;
                        if (fabsf(v) > p.clamp)
                        {
                            s = 2; // Bit 1 = clamp.
                            v = copysignf(p.clamp, v);
                        }

                        // Write signs.
                        bool signValidWrite = ((unsigned int)signXb < (unsigned int)p.sShape.x & signY >= minY & (unsigned int)signY < (unsigned int)p.sShape.y);
                        if (signValidWrite)
                        {
                            s <<= signXo2;
                            s |= __shfl_xor_sync(groupMask, s, 1);
                            s |= __shfl_xor_sync(groupMask, s, 2);
                            p.so[signIdx] = s;
                        }
                    }
                    else // Read sign and apply.
                    {
                        bool signValidRead = ((unsigned int)signX < (unsigned int)p.sWidth & (unsigned int)signY < (unsigned int)p.sShape.y);
                        if (signValidRead)
                        {
                            int s = p.si[signIdx] >> signXo2; // Unmasked.
                            if (s & 1) v *= p.slope;
                            if (s & 2) v = 0.f;
                        }
                    }

                    s_tileUpXY[idx] = v;
                }
            }
        }
        else if (filterMode == MODE_FUSD || filterMode == MODE_FUFD)
        {
            // Full upsampling filter.
            if (up == 4)
            {
                // 4-wide.
                __syncthreads();
                int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH + p.sOfs.y : 0; // Skip already written signs.
                for (int idx = threadIdx.x*up; idx < tileUpW * tileUpH; idx += blockDim.x*up)
                {
                    int relUpX0, relUpY0;
                    fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                    int relInX0 = CEIL_DIV(relUpX0 - phaseInX, up);
                    int relInY0 = CEIL_DIV(relUpY0 - phaseInY, up);
                    int src0 = relInX0 + tileInW * relInY0;
                    int tap0y = (relInY0 * up + phaseInY - relUpY0);

                    #define X_LOOP(TAPY, PX) \
                        for (int sx = 0; sx < fuSize / up; sx++) \
                        { \
                            v.x += x * (scalar_t)c_fu[(sx * up + (((PX) - 0) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; if ((PX) == 0) x = s_tileIn[src0 + 1 + sx + sy * tileInW]; \
                            v.y += x * (scalar_t)c_fu[(sx * up + (((PX) - 1) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; if ((PX) == 1) x = s_tileIn[src0 + 1 + sx + sy * tileInW]; \
                            v.z += x * (scalar_t)c_fu[(sx * up + (((PX) - 2) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; if ((PX) == 2) x = s_tileIn[src0 + 1 + sx + sy * tileInW]; \
                            v.w += x * (scalar_t)c_fu[(sx * up + (((PX) - 3) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; if ((PX) == 3) x = s_tileIn[src0 + 1 + sx + sy * tileInW]; \
                        }

                    vec4_t v = InternalType<T>::zero_vec4();
                    if (tap0y == 0 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(0, 0) }
                    if (tap0y == 0 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(0, 1) }
                    if (tap0y == 0 && phaseInX == 2)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(0, 2) }
                    if (tap0y == 0 && phaseInX == 3)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(0, 3) }
                    if (tap0y == 1 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(1, 0) }
                    if (tap0y == 1 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(1, 1) }
                    if (tap0y == 1 && phaseInX == 2)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(1, 2) }
                    if (tap0y == 1 && phaseInX == 3)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(1, 3) }
                    if (tap0y == 2 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(2, 0) }
                    if (tap0y == 2 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(2, 1) }
                    if (tap0y == 2 && phaseInX == 2)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(2, 2) }
                    if (tap0y == 2 && phaseInX == 3)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(2, 3) }
                    if (tap0y == 3 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(3, 0) }
                    if (tap0y == 3 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(3, 1) }
                    if (tap0y == 3 && phaseInX == 2)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(3, 2) }
                    if (tap0y == 3 && phaseInX == 3)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW];
                            #pragma unroll
                            X_LOOP(3, 3) }

                    #undef X_LOOP

                    int signX = tileOutX * down + relUpX0 + p.sOfs.x;
                    int signY = tileOutY * down + relUpY0 + p.sOfs.y;
                    int signXb = (signX + p.sByteOfs) >> 2;
                    int signXo2 = ((signX + p.sByteOfs) & 3) << 1;
                    int signIdx = signXb * p.sStride.x + signY * p.sStride.y + channelIdx * p.sStride.z + batchIdx * p.sStride.w;
                    v.x *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y *= (scalar_t)((float)up * (float)up * p.gain);
                    v.z *= (scalar_t)((float)up * (float)up * p.gain);
                    v.w *= (scalar_t)((float)up * (float)up * p.gain);

                    if (writeSigns) // Determine and write sign.
                    {
                        int sx = __float_as_uint(v.x) >> 31;
                        int sy = __float_as_uint(v.y) >> 31;
                        int sz = __float_as_uint(v.z) >> 31;
                        int sw = __float_as_uint(v.w) >> 31;
                        if (sx) v.x *= p.slope; if (fabsf(v.x) > p.clamp) { sx = 2; v.x = copysignf(p.clamp, v.x); }
                        if (sy) v.y *= p.slope; if (fabsf(v.y) > p.clamp) { sy = 2; v.y = copysignf(p.clamp, v.y); }
                        if (sz) v.z *= p.slope; if (fabsf(v.z) > p.clamp) { sz = 2; v.z = copysignf(p.clamp, v.z); }
                        if (sw) v.w *= p.slope; if (fabsf(v.w) > p.clamp) { sw = 2; v.w = copysignf(p.clamp, v.w); }

                        // Write signs.
                        bool signValidWrite = ((unsigned int)signXb < (unsigned int)p.sShape.x & signY >= minY & (unsigned int)signY < (unsigned int)p.sShape.y);
                        if (signValidWrite)
                        {
                            int ss = sx + (sy << 2) + (sz << 4) + (sw << 6);
                            p.so[signIdx] = ss;
                        }
                    }
                    else // Read sign and apply.
                    {
                        if ((unsigned int)signY < (unsigned int)p.sShape.y)
                        {
                            unsigned int s = 0;
                            if ((unsigned int)signXb < (unsigned int)p.sShape.x) s = p.si[signIdx];
                            if ((unsigned int)(signXb + 1) < (unsigned int)p.sShape.x) s += p.si[signIdx + 1] << 8;
                            s >>= signXo2;
                            if ((unsigned int)signX + 0 < (unsigned int)p.sWidth) { if (s & 0x01) v.x *= p.slope; if (s & 0x02) v.x = 0.f; }
                            if ((unsigned int)signX + 1 < (unsigned int)p.sWidth) { if (s & 0x04) v.y *= p.slope; if (s & 0x08) v.y = 0.f; }
                            if ((unsigned int)signX + 2 < (unsigned int)p.sWidth) { if (s & 0x10) v.z *= p.slope; if (s & 0x20) v.z = 0.f; }
                            if ((unsigned int)signX + 3 < (unsigned int)p.sWidth) { if (s & 0x40) v.w *= p.slope; if (s & 0x80) v.w = 0.f; }
                        }
                    }

                    s_tileUpXY[idx+0] = v.x;
                    s_tileUpXY[idx+1] = v.y;
                    s_tileUpXY[idx+2] = v.z;
                    s_tileUpXY[idx+3] = v.w;
                }
            }
            else if (up == 2)
            {
                // 2 x 2-wide.
                __syncthreads();
                int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH + p.sOfs.y : 0; // Skip already written signs.
                for (int idx = threadIdx.x * 4; idx < tileUpW * tileUpH; idx += blockDim.x * 4)
                {
                    int relUpX0, relUpY0;
                    fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                    int relInX0 = CEIL_DIV(relUpX0 - phaseInX, up);
                    int relInY0 = CEIL_DIV(relUpY0 - phaseInY, up);
                    int src0 = relInX0 + tileInW * relInY0;
                    int tap0y = (relInY0 * up + phaseInY - relUpY0);

                    #define X_LOOP(TAPY, PX) \
                        for (int sx = 0; sx < fuSize / up; sx++) \
                        { \
                            v.x += x * (scalar_t)c_fu[(sx * up + (((PX) - 0) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; \
                            v.z += y * (scalar_t)c_fu[(sx * up + (((PX) - 0) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; if ((PX) == 0) { x = y; y = s_tileIn[src0 + 2 + sx + sy * tileInW]; } \
                            v.y += x * (scalar_t)c_fu[(sx * up + (((PX) - 1) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; \
                            v.w += y * (scalar_t)c_fu[(sx * up + (((PX) - 1) & (up - 1))) + (sy * up + (TAPY)) * MAX_FILTER_SIZE]; if ((PX) == 1) { x = y; y = s_tileIn[src0 + 2 + sx + sy * tileInW]; } \
                        }

                    vec4_t v = InternalType<T>::zero_vec4();
                    if (tap0y == 0 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW]; scalar_t y = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(0, 0) }
                    if (tap0y == 0 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW]; scalar_t y = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(0, 1) }
                    if (tap0y == 1 && phaseInX == 0)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW]; scalar_t y = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(1, 0) }
                    if (tap0y == 1 && phaseInX == 1)
                        #pragma unroll
                        for (int sy = 0; sy < fuSize / up; sy++) { scalar_t x = s_tileIn[src0 + sy * tileInW]; scalar_t y = s_tileIn[src0 + sy * tileInW + 1];
                            #pragma unroll
                            X_LOOP(1, 1) }

                    #undef X_LOOP

                    int signX = tileOutX * down + relUpX0 + p.sOfs.x;
                    int signY = tileOutY * down + relUpY0 + p.sOfs.y;
                    int signXb = (signX + p.sByteOfs) >> 2;
                    int signXo2 = ((signX + p.sByteOfs) & 3) << 1;
                    int signIdx = signXb * p.sStride.x + signY * p.sStride.y + channelIdx * p.sStride.z + batchIdx * p.sStride.w;
                    v.x *= (scalar_t)((float)up * (float)up * p.gain);
                    v.y *= (scalar_t)((float)up * (float)up * p.gain);
                    v.z *= (scalar_t)((float)up * (float)up * p.gain);
                    v.w *= (scalar_t)((float)up * (float)up * p.gain);

                    if (writeSigns) // Determine and write sign.
                    {
                        int sx = __float_as_uint(v.x) >> 31;
                        int sy = __float_as_uint(v.y) >> 31;
                        int sz = __float_as_uint(v.z) >> 31;
                        int sw = __float_as_uint(v.w) >> 31;
                        if (sx) v.x *= p.slope; if (fabsf(v.x) > p.clamp) { sx = 2; v.x = copysignf(p.clamp, v.x); }
                        if (sy) v.y *= p.slope; if (fabsf(v.y) > p.clamp) { sy = 2; v.y = copysignf(p.clamp, v.y); }
                        if (sz) v.z *= p.slope; if (fabsf(v.z) > p.clamp) { sz = 2; v.z = copysignf(p.clamp, v.z); }
                        if (sw) v.w *= p.slope; if (fabsf(v.w) > p.clamp) { sw = 2; v.w = copysignf(p.clamp, v.w); }

                        // Write signs.
                        bool signValidWrite = ((unsigned int)signXb < (unsigned int)p.sShape.x & signY >= minY & (unsigned int)signY < (unsigned int)p.sShape.y);
                        if (signValidWrite)
                        {
                            int ss = sx + (sy << 2) + (sz << 4) + (sw << 6);
                            p.so[signIdx] = ss;
                        }
                    }
                    else // Read sign and apply.
                    {
                        if ((unsigned int)signY < (unsigned int)p.sShape.y)
                        {
                            unsigned int s = 0;
                            if ((unsigned int)signXb < (unsigned int)p.sShape.x) s = p.si[signIdx];
                            if ((unsigned int)(signXb + 1) < (unsigned int)p.sShape.x) s += p.si[signIdx + 1] << 8;
                            s >>= signXo2;
                            if ((unsigned int)signX + 0 < (unsigned int)p.sWidth) { if (s & 0x01) v.x *= p.slope; if (s & 0x02) v.x = 0.f; }
                            if ((unsigned int)signX + 1 < (unsigned int)p.sWidth) { if (s & 0x04) v.y *= p.slope; if (s & 0x08) v.y = 0.f; }
                            if ((unsigned int)signX + 2 < (unsigned int)p.sWidth) { if (s & 0x10) v.z *= p.slope; if (s & 0x20) v.z = 0.f; }
                            if ((unsigned int)signX + 3 < (unsigned int)p.sWidth) { if (s & 0x40) v.w *= p.slope; if (s & 0x80) v.w = 0.f; }
                        }
                    }

                    s_tileUpXY[idx+0] = v.x;
                    s_tileUpXY[idx+1] = v.y;
                    s_tileUpXY[idx+2] = v.z;
                    s_tileUpXY[idx+3] = v.w;
                }
            }
            else
            {
                // Original.
                __syncthreads();
                int groupMask = 15 << ((threadIdx.x & 31) & ~3);
                int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH + p.sOfs.y : 0; // Skip already written signs.
                for (int idx = threadIdx.x; idx < tileUpW * tileUpH; idx += blockDim.x)
                {
                    int relUpX0, relUpY0;
                    fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                    int relInX0 = CEIL_DIV(relUpX0 - phaseInX, up);
                    int relInY0 = CEIL_DIV(relUpY0 - phaseInY, up);
                    int src0 = relInX0 + tileInW * relInY0;
                    int tap0 = (relInX0 * up + phaseInX - relUpX0) + MAX_FILTER_SIZE * (relInY0 * up + phaseInY - relUpY0);
                    scalar_t v = 0;
                    #pragma unroll
                    for (int sy = 0; sy < fuSize / up; sy++)
                    #pragma unroll
                    for (int sx = 0; sx < fuSize / up; sx++)
                        v += s_tileIn[src0 + sx + sy * tileInW] * (scalar_t)c_fu[tap0 + sx * up + sy * (up * MAX_FILTER_SIZE)];

                    int signX = tileOutX * down + relUpX0 + p.sOfs.x;
                    int signY = tileOutY * down + relUpY0 + p.sOfs.y;
                    int signXb = (signX + p.sByteOfs) >> 2;
                    int signXo2 = ((signX + p.sByteOfs) & 3) << 1;
                    int signIdx = signXb * p.sStride.x + signY * p.sStride.y + channelIdx * p.sStride.z + batchIdx * p.sStride.w;
                    v *= (scalar_t)((float)up * (float)up * p.gain);

                    if (writeSigns) // Determine and write sign.
                    {
                        int s = __float_as_uint(v) >> 31; // Bit 0 = sign.
                        if (s) v *= p.slope;
                        if (fabsf(v) > p.clamp)
                        {
                            s = 2; // Bit 1 = clamp.
                            v = copysignf(p.clamp, v);
                        }

                        // Write signs.
                        bool signValidWrite = ((unsigned int)signXb < (unsigned int)p.sShape.x & signY >= minY & (unsigned int)signY < (unsigned int)p.sShape.y);
                        if (signValidWrite)
                        {
                            s <<= signXo2;
                            s |= __shfl_xor_sync(groupMask, s, 1);
                            s |= __shfl_xor_sync(groupMask, s, 2);
                            p.so[signIdx] = s;
                        }
                    }
                    else // Read sign and apply.
                    {
                        bool signValidRead = ((unsigned int)signX < (unsigned int)p.sWidth & (unsigned int)signY < (unsigned int)p.sShape.y);
                        if (signValidRead)
                        {
                            int s = p.si[signIdx] >> signXo2; // Unmasked.
                            if (s & 1) v *= p.slope;
                            if (s & 2) v = 0.f;
                        }
                    }

                    s_tileUpXY[idx] = v;
                }
            }
        }
        else if (filterMode == MODE_SKIP)
        {
            // Just read/write signs.
            __syncthreads();
            int minY = tileOutY ? (tileOutY - tileOutH) * down + tileUpH + p.sOfs.y : 0; // Skip already written signs.
            for (int idx = threadIdx.x * 4; idx < tileUpW * tileUpH; idx += blockDim.x * 4)
            {
                int relUpX0, relUpY0;
                fast_div_mod<tileUpW>(relUpX0, relUpY0, idx);
                int signX = tileOutX * down + relUpX0 + p.sOfs.x;
                int signY = tileOutY * down + relUpY0 + p.sOfs.y;
                int signXb = (signX + p.sByteOfs) >> 2;
                int signIdx = signXb * p.sStride.x + signY * p.sStride.y + channelIdx * p.sStride.z + batchIdx * p.sStride.w;

                if (writeSigns) // Determine and write sign.
                {
                    bool signValidWrite = ((unsigned int)signXb < (unsigned int)p.sShape.x & signY >= minY & (unsigned int)signY < (unsigned int)p.sShape.y);
                    if (signValidWrite)
                        p.so[signIdx] = 0xaa;
                }
                else // Read sign and apply.
                {
                    if ((unsigned int)signY < (unsigned int)p.sShape.y)
                    {
                        unsigned int s = 0;
                        if ((unsigned int)signXb < (unsigned int)p.sShape.x) s = p.si[signIdx];
                        if ((unsigned int)(signXb + 1) < (unsigned int)p.sShape.x) s += p.si[signIdx + 1] << 8;
                        *((unsigned int*)&s_tileUpXY[idx]) = s;
                    }
                }
            }
        }

        // Downsampling.
        if (filterMode == MODE_SUSD || filterMode == MODE_FUSD)
        {
            // Horizontal downsampling.
            __syncthreads();
            if (down == 4)
            {
                // Calculate 4 pixels at a time.
                for (int idx = threadIdx.x * 4; idx < tileOutW * tileUpH; idx += blockDim.x * 4)
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src0 = relUpY * tileUpW + relUpX0;
                    vec4_t v = InternalType<T>::zero_vec4();
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                    {
                        v.x += s_tileUpXY[src0 +  0 + step] * (scalar_t)c_fd[step];
                        v.y += s_tileUpXY[src0 +  4 + step] * (scalar_t)c_fd[step];
                        v.z += s_tileUpXY[src0 +  8 + step] * (scalar_t)c_fd[step];
                        v.w += s_tileUpXY[src0 + 12 + step] * (scalar_t)c_fd[step];
                    }
                    s_tileDownX[idx+0] = v.x;
                    s_tileDownX[idx+1] = v.y;
                    s_tileDownX[idx+2] = v.z;
                    s_tileDownX[idx+3] = v.w;
                }
            }
            else if (down == 2)
            {
                // Calculate 2 pixels at a time.
                for (int idx = threadIdx.x * 2; idx < tileOutW * tileUpH; idx += blockDim.x * 2)
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src0 = relUpY * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                    {
                        v.x += s_tileUpXY[src0 + 0 + step] * (scalar_t)c_fd[step];
                        v.y += s_tileUpXY[src0 + 2 + step] * (scalar_t)c_fd[step];
                    }
                    s_tileDownX[idx+0] = v.x;
                    s_tileDownX[idx+1] = v.y;
                }
            }
            else
            {
                // Basic case.
                for (int idx = threadIdx.x; idx < tileOutW * tileUpH; idx += blockDim.x)
                {
                    int relOutX0, relUpY;
                    fast_div_mod<tileOutW>(relOutX0, relUpY, idx);
                    int relUpX0 = relOutX0 * down;
                    int src0 = relUpY * tileUpW + relUpX0;
                    scalar_t v = 0;
                    #pragma unroll
                    for (int step = 0; step < fdSize; step++)
                        v += s_tileUpXY[src0 + step] * (scalar_t)c_fd[step];
                    s_tileDownX[idx] = v;
                }
            }

            // Vertical downsampling & store output tile.
            __syncthreads();
            int mapOfsOut = channelIdx * p.yStride.z + batchIdx * p.yStride.w;
            for (int idx = threadIdx.x; idx < tileOutW * tileOutH; idx += blockDim.x)
            {
                int relOutX, relOutY0;
                fast_div_mod<tileOutW>(relOutX, relOutY0, idx);
                int relUpY0 = relOutY0 * down;
                int src0 = relUpY0 * tileOutW + relOutX;
                scalar_t v = 0;
                #pragma unroll
                for (int step = 0; step < fdSize; step++)
                    v += s_tileDownX[src0 + step * tileOutW] * (scalar_t)c_fd[step];

                int outX = tileOutX + relOutX;
                int outY = tileOutY + relOutY0;

                if (outX < p.yShape.x & outY < p.yShape.y)
                    *((T*)((char*)p.y + (outX * p.yStride.x + outY * p.yStride.y + mapOfsOut))) = (T)v;
            }
        }
        else if (filterMode == MODE_SUFD || filterMode == MODE_FUFD)
        {
            // Full downsampling filter.
            if (down == 2)
            {
                // 2 x 2-wide.
                __syncthreads();
                int mapOfsOut = channelIdx * p.yStride.z + batchIdx * p.yStride.w;
                for (int idx = threadIdx.x * 2; idx < tileOutW * tileOutH; idx += blockDim.x * 2)
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    int relUpX0 = relOutX0 * down;
                    int relUpY0 = relOutY0 * down;
                    int src0 = relUpY0 * tileUpW + relUpX0;
                    vec2_t v = InternalType<T>::zero_vec2();
                    #pragma unroll
                    for (int sy = 0; sy < fdSize; sy++)
                    #pragma unroll
                    for (int sx = 0; sx < fdSize; sx++)
                    {
                        v.x += s_tileUpXY[src0 + 0 + sx + sy * tileUpW] * (scalar_t)c_fd[sx + sy * MAX_FILTER_SIZE];
                        v.y += s_tileUpXY[src0 + 2 + sx + sy * tileUpW] * (scalar_t)c_fd[sx + sy * MAX_FILTER_SIZE];
                    }

                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if (outY < p.yShape.y)
                    {
                        if (outX + 0 < p.yShape.x) *((T*)((char*)p.y + ((outX + 0) * p.yStride.x + outY * p.yStride.y + mapOfsOut))) = (T)v.x;
                        if (outX + 1 < p.yShape.x) *((T*)((char*)p.y + ((outX + 1) * p.yStride.x + outY * p.yStride.y + mapOfsOut))) = (T)v.y;
                    }
                }
            }
            else
            {
                // Original. Good for down=4 too.
                __syncthreads();
                int mapOfsOut = channelIdx * p.yStride.z + batchIdx * p.yStride.w;
                for (int idx = threadIdx.x; idx < tileOutW * tileOutH; idx += blockDim.x)
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    int relUpX0 = relOutX0 * down;
                    int relUpY0 = relOutY0 * down;
                    int src0 = relUpY0 * tileUpW + relUpX0;
                    scalar_t v = 0;
                    #pragma unroll
                    for (int sy = 0; sy < fdSize; sy++)
                    #pragma unroll
                    for (int sx = 0; sx < fdSize; sx++)
                        v += s_tileUpXY[src0 + sx + sy * tileUpW] * (scalar_t)c_fd[sx + sy * MAX_FILTER_SIZE];

                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if (outX < p.yShape.x & outY < p.yShape.y)
                        *((T*)((char*)p.y + (outX * p.yStride.x + outY * p.yStride.y + mapOfsOut))) = (T)v;
                }
            }
        }
        else if (filterMode == MODE_SKIP)
        {
            // Skip computation, just write in same order as normal code.
            if (down == 2)
            {
                // 2 x 2-wide.
                __syncthreads();
                int mapOfsOut = channelIdx * p.yStride.z + batchIdx * p.yStride.w;
                for (int idx = threadIdx.x * 2; idx < tileOutW * tileOutH; idx += blockDim.x * 2)
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    vec2_t v = InternalType<T>::zero_vec2();
                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if (outY < p.yShape.y)
                    {
                        if (outX + 0 < p.yShape.x) *((T*)((char*)p.y + ((outX + 0) * p.yStride.x + outY * p.yStride.y + mapOfsOut))) = (T)v.x;
                        if (outX + 1 < p.yShape.x) *((T*)((char*)p.y + ((outX + 1) * p.yStride.x + outY * p.yStride.y + mapOfsOut))) = (T)v.y;
                    }
                }
            }
            else
            {
                // 1-wide / 4-wide.
                __syncthreads();
                int mapOfsOut = channelIdx * p.yStride.z + batchIdx * p.yStride.w;
                for (int idx = threadIdx.x; idx < tileOutW * tileOutH; idx += blockDim.x)
                {
                    int relOutX0, relOutY0;
                    fast_div_mod<tileOutW>(relOutX0, relOutY0, idx);
                    scalar_t v = 0;
                    int outX = tileOutX + relOutX0;
                    int outY = tileOutY + relOutY0;
                    if (outX < p.yShape.x & outY < p.yShape.y)
                        *((T*)((char*)p.y + (outX * p.yStride.x + outY * p.yStride.y + mapOfsOut))) = (T)v;
                }
            }
        }

        if (!enableXrep)
            break;
    }
}

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T, bool signWrite> filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel(const filtered_lrelu_kernel_params& p)
{
    const bool dbl = (sizeof(T) == 8);

    filtered_lrelu_kernel_spec s;
    s.setup     = (void*)setup_filters_kernel;
    s.exec      = NULL;
    s.tileOut   = make_int2(1, 1);
    s.numWarps  = 1;
    s.sharedKB  = 48;
    s.xrep      = 0;

    // Filter depth must be 1.
    if (p.fuShape.z != 1 || p.fdShape.z != 1)
        return s;

    // Return first matching kernel.
    // Note: High shared memory usage kernels must be before low usage so they will ever be selected.
    // Note: Small kernels must be before large kernels, otherwise large will always be selected.
    // Note: For bit-packed sign tensor to work, D*TW must be divisible by 4.
    #define CASE_V(V, SH, U, FU, D, FD, MODE, TW, TH, W, XR) \
    if ((p.fuShape.y == 1 && (MODE == MODE_SUSD || MODE == MODE_SUFD)) || (p.fuShape.y == p.fuShape.x && (MODE == MODE_FUSD || MODE == MODE_FUFD))) \
    if ((p.fdShape.y == 1 && (MODE == MODE_SUSD || MODE == MODE_FUSD)) || (p.fdShape.y == p.fdShape.x && (MODE == MODE_SUFD || MODE == MODE_FUFD))) \
    if (p.variant == V && p.sharedKB >= SH && p.up == U && p.fuShape.x <= FU && p.fuShape.y <= FU && p.down == D && p.fdShape.x <= FD && p.fdShape.y <= FD) \
    { \
        /*printf("mode=%d, up=%d, down=%d, fu=[%d,%d], fd=[%d,%d], sign_write=%d, tile=[%d,%d], w=%d, xr=%d\n", \
            MODE, U, D, p.fuShape.x, p.fuShape.y, p.fdShape.x, p.fdShape.y, signWrite?1:0, TW, TH, W, XR);*/ \
        s.exec = (void*)filtered_lrelu_kernel<T, SH, signWrite ? SIGN_WRITE : SIGN_READ, MODE, U, FU, D, FD, dbl?4:TW, dbl?1:TH, (dbl?4:W)*32, !!XR>; \
        s.tileOut = make_int2(dbl?4:TW, dbl?1:TH); \
        s.numWarps = dbl?4:W; \
        s.sharedKB = SH; \
        s.xrep = XR; \
        return s; \
    }
    #define CASE(SH, U, FU, D, FD, MODE, TW, TH, W, XR) CASE_V(0, SH, U, FU, D, FD, MODE, TW, TH, W, XR) CASE_V(0, SH, U, FU, D, FD, MODE, TW, TH, W, XR)

    // Separable filters (SUSD). ------------------------------------------------------------------

    // 1-tap kernel.
    CASE(48, /*up*/1,1,  /*down*/1,1,  /*mode*/MODE_FUFD, /*tile*/64,94, /*warps*/32, /*xrep*/ 0) // 1t-ups1-downs1 = 1t-upf1-downf1

    // 4-tap kernels.
    CASE(96, /*up*/2,8,  /*down*/2,8,  /*mode*/MODE_SUSD, /*tile*/64,54, /*warps*/28, /*xrep*/ 9) // 4t-ups2-downs2
    CASE(96, /*up*/4,16, /*down*/2,8,  /*mode*/MODE_SUSD, /*tile*/96,38, /*warps*/28, /*xrep*/ 6) // 4t-ups4-downs2
    CASE(96, /*up*/2,8,  /*down*/4,16, /*mode*/MODE_SUSD, /*tile*/24,33, /*warps*/28, /*xrep*/ 8) // 4t-ups2-downs4

    // 6-tap kernels.
    CASE(48, /*up*/2,12, /*down*/1,1,  /*mode*/MODE_SUFD, /*tile*/64,122,/*warps*/16, /*xrep*/ 0) // 6t-ups2-downs1 = 6t-ups2-downf1
    CASE(48, /*up*/1,1,  /*down*/2,12, /*mode*/MODE_FUSD, /*tile*/32,35, /*warps*/16, /*xrep*/ 0) // 6t-ups1-downs2 = 6t-upf1-downs2
    CASE(96, /*up*/2,12, /*down*/2,12, /*mode*/MODE_SUSD, /*tile*/64,50, /*warps*/28, /*xrep*/10) // 6t-ups2-downs2
    CASE(96, /*up*/4,24, /*down*/2,12, /*mode*/MODE_SUSD, /*tile*/64,55, /*warps*/28, /*xrep*/ 4) // 6t-ups4-downs2
    CASE(96, /*up*/2,12, /*down*/4,24, /*mode*/MODE_SUSD, /*tile*/16,41, /*warps*/28, /*xrep*/ 5) // 6t-ups2-downs4
    CASE(48, /*up*/4,24, /*down*/4,24, /*mode*/MODE_SUSD, /*tile*/16,23, /*warps*/16, /*xrep*/ 8) // 6t-ups4-downs4

    // 7-tap kernels.
    CASE(96, /*up*/2,14, /*down*/2,14, /*mode*/MODE_SUSD, /*tile*/64,50, /*warps*/28, /*xrep*/10) // 7t-ups2-downs2
    CASE(96, /*up*/4,28, /*down*/2,14, /*mode*/MODE_SUSD, /*tile*/64,54, /*warps*/28, /*xrep*/ 4) // 7t-ups4-downs2
    CASE(96, /*up*/2,14, /*down*/4,28, /*mode*/MODE_SUSD, /*tile*/16,38, /*warps*/28, /*xrep*/ 6) // 7t-ups2-downs4

    // 8-tap kernels.
    CASE(96, /*up*/2,16, /*down*/2,16, /*mode*/MODE_SUSD, /*tile*/64,47, /*warps*/28, /*xrep*/10) // 8t-ups2-downs2
  //CASE(96, /*up*/4,32, /*down*/2,16, /*mode*/MODE_SUSD, /*tile*/64,52, /*warps*/28, /*xrep*/ 4) // 8t-ups4-downs2 (slower than two 48kB CTAs)
    CASE(96, /*up*/2,16, /*down*/4,32, /*mode*/MODE_SUSD, /*tile*/16,35, /*warps*/28, /*xrep*/ 5) // 8t-ups2-downs4
    CASE(0,  /*up*/2,16, /*down*/2,16, /*mode*/MODE_SUSD, /*tile*/32,41, /*warps*/24, /*xrep*/ 7) // 8t-ups2-downs2 (common 48kB kernels for 6, 7, 8)
    CASE(0,  /*up*/4,32, /*down*/2,16, /*mode*/MODE_SUSD, /*tile*/32,47, /*warps*/24, /*xrep*/10) // 8t-ups4-downs2
    CASE(0,  /*up*/2,16, /*down*/4,32, /*mode*/MODE_SUSD, /*tile*/16,13, /*warps*/20, /*xrep*/18) // 8t-ups2-downs4

    // 12-tap kernels.
    CASE(96, /*up*/2,24, /*down*/2,24, /*mode*/MODE_SUSD, /*tile*/64,38, /*warps*/24, /*xrep*/ 9) // 12t-ups2-downs2
    CASE(96, /*up*/4,48, /*down*/2,24, /*mode*/MODE_SUSD, /*tile*/64,45, /*warps*/24, /*xrep*/ 6) // 12t-ups4-downs2
    CASE(96, /*up*/2,24, /*down*/4,48, /*mode*/MODE_SUSD, /*tile*/16,24, /*warps*/24, /*xrep*/ 7) // 12t-ups2-downs4
    CASE(0,  /*up*/2,24, /*down*/2,24, /*mode*/MODE_SUSD, /*tile*/32,31, /*warps*/24, /*xrep*/ 9) // 12t-ups2-downs2
    CASE(0,  /*up*/4,48, /*down*/2,24, /*mode*/MODE_SUSD, /*tile*/32,38, /*warps*/20, /*xrep*/ 7) // 12t-ups4-downs2
    CASE(0,  /*up*/2,24, /*down*/4,48, /*mode*/MODE_SUSD, /*tile*/8,13,  /*warps*/20, /*xrep*/12) // 12t-ups2-downs4

    // 16-tap kernels.
    CASE(96, /*up*/2,32, /*down*/2,32, /*mode*/MODE_SUSD, /*tile*/32,64, /*warps*/24, /*xrep*/ 3) // 16t-ups2-downs2
    CASE(96, /*up*/4,64, /*down*/2,32, /*mode*/MODE_SUSD, /*tile*/48,54, /*warps*/28, /*xrep*/ 0) // 16t-ups4-downs2
    CASE(96, /*up*/2,32, /*down*/4,64, /*mode*/MODE_SUSD, /*tile*/16,15, /*warps*/24, /*xrep*/ 8) // 16t-ups2-downs4

    // Full (non-separable) filters (FUFD). -------------------------------------------------------

    // 6-tap kernels.
    CASE(96, /*up*/2,12, /*down*/2,12, /*mode*/MODE_FUFD, /*tile*/96,41, /*warps*/32, /*xrep*/11) // 6t-upf2-downf2
    CASE(96, /*up*/4,24, /*down*/2,12, /*mode*/MODE_FUFD, /*tile*/56,86, /*warps*/31, /*xrep*/ 0) // 6t-upf4-downf2
    CASE(96, /*up*/2,12, /*down*/4,24, /*mode*/MODE_FUFD, /*tile*/32,27, /*warps*/32, /*xrep*/ 0) // 6t-upf2-downf4
    CASE(0,  /*up*/2,12, /*down*/2,12, /*mode*/MODE_FUFD, /*tile*/64,28, /*warps*/20, /*xrep*/ 8) // 6t-upf2-downf2
    CASE(0,  /*up*/4,24, /*down*/2,12, /*mode*/MODE_FUFD, /*tile*/56,39, /*warps*/31, /*xrep*/ 0) // 6t-upf4-downf2
    CASE(0,  /*up*/2,12, /*down*/4,24, /*mode*/MODE_FUFD, /*tile*/16,22, /*warps*/24, /*xrep*/ 7) // 6t-upf2-downf4

    // 8-tap kernels.
    CASE(96, /*up*/2,16, /*down*/2,16, /*mode*/MODE_FUFD, /*tile*/56,64, /*warps*/24, /*xrep*/ 0) // 8t-upf2-downf2
    CASE(96, /*up*/4,32, /*down*/2,16, /*mode*/MODE_FUFD, /*tile*/56,79, /*warps*/31, /*xrep*/ 0) // 8t-upf4-downf2
    CASE(96, /*up*/2,16, /*down*/4,32, /*mode*/MODE_FUFD, /*tile*/32,23, /*warps*/32, /*xrep*/ 0) // 8t-upf2-downf4
    CASE(0,  /*up*/2,16, /*down*/2,16, /*mode*/MODE_FUFD, /*tile*/40,38, /*warps*/24, /*xrep*/ 7) // 8t-upf2-downf2
    CASE(0,  /*up*/4,32, /*down*/2,16, /*mode*/MODE_FUFD, /*tile*/40,50, /*warps*/32, /*xrep*/ 0) // 8t-upf4-downf2
    CASE(0,  /*up*/2,16, /*down*/4,32, /*mode*/MODE_FUFD, /*tile*/16,16, /*warps*/32, /*xrep*/ 3) // 8t-upf2-downf4

    // Full up, separable down (FUSD). ------------------------------------------------------------

    // 4-tap kernels.
    CASE(48, /*up*/2,8,  /*down*/2,8,  /*mode*/MODE_FUSD, /*tile*/96,17, /*warps*/16, /*xrep*/ 0) // 4t-upf2-downs2
    CASE(48, /*up*/2,8,  /*down*/4,16, /*mode*/MODE_FUSD, /*tile*/16,27, /*warps*/20, /*xrep*/ 0) // 4t-upf2-downs4

    // 6-tap kernels.
    CASE(96, /*up*/2,12, /*down*/2,12, /*mode*/MODE_FUSD, /*tile*/96,35, /*warps*/28, /*xrep*/ 0) // 6t-upf2-downs2
    CASE(96, /*up*/4,24, /*down*/2,12, /*mode*/MODE_FUSD, /*tile*/56,59, /*warps*/31, /*xrep*/ 0) // 6t-upf4-downs2
    CASE(96, /*up*/2,12, /*down*/4,24, /*mode*/MODE_FUSD, /*tile*/48,16, /*warps*/28, /*xrep*/ 0) // 6t-upf2-downs4
    CASE(96, /*up*/4,24, /*down*/4,24, /*mode*/MODE_FUSD, /*tile*/28,33, /*warps*/32, /*xrep*/ 4) // 6t-upf4-downs4
    CASE(0,  /*up*/2,12, /*down*/2,12, /*mode*/MODE_FUSD, /*tile*/64,24, /*warps*/16, /*xrep*/ 0) // 6t-upf2-downs2
    CASE(0,  /*up*/4,24, /*down*/2,12, /*mode*/MODE_FUSD, /*tile*/56,27, /*warps*/31, /*xrep*/ 5) // 6t-upf4-downs2
    CASE(0,  /*up*/2,12, /*down*/4,24, /*mode*/MODE_FUSD, /*tile*/16,22, /*warps*/24, /*xrep*/ 5) // 6t-upf2-downs4

    // 8-tap kernels.
    CASE(0,  /*up*/2,16, /*down*/2,16, /*mode*/MODE_FUSD, /*tile*/32,47, /*warps*/32, /*xrep*/ 0) // 8t-upf2-downs2 // UNOPTIMIZED
    CASE(0,  /*up*/4,32, /*down*/2,16, /*mode*/MODE_FUSD, /*tile*/56,25, /*warps*/31, /*xrep*/ 0) // 8t-upf4-downs2 // UNOPTIMIZED
    CASE(0,  /*up*/2,16, /*down*/4,32, /*mode*/MODE_FUSD, /*tile*/16,17, /*warps*/24, /*xrep*/ 0) // 8t-upf2-downs4 // UNOPTIMIZED

    // Separable up, full down (SUFD). ------------------------------------------------------------

    // 4-tap kernels.
    CASE(48, /*up*/2,8,  /*down*/2,8,  /*mode*/MODE_SUFD, /*tile*/64,25, /*warps*/16, /*xrep*/ 0) // 4t-ups2-downf2
    CASE(48, /*up*/4,16, /*down*/2,8,  /*mode*/MODE_SUFD, /*tile*/32,62, /*warps*/16, /*xrep*/ 0) // 4t-ups4-downf2

    // 6-tap kernels.
    CASE(96, /*up*/2,12, /*down*/2,12, /*mode*/MODE_SUFD, /*tile*/64,50, /*warps*/31, /*xrep*/ 0) // 6t-ups2-downf2
    CASE(96, /*up*/2,12, /*down*/4,24, /*mode*/MODE_SUFD, /*tile*/28,25, /*warps*/32, /*xrep*/ 0) // 6t-ups2-downf4
    CASE(96, /*up*/4,24, /*down*/2,12, /*mode*/MODE_SUFD, /*tile*/64,59, /*warps*/24, /*xrep*/ 0) // 6t-ups4-downf2
    CASE(48, /*up*/4,24, /*down*/4,24, /*mode*/MODE_SUFD, /*tile*/16,23, /*warps*/16, /*xrep*/ 1) // 6t-ups4-downf4
    CASE(0,  /*up*/2,12, /*down*/2,12, /*mode*/MODE_SUFD, /*tile*/48,30, /*warps*/24, /*xrep*/ 0) // 6t-ups2-downf2
    CASE(0,  /*up*/2,12, /*down*/4,24, /*mode*/MODE_SUFD, /*tile*/12,24, /*warps*/27, /*xrep*/ 0) // 6t-ups2-downf4
    CASE(0,  /*up*/4,24, /*down*/2,12, /*mode*/MODE_SUFD, /*tile*/64,27, /*warps*/32, /*xrep*/ 3) // 6t-ups4-downf2

    // 8-tap kernels.
    CASE(0,  /*up*/2,16, /*down*/2,16, /*mode*/MODE_SUFD, /*tile*/32,41, /*warps*/32, /*xrep*/ 0) // 8t-ups2-downf2 // UNOPTIMIZED
    CASE(0,  /*up*/4,32, /*down*/2,16, /*mode*/MODE_SUFD, /*tile*/48,32, /*warps*/20, /*xrep*/ 0) // 8t-ups4-downf2 // UNOPTIMIZED
    CASE(0,  /*up*/2,16, /*down*/4,32, /*mode*/MODE_SUFD, /*tile*/16,13, /*warps*/32, /*xrep*/ 0) // 8t-ups2-downf4 // UNOPTIMIZED

    // --------------------------------------------------------------------------------------------

    #undef CASE
    #undef CASE_SH
    #undef CASE_V

    return s;
}

//------------------------------------------------------------------------
