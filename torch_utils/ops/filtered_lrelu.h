#include <cuda_runtime.h>

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct filtered_lrelu_kernel_params
{
    const void*             x;
    const float*            fu;
    const float*            fd;
    const void*             b;
    const unsigned char*    si;
    void*                   y;
    unsigned char*          so;

    int     up;
    int     down;
    int2    pad0;
    float   gain;
    float   slope;
    float   clamp;
    int     flip;
    int     variant;    // For performance debugging.
    int     sharedKB;   // Maximum total shared memory available in kB.
    int     tilesXdim;  // Original number of horizontal output tiles.
    int     tilesXrep;  // Number of horizontal tiles per CTA.

    int4    xShape;     // [width, height, channel, batch]
    int4    xStride;
    int3    fuShape;    // [size, 1, 1] | [size, size, 1] | [size, components, 2]
    int3    fuStride;
    int3    fdShape;    // [size, 1, 1] | [size, size, 1] | [size, components, 2]
    int3    fdStride;
    int     bStride;
    int4    yShape;     // [width, height, channel, batch]
    int4    yStride;
    int4    sShape;     // [width, height, channel, batch]
    int4    sStride;
    int2    sOfs;
    int     sByteOfs;   // Logical offset for byte alignment.
    int     sWidth;     // Logical width in pixels.
};

//------------------------------------------------------------------------
// CUDA kernel specialization.

struct filtered_lrelu_kernel_spec
{
    void*   setup;
    void*   exec;
    int2    tileOut;
    int     numWarps;
    int     sharedKB;   // Dynamic shared memory requested in kB.
    int     xrep;       // X tiles per CTA.
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T, bool signWrite> filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel(const filtered_lrelu_kernel_params& p);

//------------------------------------------------------------------------
