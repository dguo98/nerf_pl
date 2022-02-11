#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "filtered_lrelu.h"
#include <iostream>

//------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor> filtered_lrelu(
    torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b, torch::Tensor si,
    int up, int down, int px0, int py0, int px1, int py1, float gain, float slope, float clamp, bool flip_filters, int s_zero_ofs, int variant)
{
    // Set CUDA device.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    // Validate arguments.
    TORCH_CHECK(fu.device() == x.device() && fd.device() == x.device() && b.device() == x.device(), "all input tensors must reside on the same device");
    TORCH_CHECK(fu.dtype() == torch::kFloat && fd.dtype() == torch::kFloat, "fu and fd must be float32");
    TORCH_CHECK(b.dtype() == x.dtype(), "b must have the same dtype as x");
    TORCH_CHECK(x.numel() <= INT_MAX && fu.numel() <= INT_MAX && fd.numel() <= INT_MAX && b.numel() <= INT_MAX, "inputs are too large");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(fu.dim() == 3 && fd.dim() == 3, "fu and fd must be rank 3");
    TORCH_CHECK(b.dim() == 1 && b.size(0) == x.size(1), "b must be a vector with the same number of channels as x");
    TORCH_CHECK(up >= 1 && down >= 1, "up and down must be at least 1");

    // Dig up input sizes.
    int xw = (int)x.size(3);
    int xh = (int)x.size(2);
    int fut = (int)fu.size(-1) - 1;
    int fdt = (int)fd.size(-1) - 1;

    // Allocate output.
    int yw = (xw * up + (px0 + px1) - (fut + fdt) + (down - 1)) / down;
    int yh = (xh * up + (py0 + py1) - (fut + fdt) + (down - 1)) / down;
    TORCH_CHECK(yw >= 1 && yh >= 1, "output must be at least 1x1");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), yh, yw}, x.options(), x.suggest_memory_format());

    // Allocate signs.
    int sw = xw * up - fut + std::min(px0, fut) + std::min(px1, fut - (up - 1));
    int sh = xh * up - fut + std::min(py0, fut) + std::min(py1, fut - (up - 1));
    int sw_bytes = ((sw + 3) >> 2) + 1; // Width in bytes, rounded up. One extra byte to ensure byte alignment on the left.
    torch::Tensor so;
    torch::Tensor s = si;
    if (!s.numel())
        s = so = torch::empty({x.size(0), x.size(1), sh, sw_bytes}, x.options().dtype(torch::kUInt8), at::MemoryFormat::Contiguous);

    // Validate signs.
    TORCH_CHECK(s.dtype() == torch::kUInt8, "signs must be uint8");
    TORCH_CHECK(s.device() == x.device(), "signs must reside on the same device as x");
    TORCH_CHECK(s.dim() == 4, "signs must be rank 4");
    TORCH_CHECK(s.size(0) == x.size(0) && s.size(1) == x.size(1), "signs must have same batch & channels as x");
    TORCH_CHECK(s.size(2) == sh && s.size(3) == sw_bytes, "signs must be consistent between forward & backward");

    // Make sure that we can index everything with int32.
    TORCH_CHECK(x.nbytes() <= INT_MAX, "input is too large");
    TORCH_CHECK(y.nbytes() <= INT_MAX, "output is too large");
    TORCH_CHECK(s.nbytes() <= INT_MAX, "signs are too large");

    // Initialize CUDA kernel parameters.
    filtered_lrelu_kernel_params p;
    p.x         = x.data_ptr();
    p.fu        = fu.data_ptr<float>();
    p.fd        = fd.data_ptr<float>();
    p.b         = b.data_ptr();
    p.si        = (si.numel()) ? si.data_ptr<unsigned char>() : NULL;
    p.y         = y.data_ptr();
    p.so        = (so.numel()) ? so.data_ptr<unsigned char>() : NULL;

    p.up        = up;
    p.down      = down;
    p.pad0      = make_int2(px0, py0);
    p.gain      = gain;
    p.slope     = slope;
    p.clamp     = clamp;
    p.flip      = (flip_filters) ? 1 : 0;
    p.variant   = variant;

    p.gain      = gain;
    p.xShape    = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.xStride   = make_int4((int)x.stride(3), (int)x.stride(2), (int)x.stride(1), (int)x.stride(0));
    p.fuShape   = make_int3((int)fu.size(2), (int)fu.size(1), (int)fu.size(0));
    p.fuStride  = make_int3((int)fu.stride(2), (int)fu.stride(1), (int)fu.stride(0));
    p.fdShape   = make_int3((int)fd.size(2), (int)fd.size(1), (int)fd.size(0));
    p.fdStride  = make_int3((int)fd.stride(2), (int)fd.stride(1), (int)fd.stride(0));
    p.bStride   = (int)b.stride(0);
    p.yShape    = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.yStride   = make_int4((int)y.stride(3), (int)y.stride(2), (int)y.stride(1), (int)y.stride(0));
    p.sShape    = make_int4((int)s.size(3), (int)s.size(2), (int)s.size(1), (int)s.size(0));
    p.sStride   = make_int4((int)s.stride(3), (int)s.stride(2), (int)s.stride(1), (int)s.stride(0));
    p.sOfs      = make_int2(std::min(fut - p.pad0.x, 0), std::min(fut - p.pad0.y, 0));
    p.sByteOfs  = (-s_zero_ofs) & 3; // Byte-alignment offset.
    p.sWidth    = sw;                // Logical width of sign buffer.

    // Figure out how much shared memory is available.
    int maxSharedBytes = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, x.device().index()));
    p.sharedKB = (maxSharedBytes + 1023) >> 10;

    // Choose CUDA kernel.
    filtered_lrelu_kernel_spec spec;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filtered_lrelu_cuda", [&]
    {
        // Convert input/output strides into byte strides to avoid some indexing logic.
        p.bStride   *= sizeof(scalar_t);
        p.xStride.x *= sizeof(scalar_t); p.xStride.y *= sizeof(scalar_t); p.xStride.z *= sizeof(scalar_t); p.xStride.w *= sizeof(scalar_t);
        p.yStride.x *= sizeof(scalar_t); p.yStride.y *= sizeof(scalar_t); p.yStride.z *= sizeof(scalar_t); p.yStride.w *= sizeof(scalar_t);
        if (p.so){
//             std::cout << sizeof(scalar_t);
//             std::cout << "RUNNING A!!!!!!!!!!!!!!!!!!!!!!!";
            spec = choose_filtered_lrelu_kernel<scalar_t, true>(p);}
        else{
//             std::cout << "RUNNING B!!!!!!!!!!!!!!!!!!!!!!!!";
            spec = choose_filtered_lrelu_kernel<scalar_t, false>(p);}
    });
    TORCH_CHECK(spec.exec, "no appropriate CUDA kernel found");

    // Launch CUDA kernel.
    void* args[] = {&p};
    int bx = spec.numWarps * 32;
    int gx = (p.yShape.x - 1) / spec.tileOut.x + 1;
    int gy = (p.yShape.y - 1) / spec.tileOut.y + 1;
    int gz = p.yShape.z * p.yShape.w;

    // Repeat multiple tiles in a CTA?
    if (spec.xrep)
    {
        p.tilesXrep = spec.xrep;
        p.tilesXdim = gx;

        gx = (gx + p.tilesXrep - 1) / p.tilesXrep;
        std::swap(gx, gy);
    }

    AT_CUDA_CHECK(cudaLaunchKernel(spec.setup, 1, 1024, args, 0, at::cuda::getCurrentCUDAStream()));
    AT_CUDA_CHECK(cudaFuncSetCacheConfig(spec.exec, cudaFuncCachePreferShared));
    if (spec.sharedKB) // Need dynamic shared memory?
        AT_CUDA_CHECK(cudaFuncSetAttribute(spec.exec, cudaFuncAttributeMaxDynamicSharedMemorySize, spec.sharedKB << 10));
    AT_CUDA_CHECK(cudaFuncSetSharedMemConfig(spec.exec, cudaSharedMemBankSizeFourByte));
    AT_CUDA_CHECK(cudaLaunchKernel(spec.exec, dim3(gx, gy, gz), bx, args, spec.sharedKB << 10, at::cuda::getCurrentCUDAStream()));
    return std::make_tuple(y, so);
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("filtered_lrelu", &filtered_lrelu);
}

//------------------------------------------------------------------------
