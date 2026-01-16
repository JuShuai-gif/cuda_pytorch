#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "test1.fatbin.c"
extern void __device_stub__Z3addPiS_S_i(int *, int *, int *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z3addPiS_S_i(int *__par0, int *__par1, int *__par2, int __par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaLaunch(((char *)((void ( *)(int *, int *, int *, int))add)));}
# 4 "test1.cu"
void add( int *__cuda_0,int *__cuda_1,int *__cuda_2,int __cuda_3)
# 4 "test1.cu"
{__device_stub__Z3addPiS_S_i( __cuda_0,__cuda_1,__cuda_2,__cuda_3);



}
# 1 "./simple_add_tmp/test1.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T5) {  __nv_dummy_param_ref(__T5); __nv_save_fatbinhandle_for_managed_rt(__T5); __cudaRegisterEntry(__T5, ((void ( *)(int *, int *, int *, int))add), _Z3addPiS_S_i, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
