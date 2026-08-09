// Purpose: 教学 STREAM Copy/Scale/Add/Triad，单线程与OpenMP。
// Recommended: perf, VTune Memory Access, LIKWID, PCM.
#include "benchmark.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
int main(){std::size_t n=1<<24;std::vector<double>a(n,1),b(n,2),c(n);auto run=[&](int op){auto begin=lab::Clock::now();
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
for(std::size_t i=0;i<n;++i){if(op==0)c[i]=a[i];else if(op==1)b[i]=3*c[i];else if(op==2)c[i]=a[i]+b[i];else a[i]=b[i]+3*c[i];}double sec=std::chrono::duration<double>(lab::Clock::now()-begin).count();std::size_t streams=op<2?2:3;std::cout<<(op==0?"Copy":op==1?"Scale":op==2?"Add":"Triad")<<" GB/s="<<n*sizeof(double)*streams/sec/1e9<<'\n';};for(int warm=0;warm<2;++warm)for(int op=0;op<4;++op)run(op);std::cout<<"checksum="<<a[n/2]+b[n/2]+c[n/2];
#ifdef HAVE_OPENMP
std::cout<<" threads="<<omp_get_max_threads();
#endif
std::cout<<'\n';}
