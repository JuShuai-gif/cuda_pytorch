#include <chrono>
#include <iostream>
#include <vector>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
int main(){std::vector<double>a(40000000,1.0);auto s=std::chrono::steady_clock::now();double sum=0;
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
for(std::size_t i=0;i<a.size();++i)sum+=a[i]*(i%17);std::cout<<"sum="<<sum<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count();
#ifdef HAVE_OPENMP
std::cout<<" threads="<<omp_get_max_threads();
#endif
std::cout<<'\n';}
