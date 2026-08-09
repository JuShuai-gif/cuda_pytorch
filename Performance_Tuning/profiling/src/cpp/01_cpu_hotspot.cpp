#include <chrono>
#include <cmath>
#include <iostream>
static double hot(std::size_t n){ volatile double s=0; for(std::size_t i=1;i<n;++i) s+=std::sin(i*.001)*std::sqrt(double(i)); return s; }
int main(int argc,char**argv){ auto n=argc>1?std::stoull(argv[1]):30000000ULL; auto t=std::chrono::steady_clock::now(); std::cout<<"result="<<hot(n)<<" elapsed_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()<<'\n'; }

