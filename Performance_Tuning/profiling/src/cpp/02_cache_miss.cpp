#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
int main(int argc,char**argv){ std::size_t n=argc>1?std::stoull(argv[1]):8000000; std::vector<std::size_t> next(n); std::iota(next.begin(),next.end(),0); std::mt19937 g(42); std::shuffle(next.begin(),next.end(),g); std::size_t p=0; volatile std::size_t sum=0; auto t=std::chrono::steady_clock::now(); for(std::size_t i=0;i<n*4;++i){p=next[p];sum+=p;} std::cout<<"checksum="<<sum<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()<<'\n'; }

