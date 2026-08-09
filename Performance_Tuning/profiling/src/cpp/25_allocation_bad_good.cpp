// Purpose: 每帧分配vs预分配复用；Recommended: heaptrack, perf, Massif.
#include "benchmark.hpp"
#include <atomic>
#include <cstddef>
#include <iostream>
#include <vector>
std::atomic<std::size_t> allocations{0},bytes{0};
void* operator new(std::size_t n){allocations++;bytes+=n;if(void*p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}void operator delete(void*p,std::size_t)noexcept{std::free(p);}
int main(){constexpr int frames=20000,n=4096;auto bad=[](){double s=0;for(int f=0;f<frames;++f){std::vector<float>v;for(int i=0;i<n;++i)v.push_back(float(i+f));s+=v.back();}return s;};std::vector<float>buffer;buffer.reserve(n);auto good=[&](){double s=0;for(int f=0;f<frames;++f){buffer.clear();for(int i=0;i<n;++i)buffer.push_back(float(i+f));s+=buffer.back();}return s;};
auto before=allocations.load();auto bs=lab::benchmark(bad,1,3);auto bad_alloc=allocations.load()-before;before=allocations.load();auto gs=lab::benchmark(good,1,3);auto good_alloc=allocations.load()-before;if(std::abs(bad()-good())>.1)return 2;lab::print_stats("bad",bs);lab::print_stats("good",gs);std::cout<<"bad_allocations="<<bad_alloc<<" good_allocations="<<good_alloc<<" observed_allocated_bytes="<<bytes.load()<<'\n';}
