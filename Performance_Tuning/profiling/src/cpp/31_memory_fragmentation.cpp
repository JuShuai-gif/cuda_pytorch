// Purpose: 不同尺寸交错分配/释放，演示fragmentation不同于leak；Recommended: heaptrack, Massif, jemalloc/tcmalloc.
#include "benchmark.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unistd.h>
std::size_t rss(){std::ifstream f("/proc/self/statm");std::size_t all,res;f>>all>>res;return res*sysconf(_SC_PAGESIZE);}
int main(){std::mt19937 g(42);std::uniform_int_distribution<int>d(64,65536);std::vector<std::unique_ptr<char[]>>p(20000);for(auto&x:p)x=std::make_unique<char[]>(d(g));auto peak=rss();for(std::size_t i=0;i<p.size();i+=2)p[i].reset();auto after_free=rss();for(std::size_t i=0;i<p.size();i+=2)p[i]=std::make_unique<char[]>(d(g)/2);auto refill=rss();p.clear();std::cout<<"rss_peak="<<peak<<" rss_after_partial_free="<<after_free<<" rss_after_refill="<<refill<<" rss_after_release="<<rss()<<" live_objects=0\n";}
