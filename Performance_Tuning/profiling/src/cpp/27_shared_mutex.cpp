// Purpose: read-mostly下mutex vs shared_mutex；并非保证shared_mutex更快。
#include "benchmark.hpp"
#include <atomic>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>
template<class M,bool Shared> uint64_t run(){M m;uint64_t value=1,sum=0;std::mutex sum_m;std::vector<std::thread>ts;for(int r=0;r<12;++r)ts.emplace_back([&]{uint64_t local=0;for(int i=0;i<200000;++i){if constexpr(Shared){std::shared_lock<M>g(m);local+=value;}else{std::lock_guard<M>g(m);local+=value;}}std::lock_guard<std::mutex>g(sum_m);sum+=local;});ts.emplace_back([&]{for(int i=0;i<2000;++i){std::unique_lock<M>g(m);++value;}});for(auto&t:ts)t.join();return sum+value;}
int main(){auto a=lab::benchmark([]{return run<std::mutex,false>();},1,5);auto b=lab::benchmark([]{return run<std::shared_mutex,true>();},1,5);lab::print_stats("mutex",a);lab::print_stats("shared_mutex",b);}
