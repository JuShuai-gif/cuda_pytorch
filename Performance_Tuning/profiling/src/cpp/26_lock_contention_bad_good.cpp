// Purpose: 大临界区vs局部归约批量更新；Recommended: perf, strace -c, VTune Locks, bpftrace.
#include "benchmark.hpp"
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
int main(){const int threads=std::min(12u,std::max(2u,std::thread::hardware_concurrency())),n=300000;auto run=[&](bool good){std::mutex m;double total=0;std::vector<std::thread>ts;for(int t=0;t<threads;++t)ts.emplace_back([&,t]{if(good){double local=0;for(int i=0;i<n;++i)local+=std::sqrt(i+t+1.0);std::lock_guard<std::mutex>g(m);total+=local;}else for(int i=0;i<n;++i){double v=std::sqrt(i+t+1.0);std::lock_guard<std::mutex>g(m);total+=v;}});for(auto&x:ts)x.join();return total;};if(std::abs(run(false)-run(true))>1e-5*run(true))return 2;lab::print_stats("bad_large_critical_section",lab::benchmark([&]{return run(false);},1,5));lab::print_stats("good_local_reduce",lab::benchmark([&]{return run(true);},1,5));}
