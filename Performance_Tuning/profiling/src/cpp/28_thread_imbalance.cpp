// Purpose: 70%单线程分区vs动态原子任务分配；Recommended: pidstat, perf, VTune timeline.
#include "benchmark.hpp"
#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>
int main(){const int nt=8;constexpr int tasks=8000,work=3000;auto kernel=[](int id){double s=0;for(int k=0;k<work;++k)s+=std::sin(id+k*.001);return s;};auto run=[&](bool balanced){std::vector<double>partial(nt);std::vector<std::thread>ts;std::atomic<int>next{0};for(int t=0;t<nt;++t)ts.emplace_back([&,t]{int begin=0,end=0;if(!balanced){begin=t?5600+(t-1)*(tasks-5600)/(nt-1):0;end=t?5600+t*(tasks-5600)/(nt-1):5600;}if(balanced){for(;;){int i=next.fetch_add(1);if(i>=tasks)break;partial[t]+=kernel(i);}}else for(int i=begin;i<end;++i)partial[t]+=kernel(i);});for(auto&x:ts)x.join();return std::accumulate(partial.begin(),partial.end(),0.0);};double x=run(false),y=run(true);if(std::abs(x-y)>1e-5*std::abs(x))return 2;lab::print_stats("imbalanced",lab::benchmark([&]{return run(false);},1,4));lab::print_stats("balanced",lab::benchmark([&]{return run(true);},1,4));}
