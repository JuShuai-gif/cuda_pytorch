// Purpose: perf 热点采样；Bad: 重复昂贵超越函数；Good: 递推复用。
// Recommended Profiler: perf record/report/annotate, FlameGraph, VTune Hotspots.
#include "benchmark.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
constexpr std::size_t N=3000000;
__attribute__((noinline)) double hot_bad(){double s=0;for(std::size_t i=1;i<=N;++i)s+=std::sin(i*.0001)*std::sqrt(i);return s;}
__attribute__((noinline)) double hot_good(){double s=0,angle=.0001;const double sd=std::sin(.0001),cd=std::cos(.0001);double sn=sd,cs=cd;for(std::size_t i=1;i<=N;++i){s+=sn*std::sqrt(i);double ns=sn*cd+cs*sd;cs=cs*cd-sn*sd;sn=ns;}return s;}
__attribute__((noinline)) double medium(){double s=0;for(std::size_t i=1;i<N/4;++i)s+=std::log1p(i);return s;}
__attribute__((noinline)) double cold(){double s=0;for(std::size_t i=1;i<N/40;++i)s+=std::sqrt(i);return s;}
int main(int argc,char**argv){std::string mode=argc>1?argv[1]:"both";auto run=[&](bool good){return (good?hot_good():hot_bad())+medium()+cold();};double a=run(false),b=run(true);double rel=std::abs(a-b)/std::max(1.0,std::abs(a));std::cout<<"correctness_relative_error="<<rel<<'\n';if(rel>1e-5)return 2;if(mode!="good")lab::print_stats("bad",lab::benchmark([&]{return run(false);},1,5));if(mode!="bad")lab::print_stats("good",lab::benchmark([&]{return run(true);},1,5));}
