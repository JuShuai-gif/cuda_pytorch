// Purpose: 50Hz控制循环wake-up/execution/jitter/deadline统计；可选 --interference。
#include "benchmark.hpp"
#include <atomic>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
int main(int argc,char**argv){bool load=argc>1&&std::string(argv[1])=="--interference";constexpr int samples=200,period_ms=20;std::atomic<bool>stop{false};std::thread noise;if(load)noise=std::thread([&]{volatile double x=0;while(!stop)x+=std::sin(x+.1);});std::vector<double>wakeup,execution,periods;int misses=0;auto epoch=lab::Clock::now();auto last=epoch;for(int i=1;i<=samples;++i){auto target=epoch+std::chrono::milliseconds(i*period_ms);std::this_thread::sleep_until(target);auto begin=lab::Clock::now();wakeup.push_back(std::chrono::duration<double,std::milli>(begin-target).count());periods.push_back(std::chrono::duration<double,std::milli>(begin-last).count());last=begin;volatile double x=0;for(int k=0;k<100000;++k)x+=std::sqrt(k+1.0);auto end=lab::Clock::now();double exec=std::chrono::duration<double,std::milli>(end-begin).count();execution.push_back(exec);if(end>target+std::chrono::milliseconds(period_ms))++misses;}stop=true;if(noise.joinable())noise.join();lab::print_stats("wakeup_latency",wakeup);lab::print_stats("execution",execution);lab::print_stats("period_jitter",periods);std::cout<<"period_ms="<<period_ms<<" deadline_miss_count="<<misses<<" deadline_miss_ratio="<<double(misses)/samples<<'\n';}
