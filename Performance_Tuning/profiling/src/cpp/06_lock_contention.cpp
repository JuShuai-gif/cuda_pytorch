#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
int main(){std::mutex m;long long x=0;int n=std::min(16u,std::max(2u,std::thread::hardware_concurrency()));std::vector<std::thread>ts;auto s=std::chrono::steady_clock::now();for(int k=0;k<n;++k)ts.emplace_back([&]{for(int i=0;i<1000000;++i){std::lock_guard<std::mutex>g(m);++x;}});for(auto&t:ts)t.join();std::cout<<"value="<<x<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count()<<'\n';}

