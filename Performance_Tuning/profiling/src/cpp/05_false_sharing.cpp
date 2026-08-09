#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
struct Packed{std::atomic<long long> x{0};}; struct alignas(64) Padded{std::atomic<long long>x{0};};
template<class T> double run(int n){std::vector<T> a(n);std::vector<std::thread>ts;auto s=std::chrono::steady_clock::now();for(int k=0;k<n;++k)ts.emplace_back([&,k]{for(int i=0;i<10000000;++i)a[k].x.fetch_add(1,std::memory_order_relaxed);});for(auto&t:ts)t.join();return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count();}
int main(){int n=std::min(8u,std::max(2u,std::thread::hardware_concurrency()));std::cout<<"threads="<<n<<" packed_ms="<<run<Packed>(n)<<" padded_ms="<<run<Padded>(n)<<'\n';}

