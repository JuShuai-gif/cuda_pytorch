#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
static long long count(const std::vector<int>&v){ long long s=0; for(int x:v) if(x>500) s+=x; return s; }
int main(){std::vector<int> v(30000000);std::mt19937 g(7);std::uniform_int_distribution<int>d(0,1000);for(auto&x:v)x=d(g);auto t=std::chrono::steady_clock::now();auto a=count(v);std::sort(v.begin(),v.end());auto m=std::chrono::steady_clock::now();auto b=count(v);auto e=std::chrono::steady_clock::now();std::cout<<a+b<<" random_ms="<<std::chrono::duration<double,std::milli>(m-t).count()<<" sorted_ms="<<std::chrono::duration<double,std::milli>(e-m).count()<<'\n';}

