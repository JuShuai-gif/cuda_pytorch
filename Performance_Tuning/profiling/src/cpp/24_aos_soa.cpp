// Purpose: 只处理x/y时比较AoS与SoA；Recommended: perf, VTune, vectorization report.
#include "benchmark.hpp"
#include <cmath>
#include <iostream>
#include <vector>
struct Particle{float x,y,z,vx,vy,vz,mass,pad;};struct SoA{std::vector<float>x,y,z,vx,vy,vz,mass,pad;explicit SoA(std::size_t n):x(n,1),y(n,2),z(n),vx(n),vy(n),vz(n),mass(n),pad(n){}};
int main(){std::size_t n=1<<24;std::vector<Particle>a(n);for(auto&p:a){p.x=1;p.y=2;}SoA s(n);auto aos=[&]{double sum=0;for(auto&p:a)sum+=p.x*.5f+p.y*.25f;return sum;};auto soa=[&]{double sum=0;for(std::size_t i=0;i<n;++i)sum+=s.x[i]*.5f+s.y[i]*.25f;return sum;};if(std::abs(aos()-soa())>1)return 2;lab::print_stats("AoS",lab::benchmark(aos,2,8));lab::print_stats("SoA",lab::benchmark(soa,2,8));}
