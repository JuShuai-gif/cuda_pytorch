// Purpose: 连续/随机与二维 row/column locality；Recommended: perf stat, Cachegrind, VTune.
#include "benchmark.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
int main(){constexpr std::size_t n=1<<24,side=4096;std::vector<uint32_t>a(n);std::iota(a.begin(),a.end(),1);std::vector<uint32_t>idx(n);std::iota(idx.begin(),idx.end(),0);std::mt19937 g(42);std::shuffle(idx.begin(),idx.end(),g);
volatile const uint32_t* memory=a.data();auto seq=[&]{uint64_t s=0;for(std::size_t i=0;i<n;++i)s+=memory[i];return s;};auto rnd=[&]{uint64_t s=0;for(auto i:idx)s+=memory[i];return s;};auto row=[&]{uint64_t s=0;for(std::size_t r=0;r<side;++r)for(std::size_t c=0;c<side;++c)s+=memory[r*side+c];return s;};auto col=[&]{uint64_t s=0;for(std::size_t c=0;c<side;++c)for(std::size_t r=0;r<side;++r)s+=memory[r*side+c];return s;};
auto expected=seq();if(rnd()!=expected||row()!=expected||col()!=expected)return 2;lab::print_stats("sequential",lab::benchmark(seq,1,5));lab::print_stats("random",lab::benchmark(rnd,1,5));lab::print_stats("row_major",lab::benchmark(row,1,5));lab::print_stats("column_major",lab::benchmark(col,1,5));std::cout<<"checksum="<<expected<<'\n';}
