// Purpose: 区分页错误与TLB/cache；Bad: 随机逐页；Good: 顺序逐页。
// Recommended: perf stat -e page-faults,minor-faults,major-faults，加 perf list 中可用 dTLB 事件。
#include "benchmark.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <unistd.h>
#include <vector>
int main(){const std::size_t page=sysconf(_SC_PAGESIZE),pages=32768;std::vector<unsigned char>data(page*pages);std::vector<std::size_t>order(pages);std::iota(order.begin(),order.end(),0);std::mt19937 g(7);std::shuffle(order.begin(),order.end(),g);for(std::size_t p=0;p<pages;++p)data[p*page]=1;
auto seq=[&]{uint64_t s=0;for(std::size_t p=0;p<pages;++p)s+=data[p*page];return s;};auto rnd=[&]{uint64_t s=0;for(auto p:order)s+=data[p*page];return s;};if(seq()!=rnd())return 2;lab::print_stats("sequential_pages",lab::benchmark(seq,2,8));lab::print_stats("random_pages",lab::benchmark(rnd,2,8));std::cout<<"page_size="<<page<<" bytes="<<data.size()<<'\n';}
