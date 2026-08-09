// Purpose: 每字节write/open-close vs 单次批量write；Recommended: strace -c -T -f.
#include "benchmark.hpp"
#include <fcntl.h>
#include <iostream>
#include <string>
#include <unistd.h>
uint64_t run(bool good){constexpr int n=100000;std::string data(n,'x');uint64_t done=0;if(good){int fd=open("/dev/null",O_WRONLY);auto r=write(fd,data.data(),data.size());if(r>0)done=r;close(fd);}else for(char c:data){int fd=open("/dev/null",O_WRONLY);auto r=write(fd,&c,1);if(r>0)done+=r;close(fd);}return done;}
int main(){if(run(false)!=run(true))return 2;lab::print_stats("bad_small_syscalls",lab::benchmark([]{return run(false);},1,3));lab::print_stats("good_batched",lab::benchmark([]{return run(true);},1,3));}
