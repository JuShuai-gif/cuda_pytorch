#include <fcntl.h>
#include <iostream>
#include <unistd.h>
int main(){char c='x';for(int i=0;i<200000;++i){int fd=open("/dev/null",O_WRONLY);write(fd,&c,1);close(fd);}std::cout<<"完成 20 万组 open/write/close\n";}

