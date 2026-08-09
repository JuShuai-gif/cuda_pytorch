#include <chrono>
#include <fstream>
#include <iostream>
int main(int argc,char**argv){std::string p=argc>1?argv[1]:"io_bottleneck.tmp";auto s=std::chrono::steady_clock::now();std::ofstream f(p);for(int i=0;i<200000;++i){f<<i<<'\n';f.flush();}std::cout<<"path="<<p<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count()<<'\n';}

