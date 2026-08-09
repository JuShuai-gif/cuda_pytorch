#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
int main(){auto s=std::chrono::steady_clock::now();std::size_t sum=0;for(int r=0;r<2000000;++r){auto p=std::make_unique<std::vector<int>>(32,r);sum+=(*p)[0];}std::cout<<"sum="<<sum<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count()<<'\n';}

