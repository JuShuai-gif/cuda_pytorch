#include <chrono>
#include <iostream>
#include <vector>
int main(int argc,char**argv){std::size_t n=argc>1?std::stoull(argv[1]):50000000;std::vector<float>a(n,1),b(n,2),c(n);auto t=std::chrono::steady_clock::now();for(int r=0;r<5;++r)for(std::size_t i=0;i<n;++i)c[i]=a[i]+3*b[i];double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::cout<<"GB/s="<<(5.0*n*3*sizeof(float)/s/1e9)<<" checksum="<<c[n/2]<<'\n';}

