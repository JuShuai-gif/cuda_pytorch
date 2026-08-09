#include <chrono>
#include <iostream>
#include <vector>
int main(){std::size_t n=50000000;std::vector<float>a(n,1),b(n,2),c(n);auto s=std::chrono::steady_clock::now();for(int r=0;r<10;++r)for(std::size_t i=0;i<n;++i)c[i]=a[i]*b[i]+c[i];std::cout<<"checksum="<<c[n/2]<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-s).count()<<'\n';}

