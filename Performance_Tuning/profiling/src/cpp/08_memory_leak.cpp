#include <iostream>
int main(int argc,char**argv){int n=argc>1?std::stoi(argv[1]):1000;for(int i=0;i<n;++i){auto*p=new char[4096];p[0]=char(i);if(i%4==0)delete[]p;}std::cout<<"故意泄漏约 "<<(n-n/4)*4096/1024<<" KiB\n";}

