// Purpose: 危险正确性实验；仅手动用ASan/Valgrind运行，不进入自动脚本。
#include <iostream>
int main(){auto p=new int[8];for(int i=0;i<=8;++i)p[i]=i;volatile int x=p[8];delete[]p;std::cout<<"intentional_overflow="<<x<<'\n';}
