#include <iostream>
int main(){int*p=new int(42);delete p;volatile int x=*p;std::cout<<"故意 UAF value="<<x<<'\n';}

