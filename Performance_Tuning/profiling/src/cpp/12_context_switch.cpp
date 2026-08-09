#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
int main(){std::mutex m;std::condition_variable cv;bool turn=false;int count=0;constexpr int N=200000;auto f=[&](bool mine){for(int i=0;i<N;++i){std::unique_lock<std::mutex>l(m);cv.wait(l,[&]{return turn==mine;});++count;turn=!turn;l.unlock();cv.notify_one();}};std::thread a(f,false),b(f,true);a.join();b.join();std::cout<<"handoffs="<<count<<'\n';}

