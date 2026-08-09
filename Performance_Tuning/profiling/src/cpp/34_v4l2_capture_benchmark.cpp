// Purpose: 真实V4L2 Camera capture latency/FPS/drop/sequence gap实验。
// Bad/Good: 通过--buffers、--sleep-ms和目标pixel format做A/B。
// Recommended Profiler: strace, perf, trace-cmd, v4l2-ctl; 与nsys阶段时间线关联。
#include "benchmark.hpp"
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Buffer { void* ptr{}; std::size_t length{}; };
static int xioctl(int fd, unsigned long request, void* arg) {
  int rc; do rc = ioctl(fd, request, arg); while (rc < 0 && errno == EINTR); return rc;
}
static uint32_t fourcc(const std::string& s) {
  if (s.size()!=4) throw std::invalid_argument("format必须是4字符FOURCC，如YUYV/MJPG");
  return v4l2_fourcc(s[0],s[1],s[2],s[3]);
}
int main(int argc,char**argv){
  std::string device="/dev/video0",fmt="YUYV";int width=640,height=480,frames=300,buffers=4,sleep_ms=0;
  for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::invalid_argument("参数缺值");return std::string(argv[i]);};if(a=="--device")device=value();else if(a=="--width")width=std::stoi(value());else if(a=="--height")height=std::stoi(value());else if(a=="--format")fmt=value();else if(a=="--frames")frames=std::stoi(value());else if(a=="--buffers")buffers=std::stoi(value());else if(a=="--sleep-ms")sleep_ms=std::stoi(value());else if(a=="--help"){std::cout<<"--device /dev/video0 --width 640 --height 480 --format YUYV --frames 300 --buffers 4 --sleep-ms 0\n";return 0;}}
  int fd=open(device.c_str(),O_RDWR|O_NONBLOCK);if(fd<0){std::cerr<<"SKIP: 无法打开"<<device<<": "<<strerror(errno)<<'\n';return 0;}
  v4l2_capability cap{};if(xioctl(fd,VIDIOC_QUERYCAP,&cap)<0)throw std::runtime_error("VIDIOC_QUERYCAP失败");
  v4l2_format f{};f.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;f.fmt.pix.width=width;f.fmt.pix.height=height;f.fmt.pix.pixelformat=fourcc(fmt);f.fmt.pix.field=V4L2_FIELD_ANY;if(xioctl(fd,VIDIOC_S_FMT,&f)<0)throw std::runtime_error("VIDIOC_S_FMT失败");
  v4l2_requestbuffers req{};req.count=buffers;req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;req.memory=V4L2_MEMORY_MMAP;if(xioctl(fd,VIDIOC_REQBUFS,&req)<0||req.count<2)throw std::runtime_error("申请MMAP buffer失败");
  std::vector<Buffer> mapped(req.count);for(uint32_t i=0;i<req.count;++i){v4l2_buffer b{};b.type=req.type;b.memory=req.memory;b.index=i;if(xioctl(fd,VIDIOC_QUERYBUF,&b)<0)throw std::runtime_error("QUERYBUF失败");mapped[i]={mmap(nullptr,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,b.m.offset),b.length};if(mapped[i].ptr==MAP_FAILED)throw std::runtime_error("mmap失败");if(xioctl(fd,VIDIOC_QBUF,&b)<0)throw std::runtime_error("QBUF失败");}
  auto type=req.type;if(xioctl(fd,VIDIOC_STREAMON,&type)<0)throw std::runtime_error("STREAMON失败");std::vector<double> intervals;intervals.reserve(frames);uint32_t previous=0,gaps=0;auto start=lab::Clock::now(),last=start;std::size_t bytes=0;
  for(int n=0;n<frames;){pollfd p{fd,POLLIN,0};int rc=poll(&p,1,2000);if(rc<=0){std::cerr<<"capture timeout\n";break;}v4l2_buffer b{};b.type=req.type;b.memory=req.memory;if(xioctl(fd,VIDIOC_DQBUF,&b)<0){if(errno==EAGAIN)continue;throw std::runtime_error("DQBUF失败");}auto now=lab::Clock::now();if(n)intervals.push_back(std::chrono::duration<double,std::milli>(now-last).count());last=now;if(n&&b.sequence>previous+1)gaps+=b.sequence-previous-1;previous=b.sequence;bytes+=b.bytesused;if(sleep_ms)usleep(sleep_ms*1000);if(xioctl(fd,VIDIOC_QBUF,&b)<0)throw std::runtime_error("re-QBUF失败");++n;}
  xioctl(fd,VIDIOC_STREAMOFF,&type);double sec=std::chrono::duration<double>(lab::Clock::now()-start).count();if(!intervals.empty())lab::print_stats("frame_interval",intervals);std::cout<<"device="<<device<<" negotiated="<<f.fmt.pix.width<<'x'<<f.fmt.pix.height<<" format="<<fmt<<" captured="<<frames<<" sequence_gaps="<<gaps<<" FPS="<<frames/sec<<" MBps="<<bytes/sec/1e6<<'\n';for(auto&m:mapped)munmap(m.ptr,m.length);close(fd);}
