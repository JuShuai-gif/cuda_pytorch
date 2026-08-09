// Purpose: FFmpeg/LibAV真实视频demux→decode→可选swscale逐帧性能实验。
// Bad/Good: 软件decode+CPU scale与目标硬件decode/scale CLI结果对照。
// Recommended Profiler: perf, strace, nsys（硬件后端）, tegrastats.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include "benchmark.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string av_error(int code) {
  char text[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, text, sizeof(text));
  return text;
}
static void check(int code, const char* what) {
  if (code < 0) throw std::runtime_error(std::string(what) + ": " + av_error(code));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "usage: 35_ffmpeg_decode_benchmark INPUT [max_frames] [output_width] [output_height]\n";
    return 0;
  }
  const std::string input = argv[1];
  const int max_frames = argc > 2 ? std::stoi(argv[2]) : 500;
  const int out_width_arg = argc > 3 ? std::stoi(argv[3]) : 0;
  const int out_height_arg = argc > 4 ? std::stoi(argv[4]) : 0;
  AVFormatContext* format = nullptr;
  check(avformat_open_input(&format, input.c_str(), nullptr, nullptr), "open input");
  check(avformat_find_stream_info(format, nullptr), "find stream info");
  const int stream_index = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  check(stream_index, "find video stream");
  AVStream* stream = format->streams[stream_index];
  const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
  if (!codec) throw std::runtime_error("decoder not found");
  AVCodecContext* context = avcodec_alloc_context3(codec);
  if (!context) throw std::bad_alloc();
  check(avcodec_parameters_to_context(context, stream->codecpar), "copy codec params");
  context->thread_count = 0;  // FFmpeg自动选择线程数。
  check(avcodec_open2(context, codec, nullptr), "open decoder");
  const int out_width = out_width_arg > 0 ? out_width_arg : context->width;
  const int out_height = out_height_arg > 0 ? out_height_arg : context->height;
  SwsContext* scaler = nullptr;
  std::vector<uint8_t> rgb;
  uint8_t* rgb_planes[4]{};
  int rgb_linesize[4]{};
  if (out_width != context->width || out_height != context->height) {
    scaler = sws_getContext(context->width, context->height, context->pix_fmt,
                            out_width, out_height, AV_PIX_FMT_RGB24,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!scaler) throw std::runtime_error("sws_getContext failed");
    rgb.resize(av_image_get_buffer_size(AV_PIX_FMT_RGB24, out_width, out_height, 1));
    av_image_fill_arrays(rgb_planes, rgb_linesize, rgb.data(), AV_PIX_FMT_RGB24,
                         out_width, out_height, 1);
  }
  AVPacket* packet = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  if (!packet || !frame) throw std::bad_alloc();
  std::vector<double> frame_ms;
  frame_ms.reserve(max_frames);
  int frames = 0;
  uint64_t checksum = 0;
  const auto all_begin = lab::Clock::now();
  auto receive = [&](bool flushing) {
    while (frames < max_frames) {
      const auto begin = lab::Clock::now();
      int rc = avcodec_receive_frame(context, frame);
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return;
      check(rc, "receive frame");
      if (scaler) {
        sws_scale(scaler, frame->data, frame->linesize, 0, frame->height,
                  rgb_planes, rgb_linesize);
        checksum += rgb.empty() ? 0 : rgb[(static_cast<std::size_t>(frames) * 997) % rgb.size()];
      } else {
        checksum += frame->data[0] ? frame->data[0][0] : 0;
      }
      frame_ms.push_back(std::chrono::duration<double, std::milli>(
          lab::Clock::now() - begin).count());
      ++frames;
    }
  };
  while (frames < max_frames && av_read_frame(format, packet) >= 0) {
    if (packet->stream_index == stream_index) {
      int rc = avcodec_send_packet(context, packet);
      if (rc != AVERROR(EAGAIN)) check(rc, "send packet");
      receive(false);
    }
    av_packet_unref(packet);
  }
  avcodec_send_packet(context, nullptr);
  receive(true);
  const double elapsed = std::chrono::duration<double>(lab::Clock::now() - all_begin).count();
  if (!frame_ms.empty()) lab::print_stats("decode_receive_scale", frame_ms);
  const AVRational rate = av_guess_frame_rate(format, stream, nullptr);
  std::cout << "codec=" << codec->name << " input=" << context->width << 'x' << context->height
            << " output=" << out_width << 'x' << out_height
            << " nominal_fps=" << av_q2d(rate) << " decoded_frames=" << frames
            << " wall_s=" << elapsed << " throughput_fps=" << frames / elapsed
            << " checksum=" << checksum << '\n';
  av_frame_free(&frame); av_packet_free(&packet); sws_freeContext(scaler);
  avcodec_free_context(&context); avformat_close_input(&format);
}
