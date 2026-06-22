// pooling.c
#include <float.h>
#include <math.h>
#include <string.h>

void poolingLayer_forward(int M, int H, int W, int K, float* Y, float* S, const char* pooling_type) {
    for (int m = 0; m < M; m++) {              // for each output feature map
        for (int h = 0; h < H / K; h++) {      // for each output element,
            for (int w = 0; w < W / K; w++) {  // this code assumes that H and W
                // Initialize based on pooling type
                if (strcmp(pooling_type, "max") == 0) {
                    S[m * (H / K) * (W / K) + h * (W / K) + w] = -FLT_MAX;  // For max pooling
                } else {
                    S[m * (H / K) * (W / K) + h * (W / K) + w] = 0.0f;  // For avg pooling
                }

                // Loop over KxK input window
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        float val = Y[m * H * W + (K * h + p) * W + (K * w + q)];

                        if (strcmp(pooling_type, "max") == 0) {
                            // Max pooling
                            if (val > S[m * (H / K) * (W / K) + h * (W / K) + w]) {
                                S[m * (H / K) * (W / K) + h * (W / K) + w] = val;
                            }
                        } else {
                            // Average pooling (same as original)
                            S[m * (H / K) * (W / K) + h * (W / K) + w] += val / (K * K);
                        }
                    }
                }
            }
        }
    }
}
