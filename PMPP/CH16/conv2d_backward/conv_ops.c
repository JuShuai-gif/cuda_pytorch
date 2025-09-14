#include <stdio.h>

void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K, float* dE_dY, float* W, float* dE_dX) {
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    // Initialize dE_dX to zeros
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H_in; h++) {
            for (int w = 0; w < W_in; w++) {
                dE_dX[c * H_in * W_in + h * W_in + w] = 0;
            }
        }
    }

    // Compute gradients
    for (int m = 0; m < M; m++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                for (int c = 0; c < C; c++) {
                    for (int p = 0; p < K; p++) {
                        for (int q = 0; q < K; q++) {
                            int h_in = h_out + p;
                            int w_in = w_out + q;
                            dE_dX[c * H_in * W_in + h_in * W_in + w_in] +=
                                dE_dY[m * H_out * W_out + h_out * W_out + w_out] *
                                W[m * C * K * K + c * K * K + p * K + q];
                        }
                    }
                }
            }
        }
    }
}
