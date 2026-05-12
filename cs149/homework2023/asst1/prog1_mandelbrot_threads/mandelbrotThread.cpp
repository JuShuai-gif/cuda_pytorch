#include <stdio.h>
#include <thread>
#include <cstdlib>

#include "CycleTimer.h"

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;


extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);


//
// workerThreadStart --
//
// Thread entrypoint.
// Each thread processes its assigned rows using an interleaved
// (round-robin) mapping: thread i computes rows i, i+numThreads,
// i+2*numThreads, ...
// This static interleaved assignment provides good load balance
// without any synchronization because every thread gets a roughly
// equal mix of computationally heavy (black) and light (white) rows.
//
void workerThreadStart(WorkerArgs * const args) {

    double startTime = CycleTimer::currentSeconds();

    // Interleaved row assignment: each thread strides by numThreads
    for (int j = args->threadId; j < args->height; j += args->numThreads) {
        mandelbrotSerial(
            args->x0, args->y0, args->x1, args->y1,
            args->width, args->height,
            j, 1,                          // process one row at a time
            args->maxIterations,
            args->output);
    }

    double endTime = CycleTimer::currentSeconds();
    printf("[thread %d]:\t[%.3f] ms\n", args->threadId, (endTime - startTime) * 1000);
}


//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
// Work is decomposed using interleaved row assignment (round-robin),
// which gives each thread a mix of cheap and expensive rows regardless
// of the view region being computed.
//
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
      
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
      
        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    
    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
}
