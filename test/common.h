#pragma once

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "musa_runtime.h"
#include "mccl.h"

#define MUSACHECK(cmd)                                                         \
  do {                                                                         \
    musaError_t err = cmd;                                                     \
    if (err != musaSuccess) {                                                  \
      printf("Failed: musa error %s:%d '%s'\n", __FILE__, __LINE__,            \
             musaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define MCCLCHECK(cmd)                                                         \
  do {                                                                         \
    mcclResult_t res = cmd;                                                    \
    if (res != mcclSuccess) {                                                  \
      printf("Failed, MCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             mcclGetErrorString(res));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

typedef enum {
  DATA_FLOAT = 0,
} DataType;

typedef enum {
  OP_SUM = 0,
  OP_NONE = 1,
} RedOp;

typedef struct testArgs {
  int nranks;
  size_t sizeBytes;
  size_t sendBytes;
  size_t recvBytes;
  size_t count;
  DataType type;
  const char *typeName;
  RedOp op;
  const char *opName;
  int root;
  int warmup_iters;
  int iters;
  int check;
} testArgs_t;

typedef struct threadArgs {
  int rank;
  int nranks;
  size_t sendBytes;
  size_t recvBytes;
  size_t count;
  int in_place;
  size_t in_place_offset;
  int send_is_alias;
  mcclUniqueId commId;
  mcclComm_t *comms;
  mcclComm_t comm;
  musaStream_t stream;
  musaEvent_t start;
  musaEvent_t stop;
  void *sendbuff;
  void *recvbuff;
  int errors;
  double avg_ms;
  double algbw;
  double busbw;
  const struct testEngine *engine;
  const testArgs_t *test;
} threadArgs_t;

typedef struct testEngine {
  const char *name;
  size_t defaultSizeBytes;
  int supportsInplace;
  DataType defaultType;
  const char *defaultTypeName;
  RedOp defaultOp;
  const char *defaultOpName;
  int defaultRoot;
  // Compute per-rank send/recv byte sizes from logical countBytes and nranks.
  void (*getBuffSize)(size_t *sendBytes, size_t *recvBytes, size_t countBytes,
                      int nranks);
  // Initialize device buffers (including expected data) for this rank.
  void (*initData)(threadArgs_t *args, int root, DataType type, RedOp op,
                   int in_place);
  // Enqueue one collective operation into the stream.
  void (*runTest)(threadArgs_t *args, int root, DataType type, RedOp op,
                  const void *sendbuff, void *recvbuff, musaStream_t stream);
  // Return byte offset into recv buffer for in-place send region.
  size_t (*getInplaceOffset)(size_t sendBytes, size_t recvBytes, int nranks,
                             int rank);
  // Compute algorithmic and bus bandwidth from timing.
  void (*getBw)(size_t sendBytes, size_t recvBytes, int nranks, double timeSec,
                double *algbw, double *busbw);
  // Optional data validation hook; return number of errors for this rank.
  int (*checkData)(threadArgs_t *args, int root, DataType type, RedOp op,
                   int in_place);
} testEngine_t;

extern testEngine_t mcclTestEngine;

int get_device_count();
int apply_env_override(int devicecount);
size_t get_type_size(DataType type);
void setupArgs(testArgs_t *args, const testEngine_t *engine);
