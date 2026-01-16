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

typedef struct sizeInfo {
  // sizeBytes is the per-rank payload after any operator-specific alignment.
  size_t sizeBytes; // effective payload bytes per rank
  size_t sendBytes;
  size_t recvBytes;
  size_t count; // elements passed to collective
} sizeInfo_t;

typedef struct testArgs {
  int nranks;
  // Size sweep configuration (bytes).
  size_t minBytes;
  size_t maxBytes;
  size_t sizeBytes;
  size_t sendBytes;
  size_t recvBytes;
  size_t maxSendBytes;
  size_t maxRecvBytes;
  size_t count;
  DataType type;
  const char *typeName;
  RedOp op;
  const char *opName;
  int root;
  int warmup_iters;
  int iters;
  // agg_iters: number of collectives aggregated per timing window.
  int agg_iters;
  int check;
  int blocking;
  // 0=oop, 1=ip, 2=oop+ip
  int inplace_mode; // 0=oop, 1=ip, 2=oop+ip
  double stepFactor;
  size_t stepBytes;
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
  void *base_sendbuff;
  void *base_recvbuff;
  void *sendbuff;
  void *recvbuff;
  int errors;
  double avg_us;
  double algbw;
  double busbw;
  const sizeInfo_t *sizes;
  int nSizes;
  double *time_us;
  double *algbw_out;
  double *busbw_out;
  int *errors_out;
  int *errors_by_rank;
  pthread_barrier_t *barrier;
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
void parseArgs(int argc, char **argv, testArgs_t *args,
               const testEngine_t *engine);
void setupArgsForSize(testArgs_t *args, const testEngine_t *engine,
                      size_t sizeBytes);
