// AllGather engine plugin (mccl-test)
#include "common.h"

static void allgather_getBuffSize(size_t *sendBytes, size_t *recvBytes,
                                  size_t countBytes, int nranks) {
  *sendBytes = countBytes;
  *recvBytes = countBytes * (size_t)nranks;
}

static void allgather_initData(threadArgs_t *args, int root, DataType type,
                               RedOp op, int in_place) {
  (void)root;
  (void)type;
  (void)op;
  (void)in_place;
  // Initialize send with rank value; recv zeroed for clarity.
  MUSACHECK(musaMemset(args->sendbuff, 0, args->sendBytes));
  MUSACHECK(musaMemset(args->recvbuff, 0, args->recvBytes));
  float rank_value = (float)args->rank;
  MUSACHECK(musaMemcpy(args->sendbuff, &rank_value, sizeof(float),
                       musaMemcpyHostToDevice));
}

static void allgather_runTest(threadArgs_t *args, int root, DataType type,
                              RedOp op, const void *sendbuff, void *recvbuff,
                              musaStream_t stream) {
  (void)root;
  (void)type;
  (void)op;
  // AllGather concatenates per-rank payloads in rank order.
  MCCLCHECK(mcclAllGather(sendbuff, recvbuff, args->count, mcclFloat, args->comm,
                          stream));
}

static void allgather_getBw(size_t sendBytes, size_t recvBytes, int nranks,
                            double timeSec, double *algbw, double *busbw) {
  (void)sendBytes;
  (void)nranks;
  if (timeSec <= 0.0) {
    *algbw = 0.0;
    *busbw = 0.0;
    return;
  }
  // Use total receive payload as algorithmic bytes.
  double bytes = (double)recvBytes;
  *algbw = bytes / timeSec / 1e9;
  *busbw = *algbw;
}

static int allgather_checkData(threadArgs_t *args, int root, DataType type,
                               RedOp op, int in_place) {
  (void)root;
  (void)type;
  (void)op;
  (void)in_place;
  // Validate first element of each gathered chunk equals its rank.
  int errors = 0;
  for (int r = 0; r < args->nranks; r++) {
    float host_value = -1.0f;
    const size_t offset = (size_t)r * args->count;
    MUSACHECK(musaMemcpy(&host_value,
                         (float *)args->recvbuff + offset, sizeof(float),
                         musaMemcpyDeviceToHost));
    if (host_value != (float)r) {
      errors++;
      printf("Rank %d: mismatch at chunk %d value %.0f expected %d\n",
             args->rank, r, host_value, r);
    }
  }
  return errors;
}

testEngine_t allGatherEngine = {
    .name = "allgather",
    .defaultSizeBytes = (1 << 18) * sizeof(float),
    .supportsInplace = 0,
    .defaultType = DATA_FLOAT,
    .defaultTypeName = "float",
    .defaultOp = OP_NONE,
    .defaultOpName = "none",
    .defaultRoot = 0,
    .getBuffSize = allgather_getBuffSize,
    .initData = allgather_initData,
    .runTest = allgather_runTest,
    .getBw = allgather_getBw,
    .checkData = allgather_checkData,
};

#pragma weak mcclTestEngine=allGatherEngine
