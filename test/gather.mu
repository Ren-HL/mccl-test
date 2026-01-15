// Gather engine plugin (mccl-test)
#include "common.h"

static void gather_getBuffSize(size_t *sendBytes, size_t *recvBytes,
                               size_t countBytes, int nranks) {
  *sendBytes = countBytes;
  *recvBytes = countBytes * (size_t)nranks;
}

static void gather_initData(threadArgs_t *args, int root, DataType type,
                            RedOp op, int in_place) {
  (void)type;
  (void)op;
  (void)in_place;
  MUSACHECK(musaMemset(args->sendbuff, 0, args->sendBytes));
  if (args->rank == root) {
    MUSACHECK(musaMemset(args->recvbuff, 0, args->recvBytes));
  }
  float rank_value = (float)args->rank;
  MUSACHECK(musaMemcpy(args->sendbuff, &rank_value, sizeof(float),
                       musaMemcpyHostToDevice));
}

static void gather_runTest(threadArgs_t *args, int root, DataType type,
                           RedOp op, const void *sendbuff, void *recvbuff,
                           musaStream_t stream) {
  (void)type;
  (void)op;
  MCCLCHECK(mcclGather(sendbuff, recvbuff, args->count, mcclFloat, root,
                       args->comm, stream));
}

static void gather_getBw(size_t sendBytes, size_t recvBytes, int nranks,
                         double timeSec, double *algbw, double *busbw) {
  (void)sendBytes;
  (void)nranks;
  if (timeSec <= 0.0) {
    *algbw = 0.0;
    *busbw = 0.0;
    return;
  }
  double bytes = (double)recvBytes;
  *algbw = bytes / timeSec / 1e9;
  *busbw = *algbw;
}

static int gather_checkData(threadArgs_t *args, int root, DataType type,
                            RedOp op, int in_place) {
  (void)type;
  (void)op;
  (void)in_place;
  if (args->rank != root) {
    return 0;
  }
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

static size_t gather_getInplaceOffset(size_t sendBytes, size_t recvBytes,
                                      int nranks, int rank) {
  (void)recvBytes;
  (void)nranks;
  (void)rank;
  return sendBytes;
}

testEngine_t mcclTestEngine = {
    .name = "gather",
    .defaultSizeBytes = (1 << 18) * sizeof(float),
    .supportsInplace = 1,
    .defaultType = DATA_FLOAT,
    .defaultTypeName = "float",
    .defaultOp = OP_NONE,
    .defaultOpName = "none",
    .defaultRoot = 0,
    .getBuffSize = gather_getBuffSize,
    .initData = gather_initData,
    .runTest = gather_runTest,
    .getInplaceOffset = gather_getInplaceOffset,
    .getBw = gather_getBw,
    .checkData = gather_checkData,
};
