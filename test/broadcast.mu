// Broadcast engine plugin (mccl-test)
#include "common.h"

static void broadcast_getBuffSize(size_t *sendBytes, size_t *recvBytes,
                                  size_t countBytes, int nranks) {
  (void)nranks;
  *sendBytes = countBytes;
  *recvBytes = countBytes;
}

static void broadcast_initData(threadArgs_t *args, int root, DataType type,
                               RedOp op, int in_place) {
  (void)type;
  (void)op;
  (void)in_place;
  MUSACHECK(musaMemset(args->recvbuff, 0, args->recvBytes));
  if (args->rank == root) {
    MUSACHECK(musaMemset(args->sendbuff, 0, args->sendBytes));
    float root_value = (float)root;
    MUSACHECK(musaMemcpy(args->sendbuff, &root_value, sizeof(float),
                         musaMemcpyHostToDevice));
  }
}

static void broadcast_runTest(threadArgs_t *args, int root, DataType type,
                              RedOp op, const void *sendbuff, void *recvbuff,
                              musaStream_t stream) {
  (void)type;
  (void)op;
  MCCLCHECK(mcclBroadcast(sendbuff, recvbuff, args->count, mcclFloat, root,
                          args->comm, stream));
}

static void broadcast_getBw(size_t sendBytes, size_t recvBytes, int nranks,
                            double timeSec, double *algbw, double *busbw) {
  (void)sendBytes;
  if (timeSec <= 0.0) {
    *algbw = 0.0;
    *busbw = 0.0;
    return;
  }
  double bytes = (double)recvBytes;
  *algbw = bytes / timeSec / 1e9;
  *busbw = *algbw;
}

static int broadcast_checkData(threadArgs_t *args, int root, DataType type,
                               RedOp op, int in_place) {
  (void)type;
  (void)op;
  (void)in_place;
  float host_result = -1.0f;
  MUSACHECK(musaMemcpy(&host_result, args->recvbuff, sizeof(float),
                       musaMemcpyDeviceToHost));
  if (host_result != (float)root) {
    printf("Rank %d: incorrect result %.0f (expected %.0f)\n", args->rank,
           host_result, (float)root);
    return 1;
  }
  return 0;
}

static size_t broadcast_getInplaceOffset(size_t sendBytes, size_t recvBytes,
                                         int nranks, int rank) {
  (void)sendBytes;
  (void)recvBytes;
  (void)nranks;
  (void)rank;
  return 0;
}

testEngine_t mcclTestEngine = {
    .name = "broadcast",
    .defaultSizeBytes = (1 << 20) * sizeof(float),
    .supportsInplace = 0,
    .defaultType = DATA_FLOAT,
    .defaultTypeName = "float",
    .defaultOp = OP_NONE,
    .defaultOpName = "none",
    .defaultRoot = 0,
    .getBuffSize = broadcast_getBuffSize,
    .initData = broadcast_initData,
    .runTest = broadcast_runTest,
    .getInplaceOffset = broadcast_getInplaceOffset,
    .getBw = broadcast_getBw,
    .checkData = broadcast_checkData,
};
