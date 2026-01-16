// Reduce engine plugin (mccl-test)
#include "common.h"

static void reduce_getBuffSize(size_t *sendBytes, size_t *recvBytes,
                               size_t countBytes, int nranks) {
  (void)nranks;
  *sendBytes = countBytes;
  *recvBytes = countBytes;
}

static void reduce_initData(threadArgs_t *args, int root, DataType type,
                            RedOp op, int in_place) {
  (void)type;
  (void)op;
  (void)in_place;
  MUSACHECK(musaMemset(args->sendbuff, 0, args->sendBytes));
  MUSACHECK(musaMemset(args->recvbuff, 0, args->recvBytes));
  float rank_value = (float)args->rank;
  MUSACHECK(musaMemcpy(args->sendbuff, &rank_value, sizeof(float),
                       musaMemcpyHostToDevice));
  if (args->rank != root) {
    MUSACHECK(musaMemset(args->recvbuff, 0, args->recvBytes));
  }
}

static void reduce_runTest(threadArgs_t *args, int root, DataType type,
                           RedOp op, const void *sendbuff, void *recvbuff,
                           musaStream_t stream) {
  (void)type;
  mcclDataType_t mccl_type = mcclFloat;
  mcclRedOp_t mccl_op = (op == OP_SUM) ? mcclSum : mcclSum;
  MCCLCHECK(mcclReduce(sendbuff, recvbuff, args->count, mccl_type, mccl_op, root,
                       args->comm, stream));
}

static void reduce_getBw(size_t sendBytes, size_t recvBytes, int nranks,
                         double timeSec, double *algbw, double *busbw) {
  (void)recvBytes;
  (void)nranks;
  if (timeSec <= 0.0) {
    *algbw = 0.0;
    *busbw = 0.0;
    return;
  }
  double bytes = (double)sendBytes;
  *algbw = bytes / timeSec / 1e9;
  *busbw = *algbw;
}

static int reduce_checkData(threadArgs_t *args, int root, DataType type,
                            RedOp op, int in_place) {
  (void)type;
  (void)op;
  (void)in_place;
  if (args->rank != root) {
    return 0;
  }
  const float expected = (float)(args->nranks * (args->nranks - 1) / 2);
  float host_result = -1.0f;
  MUSACHECK(musaMemcpy(&host_result, args->recvbuff, sizeof(float),
                       musaMemcpyDeviceToHost));
  if (host_result != expected) {
    printf("Rank %d: incorrect result %.0f (expected %.0f)\n", args->rank,
           host_result, expected);
    return 1;
  }
  return 0;
}

static size_t reduce_getInplaceOffset(size_t sendBytes, size_t recvBytes,
                                      int nranks, int rank) {
  (void)sendBytes;
  (void)recvBytes;
  (void)nranks;
  (void)rank;
  return 0;
}

testEngine_t mcclTestEngine = {
    .name = "reduce",
    .defaultSizeBytes = (1 << 20) * sizeof(float),
    .supportsInplace = 0,
    .defaultType = DATA_FLOAT,
    .defaultTypeName = "float",
    .defaultOp = OP_SUM,
    .defaultOpName = "sum",
    .defaultRoot = 0,
    .getBuffSize = reduce_getBuffSize,
    .initData = reduce_initData,
    .runTest = reduce_runTest,
    .getInplaceOffset = reduce_getInplaceOffset,
    .getBw = reduce_getBw,
    .checkData = reduce_checkData,
};
