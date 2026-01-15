// AllReduce engine plugin (mccl-test)
#include "common.h"

static void allreduce_getBuffSize(size_t *sendBytes, size_t *recvBytes,
                                  size_t countBytes, int nranks) {
  (void)nranks;
  *sendBytes = countBytes;
  *recvBytes = countBytes;
}

static void allreduce_initData(threadArgs_t *args, int root, DataType type,
                               RedOp op, int in_place) {
  (void)root;
  (void)type;
  (void)op;
  // Seed first element with rank; rest zeros to match expected sum.
  MUSACHECK(musaMemset(args->sendbuff, 0, args->sendBytes));
  float rank_value = (float)args->rank;
  MUSACHECK(musaMemcpy(args->sendbuff, &rank_value, sizeof(float),
                       musaMemcpyHostToDevice));
  if (!in_place) {
    MUSACHECK(musaMemset(args->recvbuff, 0, args->recvBytes));
  }
}

static void allreduce_runTest(threadArgs_t *args, int root, DataType type,
                              RedOp op, const void *sendbuff, void *recvbuff,
                              musaStream_t stream) {
  (void)root;
  // Map engine enums to MCCL types/ops.
  mcclDataType_t mccl_type = mcclFloat;
  mcclRedOp_t mccl_op = (op == OP_SUM) ? mcclSum : mcclSum;
  (void)type;
  MCCLCHECK(mcclAllReduce(sendbuff, recvbuff, args->count, mccl_type, mccl_op,
                          args->comm, stream));
}

static void allreduce_getBw(size_t sendBytes, size_t recvBytes, int nranks,
                            double timeSec, double *algbw, double *busbw) {
  (void)recvBytes;
  (void)nranks;
  if (timeSec <= 0.0) {
    *algbw = 0.0;
    *busbw = 0.0;
    return;
  }
  // AllReduce moves 2 * payload per rank for algBW.
  double bytes = (double)sendBytes;
  *algbw = (2.0 * bytes) / timeSec / 1e9;
  *busbw = *algbw;
}

static int allreduce_checkData(threadArgs_t *args, int root, DataType type,
                               RedOp op, int in_place) {
  (void)root;
  (void)type;
  (void)op;
  // Expected sum of ranks in first element.
  const float expected = (float)(args->nranks * (args->nranks - 1) / 2);
  float host_result = -1.0f;
  (void)in_place;
  MUSACHECK(musaMemcpy(&host_result, args->recvbuff, sizeof(float),
                       musaMemcpyDeviceToHost));
  if (host_result != expected) {
    printf("Rank %d: incorrect result %.0f (expected %.0f)\n", args->rank,
           host_result, expected);
    return 1;
  }
  return 0;
}

static size_t allreduce_getInplaceOffset(size_t sendBytes, size_t recvBytes,
                                         int nranks, int rank) {
  (void)sendBytes;
  (void)recvBytes;
  (void)nranks;
  (void)rank;
  return 0;
}

testEngine_t mcclTestEngine = {
    .name = "allreduce",
    .defaultSizeBytes = (1 << 20) * sizeof(float),
    .supportsInplace = 1,
    .defaultType = DATA_FLOAT,
    .defaultTypeName = "float",
    .defaultOp = OP_SUM,
    .defaultOpName = "sum",
    .defaultRoot = 0,
    .getBuffSize = allreduce_getBuffSize,
    .initData = allreduce_initData,
    .runTest = allreduce_runTest,
    .getInplaceOffset = allreduce_getInplaceOffset,
    .getBw = allreduce_getBw,
    .checkData = allreduce_checkData,
};
