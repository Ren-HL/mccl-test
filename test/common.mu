#include "common.h"

static int get_env_int(const char *name, int default_value) {
  const char *env = getenv(name);
  if (!env || env[0] == '\0') {
    return default_value;
  }
  int v = atoi(env);
  return (v > 0) ? v : default_value;
}

static size_t get_env_size_t(const char *name, size_t default_value) {
  const char *env = getenv(name);
  if (!env || env[0] == '\0') {
    return default_value;
  }
  unsigned long long v = strtoull(env, NULL, 10);
  return (v > 0) ? (size_t)v : default_value;
}

int get_device_count() {
  int devicecount = 0;
  MUSACHECK(musaGetDeviceCount(&devicecount));
  return devicecount;
}

int apply_env_override(int devicecount) {
  const char *nThreadsEnv = getenv("NTHREADS");
  if (nThreadsEnv) {
    int override = atoi(nThreadsEnv);
    if (override > 0 && override <= devicecount) {
      devicecount = override;
    }
  }
  return devicecount;
}

size_t get_type_size(DataType type) {
  switch (type) {
  case DATA_FLOAT:
  default:
    return sizeof(float);
  }
}

void setupArgs(testArgs_t *args, const testEngine_t *engine) {
  const size_t typeSize = get_type_size(args->type);
  args->count = args->sizeBytes / typeSize;
  engine->getBuffSize(&args->sendBytes, &args->recvBytes, args->sizeBytes,
                      args->nranks);
}

static void startColl(threadArgs_t *targs, int in_place) {
  const void *send = in_place ? targs->recvbuff : targs->sendbuff;
  void *recv = targs->recvbuff;
  // In-place uses recv buffer as input; out-of-place uses send buffer.
  targs->engine->runTest(targs, targs->test->root, targs->test->type,
                         targs->test->op, send, recv, targs->stream);
}

static void completeColl(threadArgs_t *targs) {
  MUSACHECK(musaStreamSynchronize(targs->stream));
}

static void timeTest(threadArgs_t *targs, int in_place) {
  const int warmup_iters = targs->test->warmup_iters;
  const int iters = targs->test->iters;
  double total_ms = 0.0;

  // Warmup: enqueue collectives without timing.
  for (int i = 0; i < warmup_iters; i++) {
    startColl(targs, in_place);
  }
  completeColl(targs);

  // Timed iterations using events for device time.
  for (int i = 0; i < iters; i++) {
    MUSACHECK(musaEventRecord(targs->start, targs->stream));
    startColl(targs, in_place);
    MUSACHECK(musaEventRecord(targs->stop, targs->stream));
    MUSACHECK(musaEventSynchronize(targs->stop));
    float ms = 0.0f;
    MUSACHECK(musaEventElapsedTime(&ms, targs->start, targs->stop));
    total_ms += (double)ms;
  }
  completeColl(targs);

  targs->avg_ms = total_ms / (double)iters;
  // Convert average time into algorithmic/bus bandwidth via engine callback.
  targs->engine->getBw(targs->sendBytes, targs->recvBytes, targs->nranks,
                       targs->avg_ms / 1e3, &targs->algbw, &targs->busbw);
}

static void initThreadResources(threadArgs_t *targs) {
  // Each thread owns one device, stream, events, and per-rank buffers.
  MUSACHECK(musaSetDevice(targs->rank));
  MUSACHECK(musaStreamCreate(&targs->stream));
  MUSACHECK(musaEventCreate(&targs->start));
  MUSACHECK(musaEventCreate(&targs->stop));

  MUSACHECK(musaMalloc(&targs->sendbuff, targs->sendBytes));
  MUSACHECK(musaMalloc(&targs->recvbuff, targs->recvBytes));

  MCCLCHECK(mcclCommInitRank(&targs->comms[targs->rank], targs->nranks,
                             targs->commId, targs->rank));
  targs->comm = targs->comms[targs->rank];
}

static void destroyThreadResources(threadArgs_t *targs) {
  if (targs->comm) {
    MCCLCHECK(mcclCommDestroy(targs->comm));
  }
  MUSACHECK(musaFree(targs->sendbuff));
  MUSACHECK(musaFree(targs->recvbuff));
  MUSACHECK(musaEventDestroy(targs->start));
  MUSACHECK(musaEventDestroy(targs->stop));
  MUSACHECK(musaStreamDestroy(targs->stream));
}

static void *thread_main(void *arg) {
  threadArgs_t *targs = (threadArgs_t *)arg;
  initThreadResources(targs);

  // Engine-specific data init and timed run.
  targs->engine->initData(targs, targs->test->root, targs->test->type,
                          targs->test->op, targs->in_place);
  timeTest(targs, targs->in_place);

  // Optional validation path.
  if (targs->test->check && targs->engine->checkData) {
    targs->errors = targs->engine->checkData(
        targs, targs->test->root, targs->test->type, targs->test->op,
        targs->in_place);
  } else {
    targs->errors = 0;
  }

  destroyThreadResources(targs);
  return NULL;
}

static void run_mode(const testEngine_t *engine, testArgs_t *test, int in_place,
                     double *avg_ms, double *algbw, double *busbw,
                     int *total_errors) {
  mcclComm_t *comms = (mcclComm_t *)malloc(test->nranks * sizeof(mcclComm_t));
  pthread_t *threads =
      (pthread_t *)malloc(test->nranks * sizeof(pthread_t));
  threadArgs_t *targs =
      (threadArgs_t *)malloc(test->nranks * sizeof(threadArgs_t));

  mcclUniqueId commId;
  MCCLCHECK(mcclGetUniqueId(&commId));

  // One thread per device/rank.
  for (int r = 0; r < test->nranks; r++) {
    targs[r].rank = r;
    targs[r].nranks = test->nranks;
    targs[r].sendBytes = test->sendBytes;
    targs[r].recvBytes = test->recvBytes;
    targs[r].count = test->count;
    targs[r].in_place = in_place;
    targs[r].commId = commId;
    targs[r].comms = comms;
    targs[r].comm = NULL;
    targs[r].sendbuff = NULL;
    targs[r].recvbuff = NULL;
    targs[r].engine = engine;
    targs[r].test = test;
    targs[r].errors = 0;
    pthread_create(&threads[r], NULL, thread_main, &targs[r]);
  }

  int errors = 0;
  for (int r = 0; r < test->nranks; r++) {
    pthread_join(threads[r], NULL);
    errors += targs[r].errors;
  }

  // Use rank 0 timing as the representative measurement.
  *avg_ms = targs[0].avg_ms;
  *algbw = targs[0].algbw;
  *busbw = targs[0].busbw;
  *total_errors = errors;

  free(comms);
  free(threads);
  free(targs);
}

static void print_header(const testEngine_t *engine) {
  printf("# mccl-test %s\n", engine->name);
  // Keep output columns consistent with nccl-tests-style summaries.
  if (engine->supportsInplace) {
    printf("%12s %12s %8s %8s %6s %12s %12s %12s %12s %12s %12s %8s\n",
           "size(B)", "count", "type", "op", "root", "oop_time(ms)",
           "oop_algBW", "oop_busBW", "ip_time(ms)", "ip_algBW", "ip_busBW",
           "errors");
  } else {
    printf("%12s %12s %8s %8s %6s %12s %12s %12s %8s\n", "size(B)",
           "count", "type", "op", "root", "time(ms)", "algBW", "busBW",
           "errors");
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  const testEngine_t *engine = &mcclTestEngine;

  // Global test args (size/type/op) shared across all ranks.
  int nranks = get_device_count();
  nranks = apply_env_override(nranks);
  if (nranks < 1) {
    printf("No musa devices found\n");
    return EXIT_FAILURE;
  }

  testArgs_t test;
  memset(&test, 0, sizeof(test));
  test.nranks = nranks;
  test.type = engine->defaultType;
  test.typeName = engine->defaultTypeName;
  test.op = engine->defaultOp;
  test.opName = engine->defaultOpName;
  test.root = engine->defaultRoot;
  test.warmup_iters = get_env_int("WARMUP_ITERS", 5);
  test.iters = get_env_int("ITERS", 20);
  test.check = get_env_int("DATACHECK", 1);
  test.sizeBytes = get_env_size_t("SIZE_BYTES", engine->defaultSizeBytes);

  setupArgs(&test, engine);

  print_header(engine);

  double oop_ms = 0.0, oop_algbw = 0.0, oop_busbw = 0.0;
  double ip_ms = 0.0, ip_algbw = 0.0, ip_busbw = 0.0;
  int oop_errors = 0, ip_errors = 0;

  run_mode(engine, &test, 0, &oop_ms, &oop_algbw, &oop_busbw, &oop_errors);

  if (engine->supportsInplace) {
    run_mode(engine, &test, 1, &ip_ms, &ip_algbw, &ip_busbw, &ip_errors);
    printf("%12zu %12zu %8s %8s %6d %12.3f %12.2f %12.2f %12.3f %12.2f %12.2f %8d\n",
           test.sizeBytes, test.count, test.typeName, test.opName, test.root,
           oop_ms, oop_algbw, oop_busbw, ip_ms, ip_algbw, ip_busbw,
           oop_errors + ip_errors);
  } else {
    printf("%12zu %12zu %8s %8s %6d %12.3f %12.2f %12.2f %8d\n",
           test.sizeBytes, test.count, test.typeName, test.opName, test.root,
           oop_ms, oop_algbw, oop_busbw, oop_errors);
  }

  return 0;
}
