#include "common.h"

static int get_env_int(const char *name, int default_value) {
  const char *env = getenv(name);
  if (!env || env[0] == '\0') {
    return default_value;
  }
  int v = atoi(env);
  return (v > 0) ? v : default_value;
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

static size_t parse_size(const char *value, size_t default_value) {
  if (!value || value[0] == '\0') {
    return default_value;
  }
  unsigned long long v = strtoull(value, NULL, 10);
  return (v > 0) ? (size_t)v : default_value;
}

static void usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  -b, --minbytes <bytes>\n");
  printf("  -e, --maxbytes <bytes>\n");
  printf("  -f, --stepfactor <factor>\n");
  printf("  -i, --stepbytes <bytes>\n");
  printf("  -n, --iters <iters>\n");
  printf("  -w, --warmup_iters <iters>\n");
  printf("  -m, --agg_iters <iters>\n");
  printf("  -c, --check <0|1>\n");
  printf("  -z, --blocking [0|1]\n");
  printf("      --inplace\n");
  printf("      --noinplace\n");
  printf("  -r, --root <rank>\n");
}

void parseArgs(int argc, char **argv, testArgs_t *args,
               const testEngine_t *engine) {
  args->minBytes = engine->defaultSizeBytes;
  args->maxBytes = engine->defaultSizeBytes;
  args->stepFactor = 2.0;
  args->stepBytes = 0;
  args->warmup_iters = 5;
  args->iters = 20;
  args->agg_iters = 1;
  args->check = 1;
  args->blocking = 0;
  args->inplace_mode = engine->supportsInplace ? 2 : 0;
  args->root = get_env_int("ROOT", engine->defaultRoot);

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-b") || !strcmp(arg, "--minbytes")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->minBytes = parse_size(argv[++i], args->minBytes);
    } else if (!strcmp(arg, "-e") || !strcmp(arg, "--maxbytes")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->maxBytes = parse_size(argv[++i], args->maxBytes);
    } else if (!strcmp(arg, "-f") || !strcmp(arg, "--stepfactor")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->stepFactor = atof(argv[++i]);
    } else if (!strcmp(arg, "-i") || !strcmp(arg, "--stepbytes")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->stepBytes = parse_size(argv[++i], args->stepBytes);
    } else if (!strcmp(arg, "-n") || !strcmp(arg, "--iters")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->iters = atoi(argv[++i]);
    } else if (!strcmp(arg, "-w") || !strcmp(arg, "--warmup_iters")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->warmup_iters = atoi(argv[++i]);
    } else if (!strcmp(arg, "-m") || !strcmp(arg, "--agg_iters")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->agg_iters = atoi(argv[++i]);
    } else if (!strcmp(arg, "-c") || !strcmp(arg, "--check")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->check = atoi(argv[++i]);
    } else if (!strcmp(arg, "-z") || !strcmp(arg, "--blocking")) {
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        args->blocking = atoi(argv[++i]) ? 1 : 0;
      } else {
        args->blocking = 1;
      }
    } else if (!strcmp(arg, "--inplace")) {
      args->inplace_mode = 1;
    } else if (!strcmp(arg, "--noinplace")) {
      args->inplace_mode = 0;
    } else if (!strcmp(arg, "-r") || !strcmp(arg, "--root")) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
      }
      args->root = atoi(argv[++i]);
    } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    } else {
      printf("Unknown argument: %s\n", arg);
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if (args->minBytes == 0) {
    args->minBytes = get_type_size(engine->defaultType);
  }
  if (args->maxBytes < args->minBytes) {
    args->maxBytes = args->minBytes;
  }
  if (args->iters < 1) {
    args->iters = 1;
  }
  if (args->agg_iters < 1) {
    args->agg_iters = 1;
  }
  if (args->stepBytes == 0 && args->stepFactor <= 1.0) {
    args->stepFactor = 2.0;
  }
}

void setupArgsForSize(testArgs_t *args, const testEngine_t *engine,
                      size_t sizeBytes) {
  size_t sendBytes = 0;
  size_t recvBytes = 0;
  size_t typeSize = get_type_size(args->type);
  engine->getBuffSize(&sendBytes, &recvBytes, sizeBytes, args->nranks);
  if (sendBytes < typeSize) {
    sendBytes = typeSize;
  }
  args->sizeBytes = sendBytes;
  args->sendBytes = sendBytes;
  args->recvBytes = recvBytes;
  // sizeBytes -> count(elements), using effective sendBytes after alignment.
  args->count = sendBytes / typeSize;
}

static size_t next_size(const testArgs_t *args, size_t cur) {
  // Step by fixed bytes if provided; otherwise multiply by factor.
  if (args->stepBytes > 0) {
    return cur + args->stepBytes;
  }
  size_t next = (size_t)((double)cur * args->stepFactor);
  if (next <= cur) {
    next = cur + 1;
  }
  return next;
}

static sizeInfo_t *build_size_list(const testArgs_t *args,
                                   const testEngine_t *engine, int *nSizes,
                                   size_t *maxSend, size_t *maxRecv) {
  size_t cur = args->minBytes;
  size_t maxBytes = args->maxBytes;
  size_t typeSize = get_type_size(args->type);
  int cap = 16;
  int count = 0;
  sizeInfo_t *list = (sizeInfo_t *)malloc(cap * sizeof(sizeInfo_t));

  *maxSend = 0;
  *maxRecv = 0;

  while (cur <= maxBytes) {
    testArgs_t tmp = *args;
    setupArgsForSize(&tmp, engine, cur);
    sizeInfo_t info;
    info.sizeBytes = tmp.sizeBytes;
    info.sendBytes = tmp.sendBytes;
    info.recvBytes = tmp.recvBytes;
    info.count = tmp.count;

    if (info.count == 0) {
      info.count = 1;
      info.sendBytes = typeSize;
      info.recvBytes = typeSize;
      info.sizeBytes = typeSize;
    }

    if (count >= cap) {
      cap *= 2;
      list = (sizeInfo_t *)realloc(list, cap * sizeof(sizeInfo_t));
    }
    list[count++] = info;

    if (info.sendBytes > *maxSend) {
      *maxSend = info.sendBytes;
    }
    if (info.recvBytes > *maxRecv) {
      *maxRecv = info.recvBytes;
    }

    cur = next_size(args, cur);
  }

  *nSizes = count;
  return list;
}

static void startColl(threadArgs_t *targs) {
  targs->engine->runTest(targs, targs->test->root, targs->test->type,
                         targs->test->op, targs->sendbuff, targs->recvbuff,
                         targs->stream);
}

static void completeColl(threadArgs_t *targs) {
  MUSACHECK(musaStreamSynchronize(targs->stream));
}

static void timeTest(threadArgs_t *targs) {
  const int warmup_iters = targs->test->warmup_iters;
  const int iters = targs->test->iters;
  const int agg_iters = targs->test->agg_iters;
  const int blocking = targs->test->blocking;
  double total_ms = 0.0;

  // Warmup loop (not timed).
  for (int i = 0; i < warmup_iters; i++) {
    for (int j = 0; j < agg_iters; j++) {
      startColl(targs);
      if (blocking) {
        completeColl(targs);
      }
    }
    if (!blocking) {
      completeColl(targs);
    }
  }

  // Timed loop; average per collective (us).
  for (int i = 0; i < iters; i++) {
    MUSACHECK(musaEventRecord(targs->start, targs->stream));
    for (int j = 0; j < agg_iters; j++) {
      startColl(targs);
      if (blocking) {
        completeColl(targs);
      }
    }
    if (!blocking) {
      completeColl(targs);
    }
    MUSACHECK(musaEventRecord(targs->stop, targs->stream));
    MUSACHECK(musaEventSynchronize(targs->stop));
    float ms = 0.0f;
    MUSACHECK(musaEventElapsedTime(&ms, targs->start, targs->stop));
    total_ms += (double)ms;
  }

  targs->avg_us = (total_ms * 1000.0) / (double)(iters * agg_iters);
  // Convert average time to bandwidth (seconds for getBw).
  targs->engine->getBw(targs->sendBytes, targs->recvBytes, targs->nranks,
                       targs->avg_us / 1e6, &targs->algbw, &targs->busbw);
}

static void initThreadResources(threadArgs_t *targs) {
  MUSACHECK(musaSetDevice(targs->rank));
  MUSACHECK(musaStreamCreate(&targs->stream));
  MUSACHECK(musaEventCreate(&targs->start));
  MUSACHECK(musaEventCreate(&targs->stop));

  targs->base_sendbuff = NULL;
  targs->base_recvbuff = NULL;
  targs->send_is_alias = 0;

  // Allocate once for max size to avoid per-iteration malloc.
  MUSACHECK(musaMalloc(&targs->base_recvbuff, targs->test->maxRecvBytes));
  if (!targs->in_place) {
    MUSACHECK(musaMalloc(&targs->base_sendbuff, targs->test->maxSendBytes));
  }

  MCCLCHECK(mcclCommInitRank(&targs->comms[targs->rank], targs->nranks,
                             targs->commId, targs->rank));
  targs->comm = targs->comms[targs->rank];
}

static void destroyThreadResources(threadArgs_t *targs) {
  if (targs->comm) {
    MCCLCHECK(mcclCommDestroy(targs->comm));
  }
  if (targs->base_sendbuff) {
    MUSACHECK(musaFree(targs->base_sendbuff));
  }
  if (targs->base_recvbuff) {
    MUSACHECK(musaFree(targs->base_recvbuff));
  }
  MUSACHECK(musaEventDestroy(targs->start));
  MUSACHECK(musaEventDestroy(targs->stop));
  MUSACHECK(musaStreamDestroy(targs->stream));
}

static void configureBuffers(threadArgs_t *targs, const sizeInfo_t *info) {
  targs->sendBytes = info->sendBytes;
  targs->recvBytes = info->recvBytes;
  targs->count = info->count;
  targs->recvbuff = targs->base_recvbuff;
  if (targs->in_place) {
    // In-place uses a subregion of recv as send.
    targs->in_place_offset = 0;
    if (targs->engine->getInplaceOffset) {
      targs->in_place_offset =
          targs->engine->getInplaceOffset(targs->sendBytes, targs->recvBytes,
                                          targs->nranks, targs->rank);
    }
    if (targs->in_place_offset + targs->sendBytes > targs->recvBytes) {
      printf("Rank %d: invalid in-place offset %zu for recvBytes %zu\n",
             targs->rank, targs->in_place_offset, targs->recvBytes);
      exit(EXIT_FAILURE);
    }
    targs->sendbuff =
        (char *)targs->base_recvbuff + targs->in_place_offset;
  } else {
    targs->sendbuff = targs->base_sendbuff;
  }
}

static void *thread_main(void *arg) {
  threadArgs_t *targs = (threadArgs_t *)arg;
  initThreadResources(targs);

  // Iterate over all sizes in the sweep.
  for (int i = 0; i < targs->nSizes; i++) {
    configureBuffers(targs, &targs->sizes[i]);
    targs->engine->initData(targs, targs->test->root, targs->test->type,
                            targs->test->op, targs->in_place);
    timeTest(targs);

    if (targs->test->check && targs->engine->checkData) {
      // Reinitialize and run one collective for correctness validation.
      targs->engine->initData(targs, targs->test->root, targs->test->type,
                              targs->test->op, targs->in_place);
      startColl(targs);
      completeColl(targs);
      targs->errors = targs->engine->checkData(
          targs, targs->test->root, targs->test->type, targs->test->op,
          targs->in_place);
    } else {
      targs->errors = 0;
    }

    targs->errors_by_rank[targs->rank] = targs->errors;
    pthread_barrier_wait(targs->barrier);

    if (targs->rank == 0) {
      int sum = 0;
      for (int r = 0; r < targs->nranks; r++) {
        sum += targs->errors_by_rank[r];
      }
      targs->time_us[i] = targs->avg_us;
      targs->algbw_out[i] = targs->algbw;
      targs->busbw_out[i] = targs->busbw;
      targs->errors_out[i] = sum;
    }
    pthread_barrier_wait(targs->barrier);
  }

  destroyThreadResources(targs);
  return NULL;
}

static void run_mode(const testEngine_t *engine, testArgs_t *test, int in_place,
                     const sizeInfo_t *sizes, int nSizes, double *time_us,
                     double *algbw, double *busbw, int *errors) {
  mcclComm_t *comms = (mcclComm_t *)malloc(test->nranks * sizeof(mcclComm_t));
  pthread_t *threads =
      (pthread_t *)malloc(test->nranks * sizeof(pthread_t));
  threadArgs_t *targs =
      (threadArgs_t *)malloc(test->nranks * sizeof(threadArgs_t));
  int *errors_by_rank = (int *)malloc(test->nranks * sizeof(int));
  pthread_barrier_t barrier;

  pthread_barrier_init(&barrier, NULL, test->nranks);

  mcclUniqueId commId;
  MCCLCHECK(mcclGetUniqueId(&commId));

  for (int r = 0; r < test->nranks; r++) {
    targs[r].rank = r;
    targs[r].nranks = test->nranks;
    targs[r].sendBytes = 0;
    targs[r].recvBytes = 0;
    targs[r].count = 0;
    targs[r].in_place = in_place;
    targs[r].in_place_offset = 0;
    targs[r].send_is_alias = 0;
    targs[r].commId = commId;
    targs[r].comms = comms;
    targs[r].comm = NULL;
    targs[r].base_sendbuff = NULL;
    targs[r].base_recvbuff = NULL;
    targs[r].sendbuff = NULL;
    targs[r].recvbuff = NULL;
    targs[r].sizes = sizes;
    targs[r].nSizes = nSizes;
    targs[r].time_us = time_us;
    targs[r].algbw_out = algbw;
    targs[r].busbw_out = busbw;
    targs[r].errors_out = errors;
    targs[r].errors_by_rank = errors_by_rank;
    targs[r].barrier = &barrier;
    targs[r].engine = engine;
    targs[r].test = test;
    targs[r].errors = 0;
    pthread_create(&threads[r], NULL, thread_main, &targs[r]);
  }

  for (int r = 0; r < test->nranks; r++) {
    pthread_join(threads[r], NULL);
  }

  pthread_barrier_destroy(&barrier);
  free(errors_by_rank);
  free(comms);
  free(threads);
  free(targs);
}

static void print_header(const testEngine_t *engine, int want_oop, int want_ip) {
  printf("# mccl-test %s\n", engine->name);
  if (want_oop && want_ip) {
    printf("%12s %12s %8s %8s %6s %12s %12s %12s %12s %12s %12s %8s\n",
           "size(B)", "count", "type", "op", "root", "oop_time(us)",
           "oop_algBW", "oop_busBW", "ip_time(us)", "ip_algBW", "ip_busBW",
           "errors");
  } else if (want_oop) {
    printf("%12s %12s %8s %8s %6s %12s %12s %12s %8s\n", "size(B)",
           "count", "type", "op", "root", "time(us)", "algBW", "busBW",
           "errors");
  } else {
    printf("%12s %12s %8s %8s %6s %12s %12s %12s %8s\n", "size(B)",
           "count", "type", "op", "root", "ip_time(us)", "ip_algBW",
           "ip_busBW", "errors");
  }
}

int main(int argc, char **argv) {
  const testEngine_t *engine = &mcclTestEngine;

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

  parseArgs(argc, argv, &test, engine);

  int nSizes = 0;
  size_t maxSendBytes = 0;
  size_t maxRecvBytes = 0;
  sizeInfo_t *sizes = build_size_list(&test, engine, &nSizes, &maxSendBytes,
                                      &maxRecvBytes);
  test.maxSendBytes = maxSendBytes;
  test.maxRecvBytes = maxRecvBytes;

  const int want_ip = (test.inplace_mode != 0) && engine->supportsInplace;
  const int want_oop = (test.inplace_mode != 1) || !engine->supportsInplace;

  print_header(engine, want_oop, want_ip);

  double *oop_time = NULL;
  double *oop_alg = NULL;
  double *oop_bus = NULL;
  int *oop_err = NULL;
  double *ip_time = NULL;
  double *ip_alg = NULL;
  double *ip_bus = NULL;
  int *ip_err = NULL;

  if (want_oop) {
    oop_time = (double *)calloc(nSizes, sizeof(double));
    oop_alg = (double *)calloc(nSizes, sizeof(double));
    oop_bus = (double *)calloc(nSizes, sizeof(double));
    oop_err = (int *)calloc(nSizes, sizeof(int));
    run_mode(engine, &test, 0, sizes, nSizes, oop_time, oop_alg, oop_bus,
             oop_err);
  }
  if (want_ip) {
    ip_time = (double *)calloc(nSizes, sizeof(double));
    ip_alg = (double *)calloc(nSizes, sizeof(double));
    ip_bus = (double *)calloc(nSizes, sizeof(double));
    ip_err = (int *)calloc(nSizes, sizeof(int));
    run_mode(engine, &test, 1, sizes, nSizes, ip_time, ip_alg, ip_bus, ip_err);
  }

  for (int i = 0; i < nSizes; i++) {
    if (want_oop && want_ip) {
      printf("%12zu %12zu %8s %8s %6d %12.3f %12.2f %12.2f %12.3f %12.2f %12.2f %8d\n",
             sizes[i].sizeBytes, sizes[i].count, test.typeName, test.opName,
             test.root, oop_time[i], oop_alg[i], oop_bus[i], ip_time[i],
             ip_alg[i], ip_bus[i], oop_err[i] + ip_err[i]);
    } else if (want_oop) {
      printf("%12zu %12zu %8s %8s %6d %12.3f %12.2f %12.2f %8d\n",
             sizes[i].sizeBytes, sizes[i].count, test.typeName, test.opName,
             test.root, oop_time[i], oop_alg[i], oop_bus[i], oop_err[i]);
    } else {
      printf("%12zu %12zu %8s %8s %6d %12.3f %12.2f %12.2f %8d\n",
             sizes[i].sizeBytes, sizes[i].count, test.typeName, test.opName,
             test.root, ip_time[i], ip_alg[i], ip_bus[i], ip_err[i]);
    }
  }

  free(oop_time);
  free(oop_alg);
  free(oop_bus);
  free(oop_err);
  free(ip_time);
  free(ip_alg);
  free(ip_bus);
  free(ip_err);
  free(sizes);

  return 0;
}
