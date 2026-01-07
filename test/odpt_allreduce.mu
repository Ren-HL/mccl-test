// One-device-per-pthread AllReduce example on mccl/musa
#include "common.h"

void *thread_worker(void *arg) {
  threadData_t *data = (threadData_t *)arg;
  const int rank = data->rank;
  const int devicecount = data->devicecount;
  const size_t count = data->send_count;
  const float expected = (float)(devicecount * (devicecount - 1) / 2);

  int warmup_iters = 5;
  int iters = 20;
  const char *warmEnv = getenv("WARMUP_ITERS");
  const char *iterEnv = getenv("ITERS");
  if (warmEnv) {
    int v = atoi(warmEnv);
    if (v > 0) warmup_iters = v;
  }
  if (iterEnv) {
    int v = atoi(iterEnv);
    if (v > 0) iters = v;
  }

  musaStream_t stream;
  MUSACHECK(musaSetDevice(rank));
  MUSACHECK(musaStreamCreate(&stream));
  musaEvent_t start, stop;
  MUSACHECK(musaEventCreate(&start));
  MUSACHECK(musaEventCreate(&stop));

  // Allocate device buffers and seed send buffer with rank
  MUSACHECK(musaMalloc((void **)&data->sendbuff[rank], count * sizeof(float)));
  MUSACHECK(musaMalloc((void **)&data->recvbuff[rank], count * sizeof(float)));
  MUSACHECK(
      musaMemset(data->sendbuff[rank], 0, count * sizeof(float)));
  float rank_value = (float)rank;
  MUSACHECK(musaMemcpy(data->sendbuff[rank], &rank_value, sizeof(float),
                       musaMemcpyHostToDevice));

  // Initialize communicator for this rank
  MCCLCHECK(mcclCommInitRank(&data->comms[rank], devicecount, data->commId, rank));

  // Warmup iterations (not timed)
  for (int i = 0; i < warmup_iters; i++) {
    MCCLCHECK(mcclAllReduce(data->sendbuff[rank], data->recvbuff[rank], count,
                            mcclFloat, mcclSum, data->comms[rank], stream));
  }
  MUSACHECK(musaStreamSynchronize(stream));

  // Timed iterations
  double total_ms = 0.0;
  for (int i = 0; i < iters; i++) {
    MUSACHECK(musaEventRecord(start, stream));
    MCCLCHECK(mcclAllReduce(data->sendbuff[rank], data->recvbuff[rank], count,
                            mcclFloat, mcclSum, data->comms[rank], stream));
    MUSACHECK(musaEventRecord(stop, stream));
    MUSACHECK(musaEventSynchronize(stop));
    float ms = 0.0f;
    MUSACHECK(musaEventElapsedTime(&ms, start, stop));
    total_ms += (double)ms;
  }
  MUSACHECK(musaStreamSynchronize(stream));

  // Verify first element (from last iteration)
  float host_result = -1.0f;
  MUSACHECK(musaMemcpy(&host_result, data->recvbuff[rank], sizeof(float),
                       musaMemcpyDeviceToHost));
  if (host_result != expected) {
    printf("Rank %d: incorrect result %.0f (expected %.0f)\n", rank,
           host_result, expected);
  } else {
    printf("Rank %d: correct result %.0f\n", rank, host_result);
  }

  double avg_ms = total_ms / (double)iters;
  double bytes = (double)count * sizeof(float);
  double gbps = (2.0 * bytes) / (avg_ms / 1e3) / 1e9;
  printf("Rank %d: avg time %.3f ms over %d iters (warmup %d), approx bandwidth %.2f GB/s\n",
         rank, avg_ms, iters, warmup_iters, gbps);

  MCCLCHECK(mcclCommDestroy(data->comms[rank]));
  MUSACHECK(musaFree(data->sendbuff[rank]));
  MUSACHECK(musaFree(data->recvbuff[rank]));
  MUSACHECK(musaStreamDestroy(stream));
  MUSACHECK(musaEventDestroy(start));
  MUSACHECK(musaEventDestroy(stop));

  return NULL;
}

int main() {
  int devicecount = get_device_count();
  devicecount = apply_env_override(devicecount);

  if (devicecount < 1) {
    printf("No musa devices found\n");
    return EXIT_FAILURE;
  }

  printf("Launching AllReduce with %d devices (pthread per device)\n", devicecount);

  mcclComm_t *comms;
  float **sendbuff;
  float **recvbuff;
  pthread_t *threads;
  threadData_t *threadData;
  alloc_host_structs(devicecount, &comms, &sendbuff, &recvbuff, &threads,
                     &threadData);

  mcclUniqueId commId;
  MCCLCHECK(mcclGetUniqueId(&commId));

  // Keep buffer size modest so the example runs quickly
  const size_t count = 1 << 20; // 1M floats

  for (int r = 0; r < devicecount; r++) {
    threadData[r].rank = r;
    threadData[r].devicecount = devicecount;
    threadData[r].send_count = count;
    threadData[r].recv_count = count;
    threadData[r].commId = commId;
    threadData[r].comms = comms;
    threadData[r].sendbuff = sendbuff;
    threadData[r].recvbuff = recvbuff;
    pthread_create(&threads[r], NULL, thread_worker, &threadData[r]);
  }

  for (int r = 0; r < devicecount; r++) {
    pthread_join(threads[r], NULL);
  }

  free_host_structs(comms, sendbuff, recvbuff, threads, threadData);

  printf("AllReduce example finished\n");
  return 0;
}
