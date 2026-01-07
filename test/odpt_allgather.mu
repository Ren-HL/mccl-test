// One-device-per-pthread AllGather example on mccl/musa (with timing)
#include "common.h"

void *thread_worker(void *arg) {
  threadData_t *data = (threadData_t *)arg;
  const int rank = data->rank;
  const int devicecount = data->devicecount;
  const size_t send_count = data->send_count;
  const size_t recv_count = data->recv_count;

  int warmup_iters = 5;
  int iters = 20;
  const char *warmEnv = getenv("WARMUP_ITERS");
  const char *iterEnv = getenv("ITERS");
  if (warmEnv) {
    int v = atoi(warmEnv);
    if (v > 0)
      warmup_iters = v;
  }
  if (iterEnv) {
    int v = atoi(iterEnv);
    if (v > 0)
      iters = v;
  }

  musaStream_t stream;
  MUSACHECK(musaSetDevice(rank));
  MUSACHECK(musaStreamCreate(&stream));
  musaEvent_t start, stop;
  MUSACHECK(musaEventCreate(&start));
  MUSACHECK(musaEventCreate(&stop));

  // Allocate device buffers
  MUSACHECK(musaMalloc((void **)&data->sendbuff[rank],
                       send_count * sizeof(float)));
  // AllGather output is devicecount * send_count floats
  MUSACHECK(musaMalloc((void **)&data->recvbuff[rank],
                       recv_count * sizeof(float)));

  // Seed send buffer with rank
  MUSACHECK(musaMemset(data->sendbuff[rank], 0, send_count * sizeof(float)));
  float rank_value = (float)rank;
  MUSACHECK(musaMemcpy(data->sendbuff[rank], &rank_value, sizeof(float),
                       musaMemcpyHostToDevice));

  // Initialize communicator
  MCCLCHECK(
      mcclCommInitRank(&data->comms[rank], devicecount, data->commId, rank));

  // Warmup iterations (not timed)
  for (int i = 0; i < warmup_iters; i++) {
    MCCLCHECK(mcclAllGather(data->sendbuff[rank], data->recvbuff[rank],
                            send_count, mcclFloat, data->comms[rank], stream));
  }
  MUSACHECK(musaStreamSynchronize(stream));

  // Timed iterations
  double total_ms = 0.0;
  for (int i = 0; i < iters; i++) {
    MUSACHECK(musaEventRecord(start, stream));
    MCCLCHECK(mcclAllGather(data->sendbuff[rank], data->recvbuff[rank],
                            send_count, mcclFloat, data->comms[rank], stream));
    MUSACHECK(musaEventRecord(stop, stream));
    MUSACHECK(musaEventSynchronize(stop));
    float ms = 0.0f;
    MUSACHECK(musaEventElapsedTime(&ms, start, stop));
    total_ms += (double)ms;
  }
  MUSACHECK(musaStreamSynchronize(stream));

  // Verify: first float of each chunk should equal its source rank
  int errors = 0;
  for (int r = 0; r < devicecount; r++) {
    float host_value = -1.0f;
    const size_t offset = (size_t)r * send_count;
    MUSACHECK(musaMemcpy(&host_value, data->recvbuff[rank] + offset,
                         sizeof(float), musaMemcpyDeviceToHost));
    if (host_value != (float)r) {
      errors++;
      printf("Rank %d: mismatch at chunk %d value %.0f expected %d\n", rank, r,
             host_value, r);
    }
  }

  double avg_ms = total_ms / (double)iters;
  double bytes = (double)recv_count * sizeof(float);
  // Approx effective bandwidth: output bytes per rank over time
  double gbps = bytes / (avg_ms / 1e3) / 1e9;
  printf("Rank %d: AllGather avg time %.3f ms over %d iters (warmup %d), approx bandwidth %.2f GB/s, errors %d\n",
         rank, avg_ms, iters, warmup_iters, gbps, errors);

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

  printf("Launching AllGather with %d devices (pthread per device)\n",
         devicecount);

  mcclComm_t *comms;
  float **sendbuff;
  float **recvbuff;
  pthread_t *threads;
  threadData_t *threadData;
  alloc_host_structs(devicecount, &comms, &sendbuff, &recvbuff, &threads,
                     &threadData);

  mcclUniqueId commId;
  MCCLCHECK(mcclGetUniqueId(&commId));

  // Per-rank contribution size; output is scaled by devicecount
  const size_t send_count = 1 << 18; // 256K floats
  const size_t recv_count = (size_t)devicecount * send_count;

  for (int r = 0; r < devicecount; r++) {
    threadData[r].rank = r;
    threadData[r].devicecount = devicecount;
    threadData[r].send_count = send_count;
    threadData[r].recv_count = recv_count;
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

  printf("AllGather example finished\n");
  return 0;
}
