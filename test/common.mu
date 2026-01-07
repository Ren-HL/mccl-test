#include "common.h"

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

void alloc_host_structs(int devicecount, mcclComm_t **comms, float ***sendbuff,
                        float ***recvbuff, pthread_t **threads,
                        threadData_t **threadData) {
  *comms = (mcclComm_t *)malloc(devicecount * sizeof(mcclComm_t));
  *sendbuff = (float **)malloc(devicecount * sizeof(float *));
  *recvbuff = (float **)malloc(devicecount * sizeof(float *));
  *threads = (pthread_t *)malloc(devicecount * sizeof(pthread_t));
  *threadData = (threadData_t *)malloc(devicecount * sizeof(threadData_t));
}

void free_host_structs(mcclComm_t *comms, float **sendbuff, float **recvbuff,
                       pthread_t *threads, threadData_t *threadData) {
  free(comms);
  free(sendbuff);
  free(recvbuff);
  free(threads);
  free(threadData);
}
