#ifndef __COMMON_H__
#define __COMMON_H__

#pragma once

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "musa_runtime.h"
#include "mccl.h"

#define MUSACHECK(cmd)                                                         \
  do {                                                                         \
    musaError_t err = cmd;                                                     \
    if (err != musaSuccess) {                                                  \
      printf("Failed: musa error %s:%d '%s'\n", __FILE__, __LINE__,            \
             musaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define MCCLCHECK(cmd)                                                         \
  do {                                                                         \
    mcclResult_t res = cmd;                                                    \
    if (res != mcclSuccess) {                                                  \
      printf("Failed, MCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             mcclGetErrorString(res));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

typedef struct {
  int rank;
  int devicecount;
  size_t send_count; // number of floats sent by this rank
  size_t recv_count; // number of floats received per rank buffer
  mcclUniqueId commId;
  mcclComm_t *comms;
  float **sendbuff;
  float **recvbuff;
} threadData_t;

int get_device_count();
int apply_env_override(int devicecount);
void alloc_host_structs(int devicecount, mcclComm_t **comms, float ***sendbuff,
                        float ***recvbuff, pthread_t **threads,
                        threadData_t **threadData);
void free_host_structs(mcclComm_t *comms, float **sendbuff, float **recvbuff,
                       pthread_t *threads, threadData_t *threadData);

#endif