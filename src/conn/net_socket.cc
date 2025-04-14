/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <assert.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <limits.h>

#include "conn/gccl_net.h"
#include "conn/ibvwrap.h"
#include "conn/socket.h"
#include "conn/topo.h"
#include "core.h"
#include "gccl.h"
#include "param.h"

#include "glog/logging.h"

namespace gccl {

/* Init functions */
static char gcclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];
static union socketAddress gcclNetIfAddrs[MAX_IFS];
static int gcclNetIfs = -1;
pthread_mutex_t gcclSocketLock = PTHREAD_MUTEX_INITIALIZER;

enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

gcclResult_t gcclSocketInit() {
  if (gcclNetIfs == -1) {
    pthread_mutex_lock(&gcclSocketLock);
    if (gcclNetIfs == -1) {
      gcclNetIfs = findInterfaces(gcclNetIfNames, gcclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (gcclNetIfs <= 0) {
        LOG(ERROR) << "NET/Socket : no interface found";
        return gcclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';
        for (int i=0; i<gcclNetIfs; i++) {
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, gcclNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&gcclNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';
        DLOG(INFO) << "NET/Socket : Using" << line;
      }
    }
    pthread_mutex_unlock(&gcclSocketLock);
  }
  return gcclSuccess;
}

gcclResult_t gcclSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = GCCL_PTR_HOST;
  return gcclSuccess;
}

gcclResult_t gcclSocketDevices(int* ndev, int** scores) {
  gcclSocketInit();
  *ndev = gcclNetIfs;

  int* sc;
  GCCLCalloc(&sc, gcclNetIfs);
  for (int i = 0; i < gcclNetIfs; i++) {
    sc[i] = 1;
  }
  *scores = sc;

  return gcclSuccess;
}

gcclResult_t gcclSocketPciPath(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/net/%s/device", gcclNetIfNames+dev*MAX_IF_NAME_SIZE);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    LOG(ERROR) << "Could not find real path of " << devicepath;
    return gcclSystemError;
  }
  return gcclSuccess;
}

static gcclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= gcclNetIfs) return gcclInternalError;
  memcpy(addr, gcclNetIfAddrs+dev, sizeof(*addr));
  return gcclSuccess;
}

/* Communication functions */

struct gcclSocketHandle {
  union socketAddress connectAddr;
};

struct gcclSocketRequest {
  int op;
  void* data;
  int size;
  int fd;
  int offset;
  int used;
};

struct gcclSocketReqs {
  struct gcclSocketRequest* requests;
};

struct gcclSocketComm {
  int fd;
  struct gcclSocketReqs reqs;
};

gcclResult_t gcclSocketNewComm(struct gcclSocketComm** comm) {
  GCCLCalloc(comm, 1);
  (*comm)->fd = -1;
  return gcclSuccess;
}

gcclResult_t gcclSocketCreateHandle(void* opaqueHandle, const char* str) {
  struct gcclSocketHandle* handle = (struct gcclSocketHandle*) opaqueHandle;
  GCCLCHECK(GetSocketAddrFromString(&(handle->connectAddr), str));
  return gcclSuccess;
}

gcclResult_t gcclSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  struct gcclSocketHandle* handle = (struct gcclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct gcclSocketHandle) < GCCL_NET_HANDLE_MAXSIZE, "gcclSocketHandle size too large");
  // if dev >= 0, listen based on dev
  if (dev >= 0) {
    GCCLCHECK(GetSocketAddr(dev, &(handle->connectAddr)));
  } else if (dev == findSubnetIf) {
    // handle stores a remote address
    // need to find a local addr that is in the same network as the remote addr
    union socketAddress localAddr;
    char ifName[MAX_IF_NAME_SIZE];
    if (findInterfaceMatchSubnet(ifName, &localAddr, handle->connectAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      LOG(ERROR) << "NET/Socket : No usable listening interface found";
      return gcclSystemError;
    }
    // pass the local address back
    memcpy(&handle->connectAddr, &localAddr, sizeof(handle->connectAddr));
  } // Otherwise, handle stores a local address
  struct gcclSocketComm* comm;
  GCCLCHECK(gcclSocketNewComm(&comm));
  GCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  *listenComm = comm;
  return gcclSuccess;
}

gcclResult_t gcclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct gcclSocketComm* comm;
  GCCLCHECK(gcclSocketNewComm(&comm));
  struct gcclSocketHandle* handle = (struct gcclSocketHandle*) opaqueHandle;
  GCCLCHECK(connectAddress(&comm->fd, &handle->connectAddr));
  *sendComm = comm;
  return gcclSuccess;
}

gcclResult_t gcclSocketAccept(void* listenComm, void** recvComm) {
  struct gcclSocketComm* lComm = (struct gcclSocketComm*)listenComm;
  struct gcclSocketComm* rComm;
  GCCLCHECK(gcclSocketNewComm(&rComm));
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  *recvComm = rComm;
  return gcclSuccess;
}

#define MAX_REQUESTS 128

gcclResult_t gcclSocketGetRequest(struct gcclSocketReqs* reqs, int op, void* data, int size, int fd, struct gcclSocketRequest** req) {
  if (reqs->requests == NULL) {
    GCCLCalloc(&reqs->requests, MAX_REQUESTS);
  }
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct gcclSocketRequest* r = reqs->requests+i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->fd = fd;
      r->offset = -1;
      r->used = 1;
      *req = r;
      return gcclSuccess;
    }
  }
  LOG(ERROR) << "Socket : unable to allocate requests";
  return gcclInternalError;
}

gcclResult_t gcclSocketTest(void* request, int* done, size_t* size) {
  *done = 0;
  struct gcclSocketRequest *r = (struct gcclSocketRequest*)request;
  if (r == NULL) {
    LOG(ERROR) << "Socket : test called with NULL request";
    return gcclInternalError;
  }
  if (r->offset == -1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    GCCLCHECK(socketProgress(r->op, r->fd, &data, sizeof(int), &offset));

    if (offset == 0) return gcclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) GCCLCHECK(socketWait(r->op, r->fd, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == GCCL_SOCKET_RECV && data > r->size) {
      LOG(WARNING) << "NET/Socket : message truncated : receiving " << data << " bytes instead of " << r->size;
      return gcclInternalError;
    }
    r->size = data;
    r->offset = 0;
  }
  if (r->offset < r->size) {
    GCCLCHECK(socketProgress(r->op, r->fd, r->data, r->size, &r->offset));
  }
  if (r->offset == r->size) {
    if (size) *size = r->size;
    *done = 1;
    r->used = 0;
  }
  return gcclSuccess;
}

gcclResult_t gcclSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return (type != GCCL_PTR_HOST) ? gcclInternalError : gcclSuccess;
}
gcclResult_t gcclSocketDeregMr(void* comm, void* mhandle) { return gcclSuccess; }

gcclResult_t gcclSocketIsend(void* sendComm, void* data, size_t size, int type, void** request) {
  struct gcclSocketComm* comm = (struct gcclSocketComm*)sendComm;
  GCCLCHECK(gcclSocketGetRequest(&comm->reqs, GCCL_SOCKET_SEND, data, size, comm->fd, (struct gcclSocketRequest**)request));
  assert(GCCL_PTR_HOST == type);
  return gcclSuccess;
}

gcclResult_t gcclSocketIrecv(void* recvComm, void* data, size_t size, int type, void** request) {
  struct gcclSocketComm* comm = (struct gcclSocketComm*)recvComm;
  GCCLCHECK(gcclSocketGetRequest(&comm->reqs, GCCL_SOCKET_RECV, data, size, comm->fd, (struct gcclSocketRequest**)request));
  assert(GCCL_PTR_HOST == type);
  return gcclSuccess;
}

gcclResult_t gcclSocketFlush(void* recvComm, void* data, size_t size) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return gcclInternalError;
}

gcclResult_t gcclSocketClose(void* opaqueComm) {
  struct gcclSocketComm* comm = (struct gcclSocketComm*)opaqueComm;
  if (comm) {
    free(comm->reqs.requests);
    close(comm->fd);
    free(comm);
  }
  return gcclSuccess;
}

gcclNet_t gcclNetSocket = {
  "Socket",
  gcclSocketDevices,
  gcclSocketPtrSupport,
  gcclSocketListen,
  gcclSocketConnect,
  gcclSocketAccept,
  gcclSocketIsend,
  gcclSocketIrecv,
  gcclSocketFlush,
  gcclSocketTest,
  gcclSocketClose,
  gcclSocketClose,
  gcclSocketClose
};

}  // namespace gccl