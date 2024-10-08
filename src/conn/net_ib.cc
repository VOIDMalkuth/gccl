/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//#include "gccl.h"
//#include "core.h"
//#include "net.h"
//#include "param.h"
//#include "socket.h"
//#include "topo.h"
//#include "utils.h"

#include <assert.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>

#include "conn/gccl_net.h"
#include "conn/ibvwrap.h"
#include "conn/socket.h"
#include "conn/topo.h"
#include "core.h"
#include "gccl.h"
#include "param.h"

#include "glog/logging.h"

namespace gccl {

#define USE_RDMA_WRITE 1
#define USE_RDMA_SEND_INLINE 0
#define MAXNAMESIZE 64

static char gcclIbIfName[MAX_IF_NAME_SIZE];
static union socketAddress gcclIbIfAddr;
static int gcclNIbDevs = -1;
struct gcclIbDev {
  int device;
  uint8_t port;
  ibv_context* context;
  char devName[MAXNAMESIZE];
};

#define MAX_IB_PORT 15
struct userIbDev {
  char devName[MAXNAMESIZE];
  uint16_t port_en;
};

#define MAX_IB_DEVS 16
struct gcclIbDev gcclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];
pthread_mutex_t gcclIbLock = PTHREAD_MUTEX_INITIALIZER;

GCCL_PARAM(IbGidIndex, "IB_GID_INDEX", 0);
GCCL_PARAM(IbTimeout, "IB_TIMEOUT", 14);
GCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
GCCL_PARAM(IbSl, "IB_SL", 0);
GCCL_PARAM(IbTc, "IB_TC", 0);

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
static gcclResult_t gcclIbMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  size_t size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return gcclSystemError;
  memset(p, 0, size);
  *ptr = p;
  return gcclSuccess;
}

pthread_t gcclIbAsyncThread;
static void* gcclIbAsyncThreadMain(void* args) {
  struct ibv_context* context = (struct ibv_context*)args;
  while (1) {
    struct ibv_async_event event;
    if (gcclSuccess != wrap_ibv_get_async_event(context, &event)) {
      break;
    }
    char* str;
    if (gcclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) {
      break;
    }
    if (event.event_type != IBV_EVENT_COMM_EST)
      LOG(WARNING) << "NET/IB : Got async event : " << str;
    if (gcclSuccess != wrap_ibv_ack_async_event(&event)) {
      break;
    }
  }
  return NULL;
}

static void initDevices() {
  if (wrap_ibv_symbols() != gcclSuccess) {
    return;
  }
  if (gcclNIbDevs == -1) {
    pthread_mutex_lock(&gcclIbLock);
    wrap_ibv_fork_init();
    if (gcclNIbDevs == -1) {
      gcclNIbDevs = 0;
      if (findInterfaces(gcclIbIfName, &gcclIbIfAddr, MAX_IF_NAME_SIZE, 1) !=
          1) {
        DLOG(ERROR) << "NET/IB : No IP interface found.";
        return;
      }
      DLOG(INFO) << "NET/IB : Using interface " << gcclIbIfName
                << " for sideband communication";

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      char* userIbEnv = getenv("GCCL_IB_HCA");
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (gcclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) return;

      for (int d = 0; d < nIbDevs; d++) {
        struct ibv_context* context;
        if (gcclSuccess != wrap_ibv_open_device(&context, devices[d])) {
          LOG(ERROR) << "NET/IB : Unable to open device " << devices[d]->name;
          continue;
        }
        int found = 0;
        if (context) {
          struct ibv_device_attr devAttr;
          if (gcclSuccess != wrap_ibv_query_device(context, &devAttr)) {
            LOG(ERROR) << "NET/IB : Unable to query device "
                       << devices[d]->name;
            continue;
          }
          for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
            struct ibv_port_attr portAttr;
            if (gcclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
              LOG(ERROR) << "NET/IB : Unable to query port " << port;
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND &&
                portAttr.link_layer != IBV_LINK_LAYER_ETHERNET)
              continue;

            // check against user specified HCAs/ports
            if (!(matchIfList(devices[d]->name, port, userIfs, nUserIfs) ^
                  searchNot)) {
              continue;
            }
            DLOG(INFO) << "NET/IB: [" << d << "] " << devices[d]->name << ":"
                      << port << "/"
                      << (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND
                              ? "IB"
                              : "RoCE");
            gcclIbDevs[gcclNIbDevs].device = d;
            gcclIbDevs[gcclNIbDevs].port = port;
            gcclIbDevs[gcclNIbDevs].context = context;
            strncpy(gcclIbDevs[gcclNIbDevs].devName, devices[d]->name,
                    MAXNAMESIZE);
            gcclNIbDevs++;
            found++;
            pthread_create(&gcclIbAsyncThread, NULL, gcclIbAsyncThreadMain,
                           context);
          }

          if (found == 0) {
            if (gcclSuccess != wrap_ibv_close_device(context)) {
              return;
            }
          }
        }
      }
      if (nIbDevs && (gcclSuccess != wrap_ibv_free_device_list(devices))) {
        return;
      };
    }

    pthread_mutex_unlock(&gcclIbLock);
  }
}

gcclResult_t gcclIbDevices(int* ndev, int** scores) {
  initDevices();
  *ndev = gcclNIbDevs;
  int cudaDev;
  cudaGetDevice(&cudaDev);
  char* cudaPath;
  gcclResult_t err1 = getCudaPath(cudaDev, &cudaPath);
  int* sc;
  GCCLCalloc(&sc, gcclNIbDevs);
  char line[1024];
  sprintf(line, "CUDA Dev %d, IB Ports : ", cudaDev);
  for (int d = 0; d < gcclNIbDevs; d++) {
    char* mlxPath;
    gcclResult_t err2 = getMlxPath(gcclIbDevs[d].devName, &mlxPath);
    int distance = (err1 != gcclSuccess || err2 != gcclSuccess ||
                    mlxPath == NULL || cudaPath == NULL)
                       ? PATH_SOC
                       : pciDistance(mlxPath, cudaPath);
    sprintf(line + strlen(line), "%s/%d(%s) ", gcclIbDevs[d].devName,
            gcclIbDevs[d].port, pathDists[distance]);
    sc[d] = 1 + PATH_SOC - distance;
    if (err2 == gcclSuccess) free(mlxPath);
  }
  DLOG(INFO) << line;
  if (err1 == gcclSuccess) free(cudaPath);
  *scores = sc;
  return gcclSuccess;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// gcclSuccess : GDR works
// gcclSystemError : no module or module loaded but not supported by GPU
gcclResult_t gcclIbGdrSupport(int ibDev) {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    moduleLoaded =
        (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) ? 0
                                                                           : 1;
    DLOG(INFO) << "Cannot load nv_mem module";
  }
  if (moduleLoaded == 0) return gcclSystemError;
  gcclResult_t ret = gcclSystemError;
  void* ptr;
  if (cudaMalloc(&ptr, sizeof(int)) == cudaSuccess) {
    struct ibv_mr* mr;
    struct ibv_pd* pd;
    if (wrap_ibv_alloc_pd(&pd, gcclIbDevs[ibDev].context) == gcclSuccess) {
      if ((mr = wrap_direct_ibv_reg_mr(pd, ptr, sizeof(int),
                                       IBV_ACCESS_LOCAL_WRITE |
                                           IBV_ACCESS_REMOTE_WRITE |
                                           IBV_ACCESS_REMOTE_READ)) != NULL) {
        ret = gcclSuccess;
        wrap_ibv_dereg_mr(mr);
      }
      wrap_ibv_dealloc_pd(pd);
    }
    cudaFree(ptr);
  }
  return ret;
}

GCCL_PARAM(IbGdrLevel, "IB_GDR_LEVEL", -2);
GCCL_PARAM(IbCudaSupport, "IB_CUDA_SUPPORT", -2);

gcclResult_t gcclIbPtrSupport(int dev, int* supportedTypes) {
  initDevices();
  *supportedTypes = GCCL_PTR_HOST;

  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return gcclSuccess;

  int ibGdrLevel = PATH_PHB;
  if (gcclParamIbCudaSupport() != -2)
    ibGdrLevel = gcclParamIbCudaSupport() ? PATH_SOC + 1 : 0;
  if (gcclParamIbGdrLevel() != -2) ibGdrLevel = gcclParamIbGdrLevel();
  if (ibGdrLevel > 0) {
    int gdrSupport = gcclIbGdrSupport(dev);
    if (gdrSupport > 0) {
      DLOG(INFO) << "NET/IB : GPU Direct RDMA Disabled for GPU " << cudaDev
                << " / HCA " << gcclIbDevs[dev].devName << ":"
                << (gdrSupport == 1 ? "no module" : "not supported by GPU");
      ibGdrLevel = 0;
    }
  }

  if (ibGdrLevel <= 0) return gcclSuccess;

  char* cudaPath;
  if (getCudaPath(cudaDev, &cudaPath) != gcclSuccess) return gcclSuccess;
  char* mlxPath;
  if (getMlxPath(gcclIbDevs[dev].devName, &mlxPath) != gcclSuccess) {
    free(cudaPath);
    return gcclSuccess;
  }
  int distance = (mlxPath == NULL || cudaPath == NULL)
                     ? PATH_SOC
                     : pciDistance(mlxPath, cudaPath);
  free(mlxPath);
  free(cudaPath);
  if (distance < ibGdrLevel) {
    *supportedTypes |= GCCL_PTR_CUDA;
  } else {
    DLOG(INFO) << "NET/IB : GPU Direct RDMA Disabled for GPU " << cudaDev
              << " / HCA " << gcclIbDevs[dev].devName << "(distance "
              << distance << ">= " << ibGdrLevel << ")";
  }
  return gcclSuccess;
}

static gcclResult_t GetSocketAddr(union socketAddress* addr) {
  if (gcclNIbDevs == -1) initDevices();
  memcpy(addr, &gcclIbIfAddr, sizeof(*addr));
  return gcclSuccess;
}

#define MAX_REQUESTS 128

struct gcclIbQpInfo {
  uint32_t lid;
  uint8_t ib_port;
  uint32_t qpn;

  // For RoCE
  uint64_t spn;
  uint64_t iid;
  enum ibv_mtu mtu;

  // FIFO RDMA info
  uint32_t fifoRkey;
  uint64_t fifoAddr;
};

struct gcclIbHandle {
  union socketAddress connectAddr;
};

struct gcclIbMr {
  struct ibv_mr* mr;
  int refcnt;
};

struct gcclIbVerbs {
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  struct gcclIbMr mrPool[MAX_REQUESTS];
  int mrRotation;
};

struct gcclIbRequest {
  int used;
  int type;
  struct gcclIbVerbs* verbs;
  struct gcclIbMr* ibMr;
  int done;
  int size;
  int free;
};

struct gcclIbListenComm {
  int dev;
  int fd;
};

struct gcclIbSendFifo {
  uint64_t addr;
  int size;
  uint32_t seq;
  uint32_t rkey;
  uint32_t ready;
};

struct gcclIbSendComm {
  struct gcclIbSendFifo fifo[MAX_REQUESTS];
  struct gcclIbRequest reqs[MAX_REQUESTS];
  uint32_t fifoHead;
  int fd;
  int ready;
  struct gcclIbVerbs verbs;
  struct ibv_qp* qp;
  struct ibv_mr* fifoMr;
};

struct gcclIbGpuFlush {
  int enabled;
  int hostMem;
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ibv_qp* qp;
};

struct gcclIbRemFifo {
  struct gcclIbSendFifo elems[MAX_REQUESTS];
  uint64_t addr;
  uint32_t rkey;
  uint32_t tail;
  uint32_t flags;
  struct ibv_mr* mr;
  struct ibv_sge sge;
};

struct gcclIbRecvComm {
  struct gcclIbRemFifo remFifo;
  struct gcclIbRequest reqs[MAX_REQUESTS];
  int fd;
  int ready;
  struct gcclIbVerbs verbs;
  struct ibv_qp* qp;
  struct gcclIbGpuFlush gpuFlush;
};

gcclResult_t gcclIbInitVerbs(ibv_context* ctx, struct gcclIbVerbs* verbs) {
  GCCLCHECK(wrap_ibv_alloc_pd(&verbs->pd, ctx));
  GCCLCHECK(wrap_ibv_create_cq(&verbs->cq, ctx, MAX_REQUESTS, NULL, NULL, 0));
  return gcclSuccess;
}

gcclResult_t gcclIbDestroyVerbs(struct gcclIbVerbs* verbs) {
  GCCLCHECK(wrap_ibv_destroy_cq(verbs->cq));
  GCCLCHECK(wrap_ibv_dealloc_pd(verbs->pd));
  return gcclSuccess;
}

gcclResult_t gcclIbCreateQp(uint8_t ib_port, struct gcclIbVerbs* verbs,
                            int access_flags, struct ibv_qp** qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = verbs->cq;
  qpInitAttr.recv_cq = verbs->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;
  GCCLCHECK(wrap_ibv_create_qp(qp, verbs->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  GCCLCHECK(wrap_ibv_modify_qp(
      *qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return gcclSuccess;
}

gcclResult_t gcclIbRtrQp(ibv_qp* qp, struct gcclIbQpInfo* info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = info->qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->lid == 0) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = gcclParamIbGidIndex();
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = gcclParamIbTc();
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = gcclParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  GCCLCHECK(wrap_ibv_modify_qp(
      qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return gcclSuccess;
}

gcclResult_t gcclIbRtsQp(ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = gcclParamIbTimeout();
  qpAttr.retry_cnt = gcclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  GCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr,
                               IBV_QP_STATE | IBV_QP_TIMEOUT |
                                   IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                   IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return gcclSuccess;
}

gcclResult_t gcclIbListen(int dev, void* opaqueHandle, void** listenComm) {
  struct gcclIbListenComm* comm;
  GCCLCalloc(&comm, 1);
  struct gcclIbHandle* handle = (struct gcclIbHandle*)opaqueHandle;
  static_assert(sizeof(struct gcclIbHandle) < GCCL_NET_HANDLE_MAXSIZE,
                "gcclIbHandle size too large");
  comm->dev = dev;
  GCCLCHECK(GetSocketAddr(&(handle->connectAddr)));
  GCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  *listenComm = comm;
  return gcclSuccess;
}

gcclResult_t gcclIbConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct gcclIbSendComm* comm;
  GCCLCHECK(gcclIbMalloc((void**)&comm, sizeof(struct gcclIbSendComm)));

  struct gcclIbHandle* handle = (struct gcclIbHandle*)opaqueHandle;
  GCCLCHECK(connectAddress(&comm->fd, &handle->connectAddr));
  *sendComm = comm;

  // IB Setup
  initDevices(); /*NOTE: We need to do this for gcclNet unit test that bypasses
                    gccl initialization*/
  ibv_context* ctx = gcclIbDevs[dev].context;
  GCCLCHECK(gcclIbInitVerbs(ctx, &comm->verbs));
  uint8_t ib_port = gcclIbDevs[dev].port;
  GCCLCHECK(gcclIbCreateQp(ib_port, &comm->verbs, IBV_ACCESS_REMOTE_WRITE,
                           &comm->qp));

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  GCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  struct gcclIbQpInfo qpInfo;
  qpInfo.ib_port = ib_port;
  qpInfo.qpn = comm->qp->qp_num;
  qpInfo.mtu = portAttr.active_mtu;

  // Prepare my fifo
  GCCLCHECK(wrap_ibv_reg_mr(&comm->fifoMr, comm->verbs.pd, comm->fifo,
                            sizeof(struct gcclIbSendFifo) * MAX_REQUESTS,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ));
  qpInfo.fifoRkey = comm->fifoMr->rkey;
  qpInfo.fifoAddr = (uint64_t)comm->fifo;

  // RoCE support
  qpInfo.lid = portAttr.lid;
  if (qpInfo.lid) {  // IB
    DLOG(INFO) << "NET/IB: Dev " << dev << " Port " << ib_port << " qpn "
              << qpInfo.qpn << "mtu " << qpInfo.mtu << " LID " << qpInfo.lid;
  } else {  // RoCE
    union ibv_gid gid;
    GCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, gcclParamIbGidIndex(), &gid));
    qpInfo.spn = gid.global.subnet_prefix;
    qpInfo.iid = gid.global.interface_id;
    DLOG(INFO) << "NET/IB: Dev " << dev << " Port " << ib_port << " qpn "
              << qpInfo.qpn << "mtu " << qpInfo.mtu << " GID "
              << gcclParamIbGidIndex() << "(" << qpInfo.spn << "/" << qpInfo.iid
              << ")";
  }

  GCCLCHECK(socketSend(comm->fd, &qpInfo, sizeof(qpInfo)));
  return gcclSuccess;
}

GCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

gcclResult_t gcclIbAccept(void* listenComm, void** recvComm) {
  struct gcclIbListenComm* lComm = (struct gcclIbListenComm*)listenComm;
  struct gcclIbRecvComm* rComm;
  GCCLCHECK(gcclIbMalloc((void**)&rComm, sizeof(struct gcclIbRecvComm)));

  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen),
              "accept", rComm->fd);
  struct gcclIbQpInfo remQpInfo;
  GCCLCHECK(socketReceive(rComm->fd, &remQpInfo, sizeof(remQpInfo)));

  // IB setup
  ibv_context* ctx = gcclIbDevs[lComm->dev].context;
  uint8_t ib_port = gcclIbDevs[lComm->dev].port;
  struct ibv_port_attr portAttr;
  GCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  union ibv_gid gid;
  GCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, gcclParamIbGidIndex(), &gid));

  // QP Creation
  GCCLCHECK(gcclIbInitVerbs(ctx, &rComm->verbs));
  GCCLCHECK(gcclIbCreateQp(ib_port, &rComm->verbs, IBV_ACCESS_REMOTE_WRITE,
                           &rComm->qp));

  // Adjust the MTU
  remQpInfo.mtu = (enum ibv_mtu)std::min(remQpInfo.mtu, portAttr.active_mtu);

  // Setup QP
  struct ibv_qp* qp = rComm->qp;
  GCCLCHECK(gcclIbRtrQp(qp, &remQpInfo));
  GCCLCHECK(gcclIbRtsQp(qp));

  // Retain remote fifo info and prepare my RDMA ops
  rComm->remFifo.rkey = remQpInfo.fifoRkey;
  rComm->remFifo.addr = remQpInfo.fifoAddr;
  GCCLCHECK(wrap_ibv_reg_mr(&rComm->remFifo.mr, rComm->verbs.pd,
                            &rComm->remFifo.elems,
                            sizeof(struct gcclIbSendFifo) * MAX_REQUESTS,
                            IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                                IBV_ACCESS_REMOTE_READ));
  rComm->remFifo.sge.length = sizeof(struct gcclIbSendFifo);
  rComm->remFifo.sge.lkey = rComm->remFifo.mr->lkey;

#if USE_RDMA_SEND_INLINE
  // Determine whether the remFifo element data can be sent INLINE
  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init_attr;
  GCCLCHECK(wrap_ibv_query_qp(qp, &attr, IBV_QP_CAP, &init_attr));
  if (init_attr.cap.max_inline_data >= rComm->remFifo.sge.length)
    rComm->remFifo.flags = IBV_SEND_INLINE;
#endif

  // Allocate Flush dummy buffer for GPU Direct RDMA
  rComm->gpuFlush.enabled =
      (gcclIbGdrSupport(lComm->dev) == 0) && (gcclParamIbGdrFlushDisable() == 0)
          ? 1
          : 0;
  if (rComm->gpuFlush.enabled) {
    GCCLCHECK(wrap_ibv_reg_mr(&rComm->gpuFlush.hostMr, rComm->verbs.pd,
                              &rComm->gpuFlush.hostMem, sizeof(int),
                              IBV_ACCESS_LOCAL_WRITE));
    rComm->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlush.hostMem;
    rComm->gpuFlush.sge.length = 1;
    rComm->gpuFlush.sge.lkey = rComm->gpuFlush.hostMr->lkey;
    GCCLCHECK(gcclIbCreateQp(ib_port, &rComm->verbs,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
                             &rComm->gpuFlush.qp));
    struct gcclIbQpInfo localQpInfo = {.lid = portAttr.lid,
                                       .ib_port = ib_port,
                                       .qpn = rComm->gpuFlush.qp->qp_num,
                                       .spn = gid.global.subnet_prefix,
                                       .iid = gid.global.interface_id,
                                       .mtu = portAttr.active_mtu};
    GCCLCHECK(gcclIbRtrQp(rComm->gpuFlush.qp, &localQpInfo));
    GCCLCHECK(gcclIbRtsQp(rComm->gpuFlush.qp));
  }

  // Fill Handle
  struct gcclIbQpInfo qpInfo = {.lid = portAttr.lid,
                                .ib_port = ib_port,
                                .qpn = qp->qp_num,
                                .spn = gid.global.subnet_prefix,
                                .iid = gid.global.interface_id,
                                .mtu = remQpInfo.mtu};

  GCCLCHECK(socketSend(rComm->fd, &qpInfo, sizeof(qpInfo)));
  *recvComm = rComm;
  return gcclSuccess;
}

gcclResult_t gcclIbGetRequest(struct gcclIbRequest* reqs,
                              struct gcclIbRequest** req) {
  for (int i = 0; i < MAX_REQUESTS; i++) {
    struct gcclIbRequest* r = reqs + i;
    if (r->used == 0) {
      r->used = 1;
      r->type = 0;
      r->verbs = NULL;
      r->ibMr = NULL;
      r->done = 0;
      r->size = -1;
      r->free = 0;
      *req = r;
      return gcclSuccess;
    }
  }
  LOG(ERROR) << "NET/IB : unable to allocate requests";
  *req = NULL;
  return gcclInternalError;
}

gcclResult_t gcclSendCheck(struct gcclIbSendComm* comm) {
  if (comm->ready == 0) {
    struct gcclIbQpInfo remQpInfo;
    struct ibv_qp* qp = comm->qp;
    GCCLCHECK(socketReceive(comm->fd, &remQpInfo, sizeof(remQpInfo)));
    GCCLCHECK(gcclIbRtrQp(qp, &remQpInfo));
    GCCLCHECK(gcclIbRtsQp(qp));
    int go = 1;
    GCCLCHECK(socketSend(comm->fd, &go, sizeof(go)));
    comm->ready = 1;
  }
  return gcclSuccess;
}

gcclResult_t gcclRecvCheck(struct gcclIbRecvComm* comm) {
  if (comm->ready == 0) {
    int go;
    GCCLCHECK(socketReceive(comm->fd, &go, sizeof(go)));
    comm->ready = 1;
  }
  return gcclSuccess;
}

gcclResult_t gcclIbTest(void* request, int* done, size_t* size);

#define REG_ALIGN (4096)

// Cache previous MRs to avoid registering/unregistering for each Isend/Irecv
gcclResult_t gcclIbGetMr(struct gcclIbVerbs* verbs, void* data, size_t size,
                         struct gcclIbMr** mrRet) {
  uint64_t addr = (uint64_t)data;
  int elem = -1;
  assert(size > 0);

  // Look for an already existing MR
  for (int i = 0; i < MAX_REQUESTS; i++) {
    if (verbs->mrPool[i].mr == NULL) continue;
    uint64_t regAddr = (uint64_t)verbs->mrPool[i].mr->addr;
    uint64_t regSize = (uint64_t)verbs->mrPool[i].mr->length;
    if (regAddr <= addr && addr + size <= regAddr + regSize) {
      *mrRet = verbs->mrPool + i;
      verbs->mrPool[i].refcnt++;
      return gcclSuccess;
    }
  }

  // Find an unused element
  if (elem == -1) {
    elem = (verbs->mrRotation++);
    for (int i = 0; i < MAX_REQUESTS; i++) {
      elem %= MAX_REQUESTS;
      if (verbs->mrPool[elem].refcnt > 0)
        elem++;
      else
        break;
    }
    if (verbs->mrPool[elem].refcnt > 0) {
      LOG(ERROR) << "NET/IB : memory register : no MR available";
      return gcclInternalError;
    }
  }

  assert(elem < MAX_REQUESTS);
  assert(verbs->mrPool[elem].refcnt == 0);

  // Deregister / register
  uint64_t regAddr = addr & (~(REG_ALIGN - 1));
  uint64_t regSize = addr + size - regAddr;
  regSize = ((regSize + REG_ALIGN - 1) / REG_ALIGN) * REG_ALIGN;
  if (verbs->mrPool[elem].mr)
    GCCLCHECK(wrap_ibv_dereg_mr(verbs->mrPool[elem].mr));
  GCCLCHECK(wrap_ibv_reg_mr(&verbs->mrPool[elem].mr, verbs->pd, (void*)regAddr,
                            regSize,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ));
  *mrRet = verbs->mrPool + elem;
  verbs->mrPool[elem].refcnt++;
  DLOG(INFO) << "elem " << elem << " regAddr " << regAddr << " size " << regSize << " rkey " << (verbs->mrPool + elem)->mr->rkey;
  return gcclSuccess;
}

gcclResult_t gcclIbIsend(void* sendComm, void* data, size_t size, int type,
                         void** request) {
  struct gcclIbSendComm* comm = (struct gcclIbSendComm*)sendComm;
  GCCLCHECK(gcclSendCheck(comm));

  struct gcclIbRequest* req;
  GCCLCHECK(gcclIbGetRequest(comm->reqs, &req));
  req->type = type;
  req->verbs = &comm->verbs;
  assert(size <= INT32_MAX);
  req->size = size;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  struct ibv_sge sge;
  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
  } else {
    GCCLCHECK(gcclIbGetMr(&comm->verbs, data, size, &req->ibMr));
    sge.addr = (uintptr_t)data;
    sge.length = (unsigned int)size;
    sge.lkey = req->ibMr->mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  // Wait for receiver to have posted the recv
  volatile struct gcclIbSendFifo* slot =
      comm->fifo + (comm->fifoHead % MAX_REQUESTS);
  volatile uint32_t* readyPtr = &slot->ready;
  while (*readyPtr == 0) sched_yield();
#if USE_RDMA_WRITE
  __sync_synchronize();  // order the readyPtr load against rkey load below
  // Sanity checks to catch user collective call count/size mismatches
  // plus any potential programming errors
  if (size > slot->size || slot->size <= 0 || slot->addr == 0 ||
      slot->rkey == 0 || slot->seq != comm->fifoHead) {
    LOG(ERROR) << "NET/IB : collective mismatch error local size " << size << " remote " << slot->size << 
                  " addr  " << slot->addr << 
                  " rkey " << slot->rkey << " seq " << slot->seq << "/" << comm->fifoHead;
    return gcclInternalError;
  }
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.wr.rdma.remote_addr = slot->addr;
  wr.wr.rdma.rkey = slot->rkey;
  wr.imm_data = size;  // Send the message size via imm_data
  __sync_synchronize();
#endif
  // We must clear slot->ready, but reset other fields to aid
  // debugging and sanity checks
  slot->ready = 0;
  slot->addr = 0ULL;
  slot->rkey = slot->size = slot->seq = 0;
  comm->fifoHead++;

  struct ibv_send_wr* bad_wr;
  GCCLCHECK(wrap_ibv_post_send(comm->qp, &wr, &bad_wr));
  *request = req;
  return gcclSuccess;
}

gcclResult_t gcclIbPostFifo(struct gcclIbRecvComm* comm, uint32_t rkey,
                            uint64_t addr, size_t size) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct gcclIbRequest* req;
  GCCLCHECK(gcclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->verbs;
  req->free = 1;  // Not a user req ; free as soon as it is complete.
  wr.wr_id = (uint64_t)req;

  struct gcclIbSendFifo* localElem =
      comm->remFifo.elems + (comm->remFifo.tail % MAX_REQUESTS);
  localElem->addr = addr;
  localElem->rkey = rkey;
  localElem->ready = 1;
  assert(size <= INT32_MAX);  // Sanity/Debugging
  localElem->size = size;               // Sanity/Debugging
  localElem->seq = comm->remFifo.tail;  // Sanity/Debugging
  wr.wr.rdma.remote_addr =
      comm->remFifo.addr +
      (comm->remFifo.tail % MAX_REQUESTS) * sizeof(struct gcclIbSendFifo);
  wr.wr.rdma.rkey = comm->remFifo.rkey;
  comm->remFifo.sge.addr = (uint64_t)localElem;
  wr.sg_list = &comm->remFifo.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED | comm->remFifo.flags;  // IBV_SEND_INLINE

  struct ibv_send_wr* bad_wr;
  GCCLCHECK(wrap_ibv_post_send(comm->qp, &wr, &bad_wr));
  comm->remFifo.tail++;

  return gcclSuccess;
}

gcclResult_t gcclIbIrecv(void* recvComm, void* data, size_t size, int type,
                         void** request) {
  struct gcclIbRecvComm* comm = (struct gcclIbRecvComm*)recvComm;
  GCCLCHECK(gcclRecvCheck(comm));

  struct gcclIbRequest* req;
  GCCLCHECK(gcclIbGetRequest(comm->reqs, &req));
  req->type = type;
  req->verbs = &comm->verbs;
  req->size = size;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  struct ibv_sge sge;
  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
    req->ibMr = NULL;
  } else {
    GCCLCHECK(gcclIbGetMr(&comm->verbs, data, size, &req->ibMr));
    sge.addr = (uintptr_t)data;
    assert(size <= INT32_MAX);
    sge.length = (unsigned int)size;
    sge.lkey = req->ibMr->mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }

  struct ibv_recv_wr* bad_wr;
  GCCLCHECK(wrap_ibv_post_recv(comm->qp, &wr, &bad_wr));
  *request = req;

  // Post to FIFO to notify sender
  GCCLCHECK(gcclIbPostFifo(comm, req->ibMr->mr->rkey, (uint64_t)data, size));
  return gcclSuccess;
}

gcclResult_t gcclIbFlush(void* recvComm, void* data, size_t size) {
  struct gcclIbRecvComm* comm = (struct gcclIbRecvComm*)recvComm;
  if (comm->gpuFlush.enabled == 0 || size == 0) return gcclSuccess;

  struct gcclIbRequest* req;
  GCCLCHECK(gcclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->verbs;
  GCCLCHECK(gcclIbGetMr(&comm->verbs, data, 1, &req->ibMr));

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  wr.wr.rdma.remote_addr = (uint64_t)data;
  wr.wr.rdma.rkey = req->ibMr->mr->rkey;
  wr.sg_list = &comm->gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  GCCLCHECK(wrap_ibv_post_send(comm->gpuFlush.qp, &wr, &bad_wr));

  int done = 0;
  while (done == 0) {
    GCCLCHECK((gcclResult_t)gcclIbTest(req, &done, NULL));
  }

  return gcclSuccess;
}

gcclResult_t gcclIbTest(void* request, int* done, size_t* size) {
  struct gcclIbRequest* r = (struct gcclIbRequest*)request;
  *done = 0;

  while (1) {
    if (r->done == 1) {
      *done = 1;
      if (size) *size = r->size;
      r->used = 0;
      return gcclSuccess;
    }

    int wrDone = 0;
    struct ibv_wc wc;
    GCCLCHECK(wrap_ibv_poll_cq(r->verbs->cq, 1, &wc, &wrDone));
    if (wrDone == 0) return gcclSuccess;

    if (wc.status != IBV_WC_SUCCESS) {
      LOG(ERROR)
          << "NET/IB : Got completion with error %d, opcode %d, len %d, vendor "
             "err %d",
          wc.status << wc.opcode << wc.byte_len << wc.vendor_err;
      return gcclSystemError;
    }

    struct gcclIbRequest* doneReq = (struct gcclIbRequest*)wc.wr_id;
    if (doneReq) {
      if (wc.opcode == IBV_WC_RECV) {
        doneReq->size = wc.byte_len;
#if USE_RDMA_WRITE
      } else if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        doneReq->size = wc.imm_data;
#endif
      }
      if (doneReq->ibMr != NULL) {
        doneReq->ibMr->refcnt--;
        if (doneReq->ibMr->refcnt < 0)
          LOG(ERROR) << "NET/IB : doneReq %p MR %p refcount now %d" << doneReq
                     << doneReq->ibMr << doneReq->ibMr->refcnt;
      }
      doneReq->done = 1;
      if (doneReq->free == 1) {
        // This is an internal (FIFO post) req. Free it immediately.
        doneReq->used = 0;
      }
    }
  }
}

gcclResult_t gcclIbCloseSend(void* sendComm) {
  struct gcclIbSendComm* comm = (struct gcclIbSendComm*)sendComm;
  if (comm) {
    close(comm->fd);
    if (comm->qp != NULL) GCCLCHECK(wrap_ibv_destroy_qp(comm->qp));
    if (comm->fifoMr != NULL) GCCLCHECK(wrap_ibv_dereg_mr(comm->fifoMr));
    for (int i = 0; i < MAX_REQUESTS; i++) {
      if (comm->verbs.mrPool[i].mr != NULL) {
        if (comm->verbs.mrPool[i].refcnt != 0)
          LOG(ERROR) << "NET/IB : TX MR #%d has non-zero (%d) refcnt" << i
                     << comm->verbs.mrPool[i].refcnt;
        GCCLCHECK(wrap_ibv_dereg_mr(comm->verbs.mrPool[i].mr));
      }
    }
    GCCLCHECK(gcclIbDestroyVerbs(&comm->verbs));
    free(comm);
  }
  return gcclSuccess;
}

gcclResult_t gcclIbCloseRecv(void* recvComm) {
  struct gcclIbRecvComm* comm = (struct gcclIbRecvComm*)recvComm;
  if (comm) {
    close(comm->fd);
    if (comm->qp != NULL) GCCLCHECK(wrap_ibv_destroy_qp(comm->qp));
    if (comm->gpuFlush.enabled) {
      if (comm->gpuFlush.qp != NULL)
        GCCLCHECK(wrap_ibv_destroy_qp(comm->gpuFlush.qp));
      if (comm->gpuFlush.hostMr != NULL)
        GCCLCHECK(wrap_ibv_dereg_mr(comm->gpuFlush.hostMr));
    }
    if (comm->remFifo.mr != NULL)
      GCCLCHECK(wrap_ibv_dereg_mr(comm->remFifo.mr));
    for (int i = 0; i < MAX_REQUESTS; i++) {
      if (comm->verbs.mrPool[i].mr != NULL) {
        if (comm->verbs.mrPool[i].refcnt != 0)
          LOG(ERROR) << "NET/IB : RX MR #%d has non-zero (%d) refcnt" << i
                     << comm->verbs.mrPool[i].refcnt;
        GCCLCHECK(wrap_ibv_dereg_mr(comm->verbs.mrPool[i].mr));
      }
    }
    GCCLCHECK(gcclIbDestroyVerbs(&comm->verbs));
    free(comm);
  }
  return gcclSuccess;
}

gcclResult_t gcclIbCloseListen(void* listenComm) {
  struct gcclIbListenComm* comm = (struct gcclIbListenComm*)listenComm;
  if (comm) {
    close(comm->fd);
    free(comm);
  }
  return gcclSuccess;
}

gcclNet_t gcclNetIb = {"IB",
                       gcclIbDevices,
                       gcclIbPtrSupport,
                       gcclIbListen,
                       gcclIbConnect,
                       gcclIbAccept,
                       gcclIbIsend,
                       gcclIbIrecv,
                       gcclIbFlush,
                       gcclIbTest,
                       gcclIbCloseSend,
                       gcclIbCloseRecv,
                       gcclIbCloseListen};

GCCL_PARAM(IbDisable, "IB_DISABLE", 0);

bool gcclIbSupport() {
  if (gcclParamIbDisable()) return 0;
  initDevices();
  return gcclNIbDevs > 0;
}

}  // namespace gccl
