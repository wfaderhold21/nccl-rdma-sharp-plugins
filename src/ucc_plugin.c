#include "config.h"
#include "core.h"
#include "p2p_plugin.h"
#include "param.h"
#include "utils.h"
#include "nccl.h"
#include "ucp/api/ucp.h"
#include "ucc/api/ucc.h"

#define UCC_ERROR(format, ...) \
          fprintf(stderr, "UCC ERROR: %s:%d - %s() " format, __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)

extern ncclNet_v8_t ncclNetPlugin_v8;
extern ncclNet_v7_t ncclNetPlugin_v7;
extern ncclNet_v6_t ncclNetPlugin_v6;
extern ncclNet_v5_t ncclNetPlugin_v5;
int ncclNUCCDevs = -1;

struct ncclUCCMemHandle{
  int  type;
};

struct ncclUCCInfo {
  uint64_t hostId;
  uint64_t jobId;
};

struct ncclUCCListenComm {
    int dev;
    void *listenCommP2P;
};

typedef struct ncclUCCRequest {
    uint64_t id;
    void *dst;
    size_t len;
    void *keys;
    void *req_h;
} request_t;

size_t map_beg = 0;
size_t map_end = 0;
mapped_mem_t map_array[128];

int num_outstanding = 0;

struct ncclUCCCollComm {
  int    rank;
  int    nranks;
  void*  recvComm;
  void*  sendComm;
  ucc_lib_h       ucc_lib;
  ucc_context_h   ucc_ctx;
  ucc_team_h      ucc_team;
};

static __inline__ ucc_datatype_t ucc_typeConvert(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
    return UCC_DT_INT8;
  case ncclUint8:
    return UCC_DT_UINT8;
  case ncclInt32:
    return UCC_DT_INT32;
  case ncclUint32:
    return UCC_DT_UINT32;
  case ncclInt64:
    return UCC_DT_INT64;
  case ncclUint64:
    return UCC_DT_UINT64;
  case ncclFloat16:
    return UCC_DT_FLOAT16;
  case ncclFloat32:
    return UCC_DT_FLOAT32;
  case ncclFloat64:
    return UCC_DT_FLOAT64;
  default:
    return -1;
  }
}

static __inline__ ucc_reduction_op_t ucc_opConvert(ncclRedOp_t op) {
  switch (op) {
  case ncclSum:
    return UCC_OP_SUM;
  case ncclProd:
    return UCC_OP_PROD;
  case ncclMax:
    return UCC_OP_MAX;
  case ncclMin:
    return UCC_OP_MIN;
  case ncclAvg:
    return UCC_OP_AVG;
  default:
    return -1;
  }
}

int ncclUCCAllGather(void *context, void *src_buf, void *recv_buf, int len) {
  struct ncclUCCCollComm *cComm = (struct ncclUCCCollComm *)context;

  nccl_p2p_plugin_t p2p_plugin;
  void *rMhandle = NULL, *sMhandle = NULL;

  assert(cComm->recvComm != NULL);
  assert(cComm->sendComm != NULL);

  p2p_plugin = nccl_p2p_get_plugin_type();
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(ncclNetPlugin_v7.regMr(cComm->recvComm, recv_buf,
                                       cComm->nranks * len, NCCL_PTR_HOST,
                                       &rMhandle));
    NCCLCHECK(ncclNetPlugin_v7.regMr(cComm->sendComm, recv_buf,
                                       cComm->nranks * len, NCCL_PTR_HOST,
                                       &sMhandle));
  }

  int speer = cComm->rank;
  memcpy(recv_buf + speer * len, src_buf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    void *srequest = NULL, *rrequest = NULL;
    int rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      void *rbuf = ((char *)recv_buf) + rpeer * len;
      int tag = 0x69;
      if (srequest == NULL)
        NCCLCHECK(ncclNetPlugin_v7.isend(cComm->sendComm,
                                           ((char *)recv_buf) + speer * len,
                                           len, tag, sMhandle, &srequest));
      if (rrequest == NULL)
        NCCLCHECK(ncclNetPlugin_v7.irecv(cComm->recvComm, 1, &rbuf, &len,
                                           &tag, &rMhandle, &rrequest));
    }
    while (srequest || rrequest) {
      int done;
      if (rrequest)
        NCCLCHECK(ncclNetPlugin_v7.test(rrequest, &done, NULL));
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECK(ncclNetPlugin_v7.test(srequest, &done, NULL));
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(ncclNetPlugin_v7.deregMr(cComm->recvComm, rMhandle));
    NCCLCHECK(ncclNetPlugin_v7.deregMr(cComm->sendComm, sMhandle));
  }

  return 0;
}

ucc_status_t UCC_oob_allgather(void *src_buf, void *recv_buf, size_t size,
                               void *coll_info, void **request) {
  NCCLCHECK(ncclUCCAllGather(coll_info, src_buf, recv_buf, (int)size))
  return UCC_OK;
}

ucc_status_t UCC_oob_req_test(void *request) { return UCC_OK; }
ucc_status_t UCC_oob_req_free(void *request) { return UCC_OK; }

#define ncclBfloat16 9

static __inline__ ucc_datatype_t typeConvert(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
    return UCC_DT_INT8;
  case ncclUint8:
    return UCC_DT_UINT8;
  case ncclInt32:
    return UCC_DT_INT32;
  case ncclUint32:
    return UCC_DT_UINT32;
  case ncclInt64:
    return UCC_DT_INT64;
  case ncclUint64:
    return UCC_DT_UINT64;
  case ncclFloat16:
    return UCC_DT_FLOAT16;
  case ncclFloat32:
    return UCC_DT_FLOAT32;
  case ncclFloat64:
    return UCC_DT_FLOAT64;
  default:
    return -1;
  }
}

static __inline__ int typeSize(ncclDataType_t type) {
  switch (type) {
    case ncclFloat16: return 2;
    case ncclInt32: return 4;
    case ncclUint32: return 4;
    case ncclFloat32: return 4;
    case ncclInt64: return 8;
    case ncclUint64: return 8;
    case ncclFloat64: return 8;
    case ncclBfloat16: return 2;
    case ncclInt8: return 1;
    case ncclUint8: return 1;
    default:
      WARN("UCC: unsupported data type\n");
      return -1;
  }
}

static __inline__ ucc_reduction_op_t opConvert(ncclRedOp_t op) {
  switch (op) {
    case ncclSum:  return UCC_OP_SUM;
    case ncclProd: return UCC_OP_PROD;
    case ncclMax:  return UCC_OP_MAX;
    case ncclMin:  return UCC_OP_MIN;
    case ncclAvg:  return UCC_OP_AVG;
    default:       return -1;
  }
}

static ucc_status_t ncclUccCtxCreate(struct ncclUCCCollComm *cComm, void *buf, size_t len)
{
  ucc_status_t         ucc_status;
  ucc_context_params_t ctx_params;
  ucc_context_config_h ctx_config;
  ucc_mem_map_t        map;

  map.address = buf;
  map.len = len;

  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
  ctx_params.oob.allgather = UCC_oob_allgather;
  ctx_params.oob.req_test = UCC_oob_req_test;
  ctx_params.oob.req_free = UCC_oob_req_free;
  ctx_params.oob.coll_info = (void *)cComm;
  ctx_params.oob.n_oob_eps = cComm->nranks;
  ctx_params.oob.oob_ep = cComm->rank;
  ctx_params.mem_params.segments = &map;
  ctx_params.mem_params.n_segments = 1;

  if (UCC_OK != (ucc_status = ucc_context_config_read(cComm->ucc_lib, NULL, &ctx_config))) {
    UCC_ERROR("UCC context config read failed\n");
    return ucc_status;
  }
  
  if (UCC_OK != (ucc_status = ucc_context_create(cComm->ucc_lib, &ctx_params, ctx_config,
                                  &cComm->ucc_ctx))) {
    UCC_ERROR("ucc context create failed");
    ucc_context_config_release(ctx_config);
    goto cleanup_lib;
  }
  ucc_context_config_release(ctx_config);

  return UCC_OK;
cleanup_lib:
  ucc_finalize(cComm->ucc_lib);
  return ucc_status; 
}
/* nccl team create */
static ucc_status_t ncclUccTeamCreate(struct ncclUCCCollComm *cComm, size_t rank, size_t size)
{
  ucc_status_t status;
  ucc_team_params_t team_params = {
    .mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_FLAGS,
    .oob = {
      .allgather = UCC_oob_allgather,
      .req_test = UCC_oob_req_test,
      .req_free = UCC_oob_req_free,
      .coll_info = cComm,
      .n_oob_eps = cComm->nranks,
      .oob_ep = cComm->rank,
    },
    .ep = cComm->rank,
    .flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER,
  };

  if (UCC_OK != (status = ucc_team_create_post(&cComm->ucc_ctx, 1, &team_params, &cComm->ucc_team))) {
    UCC_ERROR("team create post failed\n");
    return status;
  }

  while (UCC_INPROGRESS == (status = ucc_team_create_test(cComm->ucc_team))) {}
  if (UCC_OK != status) {
    UCC_ERROR("team create failed\n");
    return status;
  }
  return UCC_OK;
}
    
ncclResult_t ncclUCCInit(ncclDebugLogger_t logFunction) {
  num_outstanding = 0;

  return ncclNetPlugin_v7.init(logFunction);
}

ncclResult_t ncclUCCDevices(int* ndev) {
  *ndev = 1;
  return ncclSuccess;
}

ncclResult_t ncclUCCGetProperties_v8(int dev, ncclNetProperties_v8_t* props) {
  return ncclNetPlugin_v8.getProperties(dev, props);
}

ncclResult_t ncclUCCGetProperties_v7(int dev, ncclNetProperties_v7_t* props) {
  return ncclNetPlugin_v7.getProperties(dev, props);
}

ncclResult_t ncclUCCGetProperties_v6(int dev, ncclNetProperties_v6_t* props) {
  return  ncclNetPlugin_v6.getProperties(dev, props);
}

ncclResult_t ncclUCCGetProperties_v5(int dev, ncclNetProperties_v5_t* props) {
  return ncclNetPlugin_v5.getProperties(dev, props);
}

/* FERROL: nothing needs to be done here */
ncclResult_t ncclUCCListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclUCCListenComm *lComm;
  ncclResult_t status;

  NCCLCHECK(ncclIbMalloc((void**)&lComm, sizeof(struct ncclUCCListenComm)));
  status = ncclNetPlugin_v7.listen(dev, opaqueHandle, &lComm->listenCommP2P);
  lComm->dev = dev;
  *listenComm = lComm;
  return status;
}

ncclResult_t ncclUCCConnect(void* handles[], int nranks, int rank, void* listenComm, void** collComm)
{
  struct ncclUCCListenComm *lComm = (struct ncclUCCListenComm *)listenComm;
  struct ncclUCCCollComm *cComm;

  /* let's create endpoints here */
  char *useUCC;
  useUCC = getenv("NCCL_UCC_ENABLE");
  if (useUCC != NULL) {
    ucc_status_t status;
    size_t buffer_len;
    int next;

    NCCLCHECK(ncclIbMalloc((void *)&cComm, sizeof(struct ncclUCCCollComm)));

    cComm->nranks = nranks;
    cComm->rank = rank;
    /* create a staging buffer. F: should we remove? */
    staging_buffer = malloc(STAGING_BUF_LEN);
    buffer_len = STAGING_BUF_LEN;

    next = (cComm->rank + 1) % nranks;
    do {
      if (cComm->sendComm == NULL) {
        NCCLCHECK(ncclNetPlugin_v6.connect(lComm->dev, handles[next], &cComm->sendComm));
      }
      if (cComm->recvComm == NULL) {
        NCCLCHECK(ncclNetPlugin_v6.accept(lComm->listenCommP2P, &cComm->recvComm));
      }
    } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

    ucc_lib_config_h lib_config;
    ucc_lib_params_t lib_params;

    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_MULTIPLE;

    if (UCC_OK != ucc_lib_config_read("NCCL", NULL, &lib_config)) {
      UCC_ERROR("UCC lib config read failed");
      return -1;
    }
    if (UCC_OK != ucc_init(&lib_params, lib_config, &cComm->ucc_lib)) {
      UCC_ERROR("UCC lib init failed");
      ucc_lib_config_release(lib_config);
      return -2;
    }
    ucc_lib_config_release(lib_config);

    status = ncclUccCtxCreate(cComm, staging_buffer, buffer_len);
    if (status != UCC_OK) {
      return !ncclSuccess;
    }

    status = ncclUccTeamCreate(cComm, rank, nranks);
    if (status != UCC_OK) {
      // ctx destroy
      return !ncclSuccess;
    }
  }

  INFO(NCCL_INIT, "UCC rank %d / %d initialized\n", rank, nranks);
  *collComm = cComm;
  return ncclSuccess;
}

ncclResult_t ncclUCCReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) {
  *supported = ((typeConvert(dataType) != -1) && (ucc_opConvert(redOp) != -1));
  return ncclSuccess;
}

ncclResult_t ncclUCCRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return ncclSuccess;
}

ncclResult_t ncclUCCRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) {
  struct ncclUCCMemHandle *mh;
  NCCLCHECK(ncclIbMalloc((void **)&mh, sizeof(struct ncclUCCMemHandle)));
  mh->type = type;
  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t ncclUCCRegMr_v7(void* collComm, void* data, int size, int type, void** mhandle) {
  return ncclUCCRegMr(collComm, data, size, type, mhandle);
}

ncclResult_t ncclUCCDeregMr(void* collComm, void* mhandle) {
  free(mhandle);
  return ncclSuccess;
}

ncclResult_t ncclUCCGetRequest(struct ncclUCCRequest* reqs, struct ncclUCCRequest** req) {
  return ncclSuccess;
}

ncclResult_t ncclUCCIallreduce(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  struct ncclUCCCollComm *cComm = (struct ncclUCCCollComm *)collComm;

  struct ncclUCCMemHandle *mr_src = (struct ncclUCCMemHandle *)sendMhandle;
  struct ncclUCCMemHandle *mr_dst = (struct ncclUCCMemHandle *)recvMhandle;
  ucc_memory_type_t src_type = (mr_src->type == NCCL_PTR_CUDA) ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;
  ucc_memory_type_t dst_type = (mr_dst->type == NCCL_PTR_CUDA) ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;
  ucc_coll_req_h reqh;
  ucc_coll_args_t coll_args = {
    .mask = 0,
    .coll_type = UCC_COLL_TYPE_ALLREDUCE,
    .src.info = {
        .buffer = sendData,
        .count = count,
        .datatype = ucc_typeConvert(dataType),
        .mem_type = src_type,
    },
    .dst.info = {
        .buffer = recvData,
        .count = count,
        .datatype = ucc_typeConvert(dataType),
        .mem_type = dst_type,
    },
    .op = ucc_opConvert(redOp),
  };
  request_t *req = malloc(sizeof(request_t));

  ucc_collective_init(&coll_args, &reqh, cComm->ucc_team);
  ucc_collective_post(reqh);
  req->req_h = reqh;
  req->dst = recvData;
  req->len = typeSize(dataType) * count;
  req->id = num_outstanding;
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclUCCIallgather(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v8_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                             void* sendMhandle, void** request)
{
  struct ncclUCCCollComm *cComm = (struct ncclUCCCollComm *)collComm;
  struct ncclUCCMemHandle *mr_src = (struct ncclUCCMemHandle *)sendMhandle;
  struct ncclUCCMemHandle *mr_dst = (struct ncclUCCMemHandle *)recvParts[0].mhandle;
  ucc_memory_type_t src_type = (mr_src->type == NCCL_PTR_CUDA) ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;
  ucc_memory_type_t dst_type = (mr_dst->type == NCCL_PTR_CUDA) ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;
  request_t *req = malloc(sizeof(request_t));
  ucc_coll_req_h reqh;
  if (bytesPerRank == recvParts[0].size) {
    ucc_coll_args_t coll_args = {
        .mask = 0,
        .coll_type = UCC_COLL_TYPE_ALLGATHER,
        .src.info = {
            .buffer = sendData,
            .count = bytesPerRank,
            .datatype = UCC_DT_INT8,
            .mem_type = src_type,
        },
        .dst.info = {
            .buffer = recvParts[0].address,
            .count = recvParts[0].size,
            .datatype = UCC_DT_INT8,
            .mem_type = dst_type,
        },
    };
    ucc_collective_init(&coll_args, &reqh, cComm->ucc_team);
    ucc_collective_post(reqh);
  }
  req->req_h = reqh;
  req->dst = recvParts[0].address;
  req->len = bytesPerRank;
  req->id = num_outstanding;
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclUCCIreducescatter(void* collComm, int nSendParts, ncclNetSGE_v8_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                 ncclDataType_t dataType, ncclRedOp_t redOp,
                                 void* recvMhandle, void** request)
{
  struct ncclUCCCollComm* cComm = (struct ncclUCCCollComm*)collComm;

  ucc_datatype_t ucc_type = typeConvert(dataType);
  if (ucc_type == -1) {
    WARN("UCC: unsupported data type\n");
    return ncclInternalError;
  }

  ucc_reduction_op_t op_type = ucc_opConvert(redOp);
  if (op_type == -1) {
    WARN("UCC: unsupported reduce operation\n");
    return ncclInternalError;
  }

  assert(nSendParts == 1);
  struct ncclUCCMemHandle *mr_src = (struct ncclUCCMemHandle *)sendParts[0].mhandle;
  struct ncclUCCMemHandle *mr_dst = (struct ncclUCCMemHandle *)recvMhandle;
  ucc_memory_type_t src_type = (mr_src->type == NCCL_PTR_CUDA) ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;
  ucc_memory_type_t dst_type = (mr_dst->type == NCCL_PTR_CUDA) ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;

  int dt_size = typeSize(dataType);
  request_t *req = malloc(sizeof(request_t));
  ucc_coll_args_t coll_args = {
    .mask = 0,
    .coll_type = UCC_COLL_TYPE_REDUCE_SCATTER,
    .src.info = {
        .buffer = sendParts[0].address,
        .count = sendParts[0].size,
        .datatype = ucc_typeConvert(dataType),
        .mem_type = src_type,
    },
    .dst.info = {
        .buffer = recvData,
        .count = bytesPerRank,
        .datatype = ucc_typeConvert(dataType),
        .mem_type = dst_type,
    },
    .op = ucc_opConvert(redOp),
  };
  ucc_coll_req_h reqh;

  ucc_collective_init(&coll_args, &reqh, cComm->ucc_team);
  ucc_collective_post(reqh);
  req->req_h = reqh;
  req->dst = recvData;
  req->len = dt_size;
  req->id = num_outstanding;

  *request = req;
  return ncclSuccess;
 }

ncclResult_t ncclUCCIflush(void* collComm, void* data, int size, void* mhandle, void **request) {
   return ncclSuccess;
}

ncclResult_t ncclUCCTest(void* request, int* done, int* size) {
  request_t *req = (request_t *)request;
  ucc_coll_req_h reqh = req->req_h;
  ucc_status_t status;

  status = ucc_collective_test(reqh);
  if (status == UCC_INPROGRESS) {
    *done = 0;
    return ncclSuccess;
  } else if (status < 0) {
    UCC_ERROR("error in test");
    return !ncclSuccess;
  }

  *done = 1;
  ucc_collective_finalize(reqh);
  return ncclSuccess;
}

ncclResult_t ncclUCCCloseColl(void* collComm) {
  struct ncclUCCCollComm *cComm = (struct ncclUCCCollComm *)collComm;

  ucc_team_destroy(cComm->ucc_team);
  ucc_context_destroy(cComm->ucc_ctx);
  ucc_finalize(cComm->ucc_lib);
  ncclNetPlugin_v7.closeRecv(cComm->recvComm);
  ncclNetPlugin_v7.closeSend(cComm->sendComm);
  free(cComm);
  return ncclSuccess;
}

ncclResult_t ncclUCCCloseListen(void* listenComm) {
  struct ncclUCCListenComm *lComm = (struct ncclUCCListenComm*)listenComm;
  ncclResult_t status;

  status = ncclNetPlugin_v7.closeListen(lComm->listenCommP2P);
  free(listenComm);
  return status;
}

ncclCollNet_v8_t ncclCollNetPlugin_v8 = {
  "UCC",
  ncclUCCInit,
  ncclUCCDevices,
  ncclUCCGetProperties_v8,
  ncclUCCListen,
  ncclUCCConnect,
  ncclUCCReduceSupport,
  ncclUCCRegMr,
  ncclUCCRegMrDmaBuf,
  ncclUCCDeregMr,
  ncclUCCIallreduce,
  ncclUCCIallgather,
  ncclUCCIreducescatter,
  ncclUCCIflush,
  ncclUCCTest,
  ncclUCCCloseColl,
  ncclUCCCloseListen
};

ncclCollNet_v7_t ncclCollNetPlugin_v7 = {
  "UCC",
  ncclUCCInit,
  ncclUCCDevices,
  ncclUCCGetProperties_v7,
  ncclUCCListen,
  ncclUCCConnect,
  ncclUCCReduceSupport,
  ncclUCCRegMr_v7,
  ncclUCCRegMrDmaBuf,
  ncclUCCDeregMr,
  ncclUCCIallreduce,
  ncclUCCIflush,
  ncclUCCTest,
  ncclUCCCloseColl,
  ncclUCCCloseListen
};

ncclCollNet_v6_t ncclCollNetPlugin_v6 = {
  "UCC",
  ncclUCCInit,
  ncclUCCDevices,
  ncclUCCGetProperties_v6,
  ncclUCCListen,
  ncclUCCConnect,
  ncclUCCReduceSupport,
  ncclUCCRegMr_v7,
  ncclUCCRegMrDmaBuf,
  ncclUCCDeregMr,
  ncclUCCIallreduce,
  ncclUCCIflush,
  ncclUCCTest,
  ncclUCCCloseColl,
  ncclUCCCloseListen
};

ncclCollNet_v5_t ncclCollNetPlugin_v5 = {
  "UCC",
  ncclUCCInit,
  ncclUCCDevices,
  ncclUCCGetProperties_v5,
  ncclUCCListen,
  ncclUCCConnect,
  ncclUCCReduceSupport,
  ncclUCCRegMr_v7,
  ncclUCCDeregMr,
  ncclUCCIallreduce,
  ncclUCCIflush,
  ncclUCCTest,
  ncclUCCCloseColl,
  ncclUCCCloseListen
};