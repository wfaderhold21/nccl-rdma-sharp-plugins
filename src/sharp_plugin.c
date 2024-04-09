/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>

#include "config.h"
#include "core.h"
#include "p2p_plugin.h"
#include "param.h"
#include "sharp/api/version.h"
#include "sharp/api/sharp_coll.h"
#include "utils.h"
#include "nccl.h"

#if HAVE_UROM
#include "urom/api/urom.h"
#include "ucc/api/ucc.h"
#include <mpi.h>
#endif

extern ncclNet_v8_t ncclNetPlugin_v8;
extern ncclNet_v7_t ncclNetPlugin_v7;
extern ncclNet_v6_t ncclNetPlugin_v6;
extern ncclNet_v5_t ncclNetPlugin_v5;

int ncclNSharpDevs = -1;
struct sharp_coll_caps sharp_caps;
static int ncclSharpV3DatatypesSupported = 0;
NCCL_PARAM(SharpGroupSizeThresh, "SHARP_GROUP_SIZE_THRESH", 2);
NCCL_PARAM(SharpV3Datatypes, "SHARP_V3_DATATYPES", 2);

enum ncclSharpRequestType {
  NCCL_SHARP_REQ_SHARP_COLL,
  NCCL_SHARP_REQ_IFLUSH,
};

struct ncclSharpRequest {
  int requestType;
  void *sharpRequest;
  int  size;
  int  used;
  void *dest;
};

struct ncclSharpListenComm {
  int   dev;
  void *listenCommP2P;
};

struct ncclSharpCollComm {
  int    rank;
  int    nranks;
  void*  recvComm;
  void*  sendComm;
  struct ncclSharpRequest*   reqs;
  struct sharp_coll_context* sharpCollContext;
  struct sharp_coll_comm*    sharpCollComm;
};

struct ncclSharpMemHandle{
  void *mr;
  void *ncclIbMr;
  int  type;
};

struct ncclSharpInfo {
  uint64_t hostId;
  uint64_t jobId;
};

int num_outstanding = 0;
struct ncclUromInfo {
  urom_service_h  service;
  urom_worker_h   worker;
  urom_domain_h   udom;
  ucc_context_h   ucc_ctx;
  ucc_team_h      ucc_team;
  uint8_t         worker_addr[UROM_WORKER_ADDR_MAX_LEN];
  uint64_t        worker_id;
};

ucp_context_h ucp_ctx;
ucp_worker_h ucp_worker;
ucp_address_t *ucp_worker_addr;
size_t ucp_worker_addr_len;
void *shared_buffer;
size_t buffer_len = 0;
struct export_buf     ebuf;



struct ncclUromInfo urom_info;
#define SHARED_BUF_LEN  536870912

#define ncclBfloat16 9

static __inline__ enum sharp_datatype typeConvert(ncclDataType_t type) {
  switch (type) {
    case ncclFloat16: return SHARP_DTYPE_FLOAT_SHORT;
    case ncclInt32: return SHARP_DTYPE_INT;
    case ncclUint32: return SHARP_DTYPE_UNSIGNED;
    case ncclFloat32: return SHARP_DTYPE_FLOAT;
    case ncclInt64: return SHARP_DTYPE_LONG;
    case ncclUint64: return SHARP_DTYPE_UNSIGNED_LONG;
    case ncclFloat64: return SHARP_DTYPE_DOUBLE;
#ifdef HAVE_SHARP_DTYPE_BFLOAT16_UINT8_INT8
    case ncclBfloat16: return (ncclSharpV3DatatypesSupported ? SHARP_DTYPE_BFLOAT16 : SHARP_DTYPE_NULL);
    case ncclInt8: return (ncclSharpV3DatatypesSupported ? SHARP_DTYPE_INT8 : SHARP_DTYPE_NULL);
    case ncclUint8: return (ncclSharpV3DatatypesSupported ? SHARP_DTYPE_UINT8 : SHARP_DTYPE_NULL);
#endif
    default: return SHARP_DTYPE_NULL;
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
      WARN("SHARP: unsupported data type\n");
      return -1;
  }
}

static __inline__ enum sharp_reduce_op opConvert(ncclRedOp_t op) {
  switch (op) {
    case ncclSum: return SHARP_OP_SUM;
    case ncclMax: return SHARP_OP_MAX;
    case ncclMin: return SHARP_OP_MIN;
    default: return SHARP_OP_NULL;
  }
}

int ncclSharpAllGather(void *context, void *buf, int len) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)context;
  nccl_p2p_plugin_t p2p_plugin;
  void* rMhandle = NULL, *sMhandle = NULL;

  assert(cComm->recvComm != NULL);
  assert(cComm->sendComm != NULL);

  p2p_plugin = nccl_p2p_get_plugin_type();
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(ncclNetPlugin_v7.regMr(cComm->recvComm, buf, cComm->nranks*len, NCCL_PTR_HOST, &rMhandle));
    NCCLCHECK(ncclNetPlugin_v7.regMr(cComm->sendComm, buf, cComm->nranks*len, NCCL_PTR_HOST, &sMhandle));
  }

  int speer = cComm->rank;
  for (int i=0; i<cComm->nranks-1; i++) {
    void* srequest = NULL, *rrequest = NULL;
    int rpeer = (speer-1+cComm->nranks)%cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
       void *rbuf = ((char*)buf)+rpeer*len;
       int tag = 0x69;
       if (srequest == NULL) NCCLCHECK(ncclNetPlugin_v7.isend(cComm->sendComm, ((char*)buf)+speer*len, len, tag, sMhandle, &srequest));
       if (rrequest == NULL) NCCLCHECK(ncclNetPlugin_v7.irecv(cComm->recvComm, 1, &rbuf, &len, &tag, &rMhandle, &rrequest));
    }
    while (srequest || rrequest) {
      int done = 0; /* silent uninitialized false positive */
      if (rrequest) NCCLCHECK(ncclNetPlugin_v7.test(rrequest, &done, NULL));
      if (done) rrequest = NULL;
      if (srequest) NCCLCHECK(ncclNetPlugin_v7.test(srequest, &done, NULL));
      if (done) srequest = NULL;
    }
    speer = rpeer;
  }
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(ncclNetPlugin_v7.deregMr(cComm->recvComm, rMhandle));
    NCCLCHECK(ncclNetPlugin_v7.deregMr(cComm->sendComm, sMhandle));
  }

  return 0;
}

int ncclSharpOobBarrier(void *ctx) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  int* dummy;
  NCCLCHECK(ncclIbMalloc((void**)&dummy, cComm->nranks*sizeof(int)));
  NCCLCHECK(ncclSharpAllGather(ctx, dummy, sizeof(int)));
  free(dummy);
  return 0;
}

int ncclSharpOobGather(void *ctx, int root, void *sbuf, void *rbuf, int size) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  int nranks = cComm->nranks;
  void *tmp;
  NCCLCHECK(ncclIbMalloc(&tmp, nranks*size));
  memcpy((void*)((ptrdiff_t)tmp + size*cComm->rank), sbuf, size);
  NCCLCHECK(ncclSharpAllGather(cComm, tmp, size));
  if (cComm->rank == root) {
    memcpy(rbuf, tmp, nranks*size);
  }
  free(tmp);
  return 0;
}

int ncclSharpOobBcast(void *ctx, void *buf, int size, int root) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  void *tmp;
  NCCLCHECK(ncclIbMalloc(&tmp, size*cComm->nranks));
  if (cComm->rank == root) {
    memcpy((void*)((ptrdiff_t)tmp+size*cComm->rank), buf, size);
  }
  NCCLCHECK(ncclSharpAllGather(cComm, tmp, size));
  if (cComm->rank != root) {
    memcpy(buf, (void*)((ptrdiff_t)tmp+size*root), size);
  }
  free(tmp);
  return 0;
}

/* UROM addition */
static urom_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   void *coll_info, void **req)
{
  MPI_Comm comm = (MPI_Comm) (uintptr_t)coll_info;
  MPI_Request request;
  MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm, &request);
  *req = request;
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  *req = UROM_OK;
  return UROM_OK;
}

static urom_status_t oob_allgather_test(void *req)
{
  /* FERROL: allgather is implemented as blocking, just return OK */
  return UROM_OK;
}

static urom_status_t oob_allgather_free(void *req)
{
  return UROM_OK;
}

/* END UROM addition */



int ncclUromInit(void) {
  urom_service_h service;
  urom_worker_h worker;
  urom_worker_params_t worker_params;
  uint64_t worker_id = UROM_WORKER_ID_ANY;
  uint8_t worker_addr[UROM_WORKER_ADDR_MAX_LEN];
  size_t  worker_addr_len = UROM_WORKER_ADDR_MAX_LEN;
  urom_device_t *dev;
  struct urom_device *device_list;
  int num_devices;
  char *dev_name = NULL;
  urom_service_params_t service_params = {};
  urom_status_t status;
  int a;

    MPI_Initialized(&a);
  if (!a) {
    MPI_Init(NULL, NULL);
    }
    num_outstanding = 0;

  /* connect to uromd */
  status = urom_get_device_list(&device_list, &num_devices);
  if (status != UROM_OK) {
    printf("barf\n");
    return -1;
  }

  dev = device_list;
  while (dev) {
    if (dev_name) {
      if (!strcmp(dev_name, dev->name)) {
        break;
      }
    } else {
      break;
    }
    dev = dev->next;
  }

  if (!dev) {
    printf("no device\n");
    return -2;
  }

  service_params.flags = UROM_SERVICE_PARAM_DEVICE;
  service_params.device = dev;
  status = urom_service_connect(&service_params, &service);
  if (status != UROM_OK) {
    fprintf(stderr, "urom_service_connect() returned error: %s\n",
            urom_status_string(status));
    status = urom_free_device_list(device_list);
    assert (status == UROM_OK);
    return -1;
  };

  urom_info.service = service;

  printf("Connected!\n");
  status = urom_free_device_list(device_list);

  /* Spawn worker */
  status = urom_worker_spawn(service, UROM_WORKER_TYPE_UCC, worker_addr,
                             &worker_addr_len, &worker_id);
  if (status != UROM_OK) {
      printf("urom_worker_spawn() returned error: %s\n",
             urom_status_string(status));
      return -1;
  }   

  printf("Spawned worker ID: %lu\n", worker_id);

  /* Worker connect */
  worker_params.serviceh        = service;
  worker_params.addr            = worker_addr;
  worker_params.addr_len        = worker_addr_len;
  worker_params.num_cmd_notifyq = 1;

  status = urom_worker_connect(&worker_params, &worker);
  if (status != UROM_OK) {
      printf("urom_worker_connect() returned error: %s\n",
             urom_status_string(status));
      return -1; 
  }

  urom_info.worker = worker;

  return 0;
}

ncclResult_t ncclSharpInit(ncclDebugLogger_t logFunction) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  srand((int) tval.tv_usec);

  /* set SHARP COLL library default for plugin */
  setenv("SHARP_COLL_ENABLE_SAT", "1", 0);
  setenv("SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD", "0", 0);
  setenv("SHARP_COLL_LOCK_ON_COMM_INIT", "1", 0);
  setenv("SHARP_COLL_LOG_LEVEL", "3", 0);

  ncclUromInit();

  return ncclNetPlugin_v7.init(logFunction);
}

ncclResult_t ncclSharpDevices(int* ndev) {
  *ndev = ncclNSharpDevs;
  return ncclSuccess;
}

ncclResult_t ncclSharpGetProperties_v8(int dev, ncclNetProperties_v8_t* props) {
  return ncclNetPlugin_v8.getProperties(dev, props);
}

ncclResult_t ncclSharpGetProperties_v7(int dev, ncclNetProperties_v7_t* props) {
  return ncclNetPlugin_v7.getProperties(dev, props);
}

ncclResult_t ncclSharpGetProperties_v6(int dev, ncclNetProperties_v6_t* props) {
  return  ncclNetPlugin_v6.getProperties(dev, props);
}

ncclResult_t ncclSharpGetProperties_v5(int dev, ncclNetProperties_v5_t* props) {
  return ncclNetPlugin_v5.getProperties(dev, props);
}

/* FERROL: nothing needs to be done here */
ncclResult_t ncclSharpListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclSharpListenComm *lComm;
  ncclResult_t status;

  NCCLCHECK(ncclIbMalloc((void**)&lComm, sizeof(struct ncclSharpListenComm)));
  status = ncclNetPlugin_v7.listen(dev, opaqueHandle, &lComm->listenCommP2P);
  lComm->dev = dev;
  *listenComm = lComm;
  return status;
}

struct export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void         *packed_memh;
    size_t        packed_memh_len;
    void         *packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

int buffer_export_ucc(ucp_context_h ucp_context, void *buf, size_t len,
                      struct export_buf *ebuf)
{
    ucs_status_t           ucs_status;
    ucp_mem_map_params_t   params;
    ucp_memh_pack_params_t pack_params;

    ebuf->ucp_context = ucp_context;

    params.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = buf;
    params.length  = len;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    assert(ucs_status == UCS_OK);
#if 1
    pack_params.field_mask = UCP_MEMH_PACK_PARAM_FIELD_FLAGS;
    pack_params.flags      = UCP_MEMH_PACK_FLAG_EXPORT;

    ucs_status = ucp_memh_pack(ebuf->memh, &pack_params, &ebuf->packed_memh,
                               &ebuf->packed_memh_len);
    if (ucs_status != UCS_OK) {
        printf("ucp_memh_pack() returned error: %s\n",
               ucs_status_string(ucs_status));
        ebuf->packed_memh     = NULL;
        ebuf->packed_memh_len = 0;
    }
#endif
    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        printf("ucp_rkey_pack() returned error: %s\n",
               ucs_status_string(ucs_status));
        return UROM_ERR_NO_RESOURCE;
    }

    printf("ucp_memh_pack() packed length: %ld\n", ebuf->packed_memh_len);
    printf("ucp_rkey_pack() packed length: %ld\n", ebuf->packed_key_len);
    return 0;
}



int ucp_init_ex(ucp_context_h *ucp_ctx, ucp_worker_h *ucp_worker, ucp_address_t **ucp_addr, size_t *len)
{
    ucs_status_t        ucs_status;
    ucp_config_t       *ucp_config;
    ucp_params_t        ucp_params;
    ucp_context_h       ucp_context;
    ucp_worker_params_t worker_params;
    ucp_address_t      *worker_addr;
    ucp_worker_h        worker;
    size_t              length;

    ucs_status = ucp_config_read(NULL, NULL, &ucp_config);
    assert(ucs_status == UCS_OK);

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG | UCP_FEATURE_RMA |
                          UCP_FEATURE_AMO64 | UCP_FEATURE_EXPORTED_MEMH;

    ucs_status = ucp_init(&ucp_params, ucp_config, &ucp_context);
    if (ucs_status != UCS_OK) {
        printf("error on ucp init\n");
        return -1;
    }

    *ucp_ctx = ucp_context;
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    ucs_status = ucp_worker_create(ucp_context, &worker_params, &worker);
    if (ucs_status != UCS_OK) {
        printf("error on worker create\n");
        return -1;
    }

    *ucp_worker = worker;
    ucs_status  = ucp_worker_get_address(worker, &worker_addr, &length);
    if (ucs_status != UCS_OK) {
        printf("failed to get address\n");
        return -1;
    }

    *ucp_addr = worker_addr;
    *len      = length;

    return 0;
}


ncclResult_t ncclSharpConnect(void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct ncclSharpListenComm* lComm = (struct ncclSharpListenComm*)listenComm;
  struct ncclSharpCollComm* cComm;
  char *useSharp;
  char *useUROM;

  if(nranks < ncclParamSharpGroupSizeThresh()) {
    INFO(NCCL_INIT|NCCL_NET|NCCL_ENV, "SHARP: Group size:%d is less than threshold:%d. fallback to non-sharp",
         nranks, ncclParamSharpGroupSizeThresh());
    return ncclInvalidUsage;
  }

  useSharp = getenv("NCCL_SHARP_DISABLE");
  if(useSharp != NULL) {
    if(strcmp(useSharp, "1") == 0) {
      INFO(NCCL_INIT|NCCL_NET|NCCL_ENV, "SHARP: Set to disable on this communicator");
      return ncclInvalidUsage;
    }
  }

  NCCLCHECK(ncclIbMalloc((void**)&cComm, sizeof(struct ncclSharpCollComm)));
  NCCLCHECK(ncclIbMalloc((void**)&cComm->reqs, sizeof(struct ncclSharpRequest)*MAX_REQUESTS));

  cComm->nranks = nranks;
  cComm->rank = rank;
  if (cComm->rank == -1) {
    WARN("Could not determine my rank\n");
    return ncclInternalError;
  }
  int next = (cComm->rank + 1) % nranks;
  do {
    if (cComm->sendComm == NULL)
      NCCLCHECK(ncclNetPlugin_v6.connect(lComm->dev, handles[next], &cComm->sendComm));
    if (cComm->recvComm == NULL)
      NCCLCHECK(ncclNetPlugin_v6.accept(lComm->listenCommP2P, &cComm->recvComm)); // From prev
  } while(cComm->sendComm == NULL || cComm->recvComm == NULL);

  struct ncclSharpInfo* allInfo;
  pid_t pid = getpid();
  pthread_t tid = pthread_self();
  NCCLCHECK(ncclIbMalloc((void**)&allInfo, sizeof(struct ncclSharpInfo)*nranks));
  allInfo[cComm->rank].hostId = gethostid();
  allInfo[cComm->rank].jobId = (((uint64_t)allInfo[cComm->rank].hostId << 32) | ((pid ^ tid) ^ rand()));
  NCCLCHECK(ncclSharpAllGather(cComm, allInfo, sizeof(struct ncclSharpInfo)));

  // Find my local rank;
  int localRank = 0;
  for (int i=0; i<cComm->rank; i++) {
    if (allInfo[cComm->rank].hostId == allInfo[i].hostId) {
      localRank++;
    }
  }
  uint64_t jobId = allInfo[0].jobId;
  free(allInfo);

  struct sharp_coll_init_spec init_spec = {0};
  init_spec.progress_func  = NULL;
  init_spec.job_id = jobId;
  init_spec.world_rank = cComm->rank;
  init_spec.world_size = nranks;
  init_spec.world_local_rank = 0;
  init_spec.enable_thread_support = 1;
  init_spec.group_channel_idx = 0;

  init_spec.oob_colls.barrier = ncclSharpOobBarrier;
  init_spec.oob_colls.bcast = ncclSharpOobBcast;
  init_spec.oob_colls.gather = ncclSharpOobGather;
  init_spec.oob_ctx = cComm;

  init_spec.config = sharp_coll_default_config;
  init_spec.config.user_progress_num_polls = 10000000;

  char devName[MAXNAMESIZE];
  ncclNetProperties_v6_t prop;
  ncclSharpGetProperties_v6(lComm->dev, &prop);
  snprintf(devName, MAXNAMESIZE, "%s:%d", prop.name, prop.port);
  init_spec.config.ib_dev_list = devName;
#if 0
  int ret = sharp_coll_init(&init_spec, &cComm->sharpCollContext);


  if (ret < 0) {
    WARN("NET/IB : SHARP coll init error: %s(%d)\n", sharp_coll_strerror(ret), ret);
    return ncclInternalError;
  }

#ifdef HAVE_SHARP_DTYPE_BFLOAT16_UINT8_INT8
  ret = sharp_coll_caps_query(cComm->sharpCollContext, &sharp_caps);
  if (ret < 0) {
    WARN("sharp_coll_caps_query failed : %s(%d)\n", sharp_coll_strerror(ret), ret);
    sharp_coll_finalize(cComm->sharpCollContext);
    return ncclInternalError;
  }

  int v3DatatypeMode = ncclParamSharpV3Datatypes();
  if (v3DatatypeMode == 1 || v3DatatypeMode == 2) {
    if (sharp_caps.support_mask.dtypes & (1<<SHARP_DTYPE_INT8))
      ncclSharpV3DatatypesSupported = 1;
    else
      WARN("SHARP int8,uint8,bfloat16 Datatypes not supported");
  }
#endif

  INFO(NCCL_INIT, "SHARP rank %d/%d initialized on %s", cComm->rank, nranks, devName);

  struct sharp_coll_comm_init_spec comm_spec;
  comm_spec.rank = cComm->rank;
  comm_spec.size = nranks;
  comm_spec.oob_ctx = cComm;
  comm_spec.group_world_ranks = NULL;

  ret = sharp_coll_comm_init(cComm->sharpCollContext, &comm_spec, &cComm->sharpCollComm);
  if (ret < 0) {
    WARN("SHARP group create: %s(%d)\n", sharp_coll_strerror(ret), ret);
    sharp_coll_finalize(cComm->sharpCollContext);
  //  return ncclInternalError;
  }
#endif
  *collComm = cComm;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* let's create endpoints here */

  useUROM = getenv("NCCL_UROM_ENABLE");
  if (1 || useUROM != NULL) {
    urom_status_t status;
    urom_worker_notify_t *notif;
    urom_mem_map_t  map = {};

    ucp_init_ex(&ucp_ctx, &ucp_worker, &ucp_worker_addr, &ucp_worker_addr_len);
    shared_buffer = malloc(SHARED_BUF_LEN);
    buffer_len = SHARED_BUF_LEN;
    buffer_export_ucc(ucp_ctx, shared_buffer, buffer_len, &ebuf);

    map.mask = UROM_WORKER_MEM_MAP_FIELD_BASE_VA | UROM_WORKER_MEM_MAP_FIELD_MKEY;
    map.base_va = (uint64_t)shared_buffer;
    map.len = buffer_len;
    map.mkey = ebuf.packed_key;
    map.mkey_len = ebuf.packed_key_len;

    ucc_lib_params_t lib_params = {
        .mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE,
    };
    urom_worker_cmd_t init_cmd = {
        .cmd_type                  = UROM_WORKER_CMD_UCC,
        .ucc.cmd_type              = UROM_WORKER_CMD_UCC_LIB_CREATE,
        .ucc.lib_create_cmd.params = &lib_params,
    };
#if 1
    urom_worker_cmd_t pass_dc_cmd = {
        .cmd_type     = UROM_WORKER_CMD_UCC,
        .ucc.cmd_type = UROM_WORKER_CMD_CREATE_PASSIVE_DATA_CHANNEL,
        .ucc.pass_dc_create_cmd.ucp_addr = ucp_worker_addr,
        .ucc.pass_dc_create_cmd.addr_len = ucp_worker_addr_len,
    };
#endif
    urom_worker_cmd_t ctx_cmd = {
        .cmd_type          = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = rank,
        .ucc.cmd_type      = UROM_WORKER_CMD_UCC_CONTEXT_CREATE,
        .ucc.context_create_cmd =
            {
              .start   = 0,
              .stride  = 1,
              .size    = size,
              .base_va = shared_buffer,
              .len     = buffer_len,
          },
    };
    urom_worker_cmd_t team_cmd = {
      .cmd_type          = UROM_WORKER_CMD_UCC,
      .ucc.dpu_worker_id = rank,
      .ucc.cmd_type      = UROM_WORKER_CMD_UCC_TEAM_CREATE,
      .ucc.team_create_cmd =
          {
              .start  = 0,
              .stride = 1,
              .size   = size,
          },
    };

    /* create a domain */
    urom_domain_params_t udom_params = {
      .flags = UROM_DOMAIN_WORKER_ADDR,
      .mask = UROM_DOMAIN_PARAM_FIELD_OOB |
              UROM_DOMAIN_PARAM_FIELD_WORKER |
              UROM_DOMAIN_PARAM_FIELD_WORKER_ID |
              UROM_DOMAIN_PARAM_FIELD_MEM_MAP,
      .oob =
          {
              .allgather     = oob_allgather,
              .req_test      = oob_allgather_test,
              .req_free      = oob_allgather_free,
              .coll_info     = MPI_COMM_WORLD,
              .n_oob_indexes = size,
              .oob_index     = rank,
          },
      .domain_worker_id = rank,
      .workers          = &urom_info.worker,
      .num_workers      = 1,
      .domain_size      = size,
      .mem_map = 
        {
        .segments = &map,
        .n_segments = 1,
        },
    };
    printf("hello\n");
    status = urom_domain_create_post(&udom_params, &urom_info.udom);
    if (status < UROM_OK) {
        printf("urom error: %s\n", urom_status_string(status));
        return status;
    }
    while (UROM_INPROGRESS == (status = urom_domain_create_test(urom_info.udom)))
      ;

    if (status < UROM_OK) {
      printf("urom error: %s\n", urom_status_string(status));
      return status;
    }
    /* create a context */
    urom_worker_push_cmdq(urom_info.worker, 0, &init_cmd);
    while (UROM_ERR_QUEUE_EMPTY ==
           (status = urom_worker_pop_notifyq(urom_info.worker, 0, &notif))) {
        sched_yield();
    }
    printf("debug: lib create notif->status: %d\n", notif->ucc.status);
#if 1
        urom_worker_push_cmdq(urom_info.worker, 0, &pass_dc_cmd);
        while (UROM_ERR_QUEUE_EMPTY ==
               (status = urom_worker_pop_notifyq(urom_info.worker, 0, &notif))) {
            sched_yield();
        }
        printf("debug: pass dc create notif->status: %d\n", notif->ucc.status);
#endif
    urom_worker_push_cmdq(urom_info.worker, 0, &ctx_cmd);
    while (UROM_ERR_QUEUE_EMPTY ==
           (status = urom_worker_pop_notifyq(urom_info.worker, 0, &notif))) {
        sched_yield();
    }
    printf("debug: ctx create notif->status: %d, ucc_context: %p\n",
           notif->ucc.status, notif->ucc.context_create_nqe.context);
    team_cmd.ucc.team_create_cmd.context_h = (void *)notif->ucc.context_create_nqe.context;
    urom_info.ucc_ctx = notif->ucc.context_create_nqe.context;
    /* create a team */
    urom_worker_push_cmdq(urom_info.worker, 0, &team_cmd);
    while (UROM_ERR_QUEUE_EMPTY ==
           (status = urom_worker_pop_notifyq(urom_info.worker, 0, &notif))) {
        sched_yield();
    }
    printf("debug: team create notif->status: %d, team_h: %p\n",
           notif->ucc.status, notif->ucc.team_create_nqe.team);
    urom_info.ucc_team = notif->ucc.team_create_nqe.team;
  }

  return ncclSuccess;
}

ncclResult_t ncclSharpReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) {
  *supported = ((typeConvert(dataType) != SHARP_DTYPE_NULL) && (opConvert(redOp) != SHARP_OP_NULL));
  return ncclSuccess;
}

ncclResult_t ncclSharpRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
#if HAVE_DECL_SHARP_COLL_REG_MR_V2
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;
  struct sharp_coll_reg_params reg_params;

  struct ncclSharpMemHandle *mh;
  NCCLCHECK(ncclIbMalloc((void**)&mh, sizeof(struct ncclSharpMemHandle)));

  reg_params.field_mask = SHARP_COLL_REG_FIELD_DMABUF_FD | SHARP_COLL_REG_FIELD_DMABUF_OFFSET;
  reg_params.dmabuf_fd = fd;
  reg_params.dmabuf_offset = offset;
  mh->type = type;
  if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr_v2(cComm->sharpCollContext, data, size, &reg_params, &(mh->mr))) {
    WARN("SHARP regmr failed\n");
    return ncclSystemError;
  }
  TRACE(NCCL_INIT,"sharpRegAddr %lx size %ld handle %x", data, size, mh->mr);

  NCCLCHECK(ncclNetPlugin_v8.regMrDmaBuf(cComm->recvComm, data, size, type, offset, fd, &mh->ncclIbMr));

  *mhandle = mh;
  return ncclSuccess;
#else
    printf("ERROR!\n");
  return ncclInternalError;
#endif
}

typedef struct reg_keys {
    uint64_t xgvmi_flag;
    size_t src_len;
    size_t dst_len;
    char rkeys[];
} reg_keys_t;

typedef struct urom_mapped_mem {
    ucp_mem_h memh;
    void *src;
    size_t len;
    size_t key_len;
    void *key;
} urom_mapped_mem_t;

size_t map_beg = 0;
size_t map_end = 0;
urom_mapped_mem_t map_array[128];

ncclResult_t ncclSharpRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) {
#if 0
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  struct ncclSharpMemHandle *mh;
  NCCLCHECK(ncclIbMalloc((void**)&mh, sizeof(struct ncclSharpMemHandle)));

  mh->type = type;
  if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(cComm->sharpCollContext, data, size, &(mh->mr))) {
    WARN("SHARP regmr failed\n");
    return ncclSystemError;
  }
  TRACE(NCCL_INIT,"sharpRegAddr %lx size %ld handle %x", data, size, mh->mr);

  NCCLCHECK(ncclNetPlugin_v8.regMr(cComm->recvComm, data, size, type, &mh->ncclIbMr));

  *mhandle = mh;
#else
    ucp_mem_map_params_t mem_params;
    ucs_status_t ucs_status;
    urom_mapped_mem_t *map = &map_array[map_end];

    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mem_params.address = data;
    mem_params.length = size;
    ucs_status = ucp_mem_map(ucp_ctx, &mem_params, &map->memh);
    assert (ucs_status == UCS_OK);

    ucs_status = ucp_rkey_pack(ucp_ctx, map->memh, &map->key, &map->key_len);
    assert (ucs_status == UCS_OK);

    map->src = data;
    map->len = size;

    map_end = (map_end + 1) % 128;
    *mhandle = map;
#endif
    printf("noop 2\n");
   return ncclSuccess;
}


ncclResult_t ncclSharpRegMr_v7(void* collComm, void* data, int size, int type, void** mhandle) {
    return ncclSuccess;
//  return ncclSharpRegMr(collComm, data, (size_t)size, type, mhandle);
}

ncclResult_t ncclSharpDeregMr(void* collComm, void* mhandle) {
#if 0
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;
  struct ncclSharpMemHandle *mh = (struct ncclSharpMemHandle *)mhandle;

  if (SHARP_COLL_SUCCESS != sharp_coll_dereg_mr(cComm->sharpCollContext, mh->mr)) {
    WARN("SHARP deregmr failed\n");
  }

  NCCLCHECK(ncclNetPlugin_v7.deregMr(cComm->recvComm, mh->ncclIbMr));

  free(mh);
#endif
    urom_mapped_mem_t *map = (urom_mapped_mem_t *)mhandle;
    ucp_mem_unmap(ucp_ctx, map->memh);
  return ncclSuccess;
}

ncclResult_t ncclSharpGetRequest(struct ncclSharpRequest* reqs, struct ncclSharpRequest** req) {
/*  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSharpRequest* r = reqs+i;
    if (r->used == 0) {
      r->used = 1;
      r->sharpRequest = NULL;
      r->size = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("SHARP : unable to allocate request");
  *req = NULL;
  return ncclInternalError;*/
    return ncclSuccess;
}

typedef struct urom_request {
    uint64_t id;
    void *dst;
    size_t len;
    void *keys;
} urom_request_t;

ncclResult_t ncclSharpIallreduce(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
#if 0
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  enum sharp_datatype sharp_type = typeConvert(dataType);
  if (sharp_type == SHARP_DTYPE_NULL) {
    WARN("SHARP: unsupported data type\n");
    return ncclInternalError;
  }

  enum sharp_reduce_op op_type = opConvert(redOp);
  if (op_type == SHARP_OP_NULL) {
    WARN("SHARP: unsupported reduce operation\n");
    return ncclInternalError;
  }

  int dt_size = typeSize(dataType);
  struct ncclSharpMemHandle *mr_sbuf = (struct ncclSharpMemHandle*)sendMhandle;
  struct ncclSharpMemHandle *mr_rbuf = (struct ncclSharpMemHandle*)recvMhandle;

  struct ncclSharpRequest* req;
  NCCLCHECK(ncclSharpGetRequest(cComm->reqs, &req));

  struct sharp_coll_reduce_spec reduce_spec;

  reduce_spec.sbuf_desc.buffer.ptr = sendData;
  reduce_spec.sbuf_desc.buffer.length = count * dt_size;
  reduce_spec.sbuf_desc.buffer.mem_handle = mr_sbuf->mr;
  reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.sbuf_desc.mem_type = (mr_sbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.rbuf_desc.buffer.ptr = recvData;
  reduce_spec.rbuf_desc.buffer.length = count * dt_size;
  reduce_spec.rbuf_desc.buffer.mem_handle = mr_rbuf->mr;
  reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.rbuf_desc.mem_type = (mr_rbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.length = count;
  reduce_spec.dtype = sharp_type;
  reduce_spec.op = op_type;
  reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

#if BLOCKING==0
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce_nb(cComm->sharpCollComm, &reduce_spec, &req->sharpRequest)) {
    WARN("SHARP allreduce failed\n");
  }
  req->size =  count * dt_size;
#else
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce(cComm->sharpCollComm, &reduce_spec)) {
    WARN("SHARP allreduce failed\n");
  }
  req->sharpRequest = (void *) 0xabababab;
  req->size =  count * dt_size;
#endif
  req->requestType = NCCL_SHARP_REQ_SHARP_COLL;
  *request = req;
#endif
    int rank;
    urom_request_t *req = malloc(sizeof(urom_request_t));
    urom_worker_notify_t *notif;
    urom_status_t status;
    int dt_size = typeSize(dataType);
    reg_keys_t *keys = malloc(sizeof(reg_keys_t) + 1024);
    
    keys->xgvmi_flag = 0;
    //FIXME: this would eventually be map_beg, not 0
    for (int i = 0; i < map_end; i++) {
        if (map_array[i].src <= sendData &&
            (map_array[i].src + map_array[i].len) >= sendData) {
            keys->src_len = map_array[i].key_len;
            memcpy(keys->rkeys, map_array[i].key, keys->src_len);
            break;
        }
    }
#if 1
    // error check?
    for (int i = 0; i < map_end; i++) {
        if (map_array[i].src <= recvData &&
            (map_array[i].src + map_array[i].len) >= recvData) {
            keys->dst_len = map_array[i].key_len;
            memcpy(keys->rkeys + keys->src_len, map_array[i].key, keys->dst_len);
            break;
        }
    }
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t size = count * dt_size;
    ucc_coll_args_t coll_args_allreduce = {
        .mask = UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER | UCC_COLL_ARGS_FIELD_FLAGS,
        .flags = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS,
        .coll_type = UCC_COLL_TYPE_ALLREDUCE,
        .src.info = {
            .buffer = (void *)sendData,
            .count = size,
            .datatype = UCC_DT_FLOAT32, //change this
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN,
        },
        .dst.info = {
            .buffer = (void *)(recvData),
            .count = size,
            .datatype = UCC_DT_FLOAT32,
            .mem_type = UCC_MEMORY_TYPE_UNKNOWN,
        },
        .global_work_buffer = keys,
    };
    urom_worker_cmd_t coll_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = rank,
        .ucc.cmd_type = UROM_WORKER_CMD_UCC_COLL,
        .ucc.coll_cmd.coll_args = &coll_args_allreduce,
        .ucc.coll_cmd.team = urom_info.ucc_team,
        .ucc.coll_cmd.use_xgvmi = 0,
        .ucc.coll_cmd.work_buffer = coll_args_allreduce.global_work_buffer,
        .ucc.coll_cmd.work_buffer_size = sizeof(reg_keys_t) + keys->src_len + keys->dst_len,
    };

    urom_worker_push_cmdq(urom_info.worker, 0, &coll_cmd);

    req->dst = recvData;
    req->len = size;
    req->id = num_outstanding;
    req->keys = keys;
    *request = req;
  return ncclSuccess;
}

ncclResult_t ncclSharpIallgather(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v8_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                             void* sendMhandle, void** request)
{
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;
  struct ncclSharpMemHandle *send_mh = (struct ncclSharpMemHandle*)sendMhandle;
  struct ncclSharpMemHandle *recv_mh = (struct ncclSharpMemHandle*)recvParts[0].mhandle;
  struct ncclSharpRequest* req;
  NCCLCHECK(ncclSharpGetRequest(cComm->reqs, &req));


  assert(nRecvParts == 1);

  struct sharp_coll_gather_spec gather_spec;

  gather_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
  gather_spec.sbuf_desc.buffer.ptr = sendData;
  gather_spec.sbuf_desc.buffer.length = bytesPerRank;
  gather_spec.sbuf_desc.buffer.mem_handle = send_mh->mr;

  gather_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
  gather_spec.rbuf_desc.buffer.ptr = recvParts[0].address;
  gather_spec.rbuf_desc.buffer.length = recvParts[0].size;
  gather_spec.rbuf_desc.buffer.mem_handle = recv_mh->mr;

  gather_spec.dtype = SHARP_DTYPE_INT8;
  gather_spec.size = recvParts[0].size;
  gather_spec.offset = windowOffset;

#if BLOCKING==0
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allgather_nb(cComm->sharpCollComm, &gather_spec, &req->sharpRequest)) {
    WARN("SHARP Allgather failed\n");
  }
  req->size = recvParts[0].size;
#else
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allgather(cComm->sharpCollComm, &gather_spec)) {
    WARN("SHARP Allgather failed\n");
  }
  req->sharpRequest = (void *) 0xabababab;
  req->size = recvSize;
#endif
  req->requestType = NCCL_SHARP_REQ_SHARP_COLL;
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclSharpIreducescatter(void* collComm, int nSendParts, ncclNetSGE_v8_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                 ncclDataType_t dataType, ncclRedOp_t redOp,
                                 void* recvMhandle, void** request)
{
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  enum sharp_datatype sharp_type = typeConvert(dataType);
  if (sharp_type == SHARP_DTYPE_NULL) {
    WARN("SHARP: unsupported data type\n");
    return ncclInternalError;
  }

  enum sharp_reduce_op op_type = opConvert(redOp);
  if (op_type == SHARP_OP_NULL) {
    WARN("SHARP: unsupported reduce operation\n");
    return ncclInternalError;
  }

  assert(nSendParts == 1);

  int dt_size = typeSize(dataType);
  struct ncclSharpMemHandle *mr_sbuf = (struct ncclSharpMemHandle*)sendParts[0].mhandle;
  struct ncclSharpMemHandle *mr_rbuf = (struct ncclSharpMemHandle*)recvMhandle;

  struct ncclSharpRequest* req;
  NCCLCHECK(ncclSharpGetRequest(cComm->reqs, &req));

  struct sharp_coll_reduce_spec reduce_spec;

  reduce_spec.sbuf_desc.buffer.ptr = sendParts[0].address;
  reduce_spec.sbuf_desc.buffer.length = sendParts[0].size;
  reduce_spec.sbuf_desc.buffer.mem_handle = mr_sbuf->mr;
  reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.sbuf_desc.mem_type = (mr_sbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.rbuf_desc.buffer.ptr = recvData;
  reduce_spec.rbuf_desc.buffer.length = bytesPerRank;
  reduce_spec.rbuf_desc.buffer.mem_handle = mr_rbuf->mr;
  reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.rbuf_desc.mem_type = (mr_rbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.length = sendParts[0].size / dt_size;
  reduce_spec.offset = windowOffset;
  reduce_spec.dtype = sharp_type;
  reduce_spec.op = op_type;
  reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

#if BLOCKING==0
  if (SHARP_COLL_SUCCESS != sharp_coll_do_reduce_scatter_nb(cComm->sharpCollComm, &reduce_spec, &req->sharpRequest)) {
    WARN("SHARP reduce_scatter failed\n");
  }
  req->size =  bytesPerRank;
#else
  if (SHARP_COLL_SUCCESS != sharp_coll_do_reduce_scatter(cComm->sharpCollComm, &reduce_spec)) {
    WARN("SHARP reduce_scater failed\n");
  }
  req->sharpRequest = (void *) 0xabababab;
  req->size =  recvCount * dt_size;
#endif
  req->requestType = NCCL_SHARP_REQ_SHARP_COLL;
  *request = req;
  return ncclSuccess;
 }

ncclResult_t ncclSharpIflush(void* collComm, void* data, int size, void* mhandle, void **request) {
  struct ncclSharpCollComm *cComm = (struct ncclSharpCollComm*)collComm;
  struct ncclSharpMemHandle *mh = (struct ncclSharpMemHandle *)mhandle;
  struct ncclSharpRequest* req;

  NCCLCHECK(ncclSharpGetRequest(cComm->reqs, &req));
  req->requestType = NCCL_SHARP_REQ_IFLUSH;
  ncclNetPlugin_v7.iflush(cComm->recvComm, 1, &data, &size, &mh->ncclIbMr, &req->sharpRequest);
  if (!req->sharpRequest) {
    *request = NULL;
     req->used = 0;
     return ncclSuccess;
   }

  *request = req;
   return ncclSuccess;
}

ncclResult_t ncclSharpTest(void* request, int* done, int* size) {
//  struct ncclSharpRequest* req = (struct ncclSharpRequest*)request;
    urom_request_t *p = (urom_request_t *)request;
    urom_worker_notify_t *notif;
    urom_status_t status;

    status = urom_worker_pop_notifyq(urom_info.worker, 0, &notif);
    if (UROM_ERR_QUEUE_EMPTY == status) {
        *done = 0;
        return ncclSuccess;
    }

    *done = 1;
//    free(p->keys);
    if (status < 0) {
        printf("ERROR IN UROM\n");
        return !ncclSuccess;
    }

  //  memcpy((void *)p->dst, shared_buffer + (128 * 1024 * 1024), p->len);

#if 0
    ncclNetPlugin_v7.test(req->sharpRequest, done, size);
    if (*done == 1) {
      req->used = 0;
    }
#endif

    return ncclSuccess;
}
#if 0
#if 0
#if BLOCKING==0
  *done = sharp_coll_req_test(req->sharpRequest);
  if (*done){
    sharp_coll_req_free(req->sharpRequest);
    *size = req->size;
    req->used = 0;
  } else {
    *done = 0;
  }
#else
  if (req->size != -1) {
    *done = 1;
    *size = req->size;
    req->used = 0;
  } else {
     *done = 0;
  }
#endif
#endif
  return ncclSuccess;
}
#endif
ncclResult_t ncclSharpCloseColl(void* collComm) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;
/*
  sharp_coll_comm_destroy(cComm->sharpCollComm);
  sharp_coll_finalize(cComm->sharpCollContext);

  NCCLCHECK(ncclNetPlugin_v7.closeRecv(cComm->recvComm));
  NCCLCHECK(ncclNetPlugin_v7.closeSend(cComm->sendComm));
  free(cComm);
*/
  return ncclSuccess;
}

ncclResult_t ncclSharpCloseListen(void* listenComm) {
  struct ncclSharpListenComm *lComm = (struct ncclSharpListenComm*)listenComm;
  ncclResult_t status;

  status = ncclNetPlugin_v7.closeListen(lComm->listenCommP2P);
  free(listenComm);
  return status;
}

ncclCollNet_v8_t ncclCollNetPlugin_v8 = {
  "SHARP",
  ncclSharpInit,
  ncclSharpDevices,
  ncclSharpGetProperties_v8,
  ncclSharpListen,
  ncclSharpConnect,
  ncclSharpReduceSupport,
  ncclSharpRegMr,
  ncclSharpRegMrDmaBuf,
  ncclSharpDeregMr,
  ncclSharpIallreduce,
  ncclSharpIallgather,
  ncclSharpIreducescatter,
  ncclSharpIflush,
  ncclSharpTest,
  ncclSharpCloseColl,
  ncclSharpCloseListen
};

ncclCollNet_v7_t ncclCollNetPlugin_v7 = {
  "SHARP",
  ncclSharpInit,
  ncclSharpDevices,
  ncclSharpGetProperties_v7,
  ncclSharpListen,
  ncclSharpConnect,
  ncclSharpReduceSupport,
  ncclSharpRegMr_v7,
  ncclSharpRegMrDmaBuf,
  ncclSharpDeregMr,
  ncclSharpIallreduce,
  ncclSharpIflush,
  ncclSharpTest,
  ncclSharpCloseColl,
  ncclSharpCloseListen
};

ncclCollNet_v6_t ncclCollNetPlugin_v6 = {
  "SHARP",
  ncclSharpInit,
  ncclSharpDevices,
  ncclSharpGetProperties_v6,
  ncclSharpListen,
  ncclSharpConnect,
  ncclSharpReduceSupport,
  ncclSharpRegMr_v7,
  ncclSharpRegMrDmaBuf,
  ncclSharpDeregMr,
  ncclSharpIallreduce,
  ncclSharpIflush,
  ncclSharpTest,
  ncclSharpCloseColl,
  ncclSharpCloseListen
};

ncclCollNet_v5_t ncclCollNetPlugin_v5 = {
  "SHARP",
  ncclSharpInit,
  ncclSharpDevices,
  ncclSharpGetProperties_v5,
  ncclSharpListen,
  ncclSharpConnect,
  ncclSharpReduceSupport,
  ncclSharpRegMr_v7,
  ncclSharpDeregMr,
  ncclSharpIallreduce,
  ncclSharpIflush,
  ncclSharpTest,
  ncclSharpCloseColl,
  ncclSharpCloseListen
};
