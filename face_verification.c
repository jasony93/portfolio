/**
 * @file  face_verification.c
 * @brief implement ghostfacenet for face verification
 * @author 20241114 Jason Yang
 */


/**************************************************************************
 *                        H E A D E R   F I L E S
 **************************************************************************/

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdarg.h"
#include <math.h>

#include "common.h"

#include "sp5k_modesw_api.h"
#include "sp5k_gfx_api.h"
#include "sp5k_dbg_api.h"
#include "sp5k_nn_api.h"
#include "sp5k_os_api.h"
#include "sp5k_rsvblk_api.h"
#include "sp5k_fs_api.h"
#include "sp5k_utility_api.h"

#include "app_sys_cfg.h"
#include "app_com_api.h"
#include "app_dbg_api.h"
#include "app_res_def.h"
#include "app_util.h"
#include "app_openpv.h"

#include "app_nn_model.h"

#include "app_nn_dms.h"
#include "app_nn_utils_gfx.h"
#include "app_nn_utils_math.h"


/**************************************************************************
 *                          C O N S T A N T S
 **************************************************************************/

#define	IMG_W	128	// nn input image width
#define	IMG_H	128	// nn input image height
#define EMBEDDING_VECTOR_SIZE	512
#define REGISTER_VECTOR_SIZE 10
#define SIM_THRESH 0.6

#define CHECK_PERFORMANCE	FALSE

/**************************************************************************
 *                              M A C R O S
 **************************************************************************/

#define ROUND_DOWN_TO_DIVISIBLE(num,div)  ( (UINT32)(num) & -(UINT32)(div) )
#define ROUND_UP_TO_DIVISIBLE(num,div)    ROUND_DOWN_TO_DIVISIBLE( (UINT32)(num) + (div) - 1, div )

/**************************************************************************
 *                          D A T A   T Y P E S
 **************************************************************************/

/**************************************************************************
 *                  E X T E R N A L   R E F E R E N C E
 **************************************************************************/

/**************************************************************************
 *              F U N C T I O N   D E C L A R A T I O N S
 **************************************************************************/

static UINT32 nnModelFvInfer(UINT32 appNnModelHandle, void *pnnInput, UINT32 nnInputSz, void *pnnOutput, UINT32 nnOutputSz);
static UINT32 nnModelFvCreate(UINT32 appNnModelHandle);
static UINT32 nnModelFvDestroy(UINT32 appNnModelHandle);

/**************************************************************************
 *                        G L O B A L   D A T A
 **************************************************************************/

static const appNnModelCommonCfg_t nnModelCfgList[] = {
	[0] = {
		.appNnModelType				 = APP_NN_MODEL_FV,
		.nn_model_name				 = "FV",
		.sp5kNnModelType			 = SP5K_NN_MODEL_NORMAL, /* sp5kNnModelList_t */
		.bin_name					 = OBJFV_PATH, /* specify Model Network Binaray File */
		.input_tensor_fmt			 = SP5K_NN_INPUT_TENSOR_FMT_RGB888_PLANE, /* sp5kNnInputTensorFmt_t : specify Input Tensor Format */
		.tensor_dim_order			 = SP5K_NN_TENSOR_DIM_ORDER_NHWC,
		.mean[0]					 = (1275*SP5K_NN_NORMALIZE_BASE_VALUE)/10,
		.mean[1]					 = (1275*SP5K_NN_NORMALIZE_BASE_VALUE)/10,
		.mean[2]					 = (1275*SP5K_NN_NORMALIZE_BASE_VALUE)/10,
		.scale						 = 128*SP5K_NN_NORMALIZE_BASE_VALUE,
		.is_file_encrypted			 = 0, /* 1 means nb file is encrypted */
        .file_encrypt_key			 = {0x11u, 0x22u, 0x33u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u, 0x99u, 0x00u, 0xAAu, 0xBBu, 0xCCu, 0xDDu, 0xEEu, 0xFFu}, /* */
		.fp_create					 = nnModelFvCreate, /* fp_appNnModelCreate_t */
		.fp_infer					 = nnModelFvInfer, /* fp_appNnModelInfer_t */
		.fp_destroy					 = nnModelFvDestroy, /* fp_appNnModelDestory_t */
	},
};

static float REGISTER_EMBEDDINGS[REGISTER_VECTOR_SIZE][EMBEDDING_VECTOR_SIZE];
static float REGISTERED_EMBEDDING[EMBEDDING_VECTOR_SIZE];
static float CHECK_EMBEDDING[EMBEDDING_VECTOR_SIZE];

UINT8 Register_cnt = 0;
UINT8 check_cnt = 0;
BOOL isChecked = FALSE;
char *embedding_name = NULL;
static float sim = 0;

/**************************************************************************
 *                        F U C T I O N  L I S T
 **************************************************************************/

static void verify_user() __attribute__((noinline));
static void verify_user()
{
	UINT32 i, j;
	UINT8 pBuf[8*EMBEDDING_VECTOR_SIZE];
	UINT32 size = sizeof(pBuf[0]) * 8 * EMBEDDING_VECTOR_SIZE;
	char verify_name[64];
	sprintf(verify_name, "B:\\%s_embedding.txt", embedding_name);
	printf("verify_name: %s\n", verify_name);

    //Reading registered vector
	if(sp5kFsFileExist(verify_name)==SUCCESS){
		ReadFile(verify_name, pBuf, size);
		UINT8 *ptr = strtok(pBuf, ",");
		j = 0;
		while (ptr != NULL) {
			REGISTERED_EMBEDDING[j] = strtof(ptr, NULL);
			ptr = strtok(NULL, ",");
			j++;
		}

		// Calculate average vector
		for (i = 0; i < EMBEDDING_VECTOR_SIZE; i++) {
			float sum = 0;
			for (j = 0; j < REGISTER_VECTOR_SIZE; j++) {
				sum += REGISTER_EMBEDDINGS[j][i];
			}
			float avg = sum / (float) REGISTER_VECTOR_SIZE;
			CHECK_EMBEDDING[i] = avg;
		}

		similarity(CHECK_EMBEDDING, EMBEDDING_VECTOR_SIZE);
		printf("SIM = [%f] \n", sim);
		if (sim < SIM_THRESH) {
			printf("User is not verified \n");
			appSoundPlay((UINT8*)appRootGet("A:\\RO_RES\\WAV\\verification_fail.WAV"));
			sp5kOsThreadSleep(50);
		} else {
			printf("User is verified \n");
			appSoundPlay((UINT8*)appRootGet("A:\\RO_RES\\WAV\\verification_success.WAV"));
			sp5kOsThreadSleep(50);
		}
	} else {
		printf("User does not exist!!! \n");
	}
	sp5kFree(embedding_name);
}

static void register_database() __attribute__((noinline));
static void register_database()
{
	char pBuf[200];
	char pCopy[sizeof(pBuf)];
	memset(pBuf, 0, sizeof(pBuf));
	BOOL embedding_exists = FALSE;
	if(sp5kFsFileExist("B:\\database.txt")==SUCCESS){
		// if at least one user is already registered, then modify the list
		printf("database already exists!!!\n");
		UINT32 size = sizeof(pBuf[0]) * 200;
		ReadFile("B:\\database.txt", pBuf, size);
		UINT32 length = strlen(pBuf);
		strcpy(pCopy, pBuf);
		char *ptr = strtok(pCopy, ",");

		while (ptr != NULL) {
			if (strcmp(ptr, embedding_name) == 0) {
				embedding_exists = TRUE;
			}
			ptr = strtok(NULL, ",");
		}

		if (!embedding_exists) {
			sprintf(&pBuf[length], ",%s", embedding_name);
		}

		sp5kFsFileDelete("B:\\database.txt");
		sp5kOsThreadSleep(100);
		WriteFile("B:\\database.txt", pBuf, sizeof(pBuf));
	}
	else{
		// if none is registered, then create the list
		sprintf(&pBuf[0], ",%s", embedding_name);
		WriteFile("B:\\database.txt", pBuf, sizeof(pBuf));
	}
}

static void register_user() __attribute__((noinline));
static void register_user()
{
	UINT32 i, j;
	char save_name[64];

    // Calculate average vector
	for (i = 0; i < EMBEDDING_VECTOR_SIZE; i++) {
		float sum = 0;
		for (j = 0; j < REGISTER_VECTOR_SIZE; j++) {
			sum += REGISTER_EMBEDDINGS[j][i];
		}
		float avg = sum / (float) REGISTER_VECTOR_SIZE;
		REGISTERED_EMBEDDING[i] = avg;
	}

	normalize(&REGISTERED_EMBEDDING, EMBEDDING_VECTOR_SIZE);

	char *pBuf = (char *)sp5kMalloc(12 * EMBEDDING_VECTOR_SIZE * sizeof(char));
	memset(pBuf, 0, sizeof(pBuf));
	UINT32 length = 0;

	for (i = 0; i < EMBEDDING_VECTOR_SIZE; i++) {
		length += sprintf((pBuf+length),",%.6f", REGISTERED_EMBEDDING[i]);
	}

	sprintf(save_name, "B:\\%s_embedding.txt", embedding_name);
	if(sp5kFsFileExist(save_name)==SUCCESS){
		sp5kFsFileDelete(save_name);
	}

	sp5kOsThreadSleep(50);
	WriteFile(save_name, pBuf, 12 * EMBEDDING_VECTOR_SIZE * sizeof(char));
	appSoundPlay((UINT8*)appRootGet("A:\\RO_RES\\WAV\\registration_finish.WAV"));
	sp5kOsThreadSleep(50);
	printf("User is registered\n");
	register_database();
	sp5kFree(pBuf);
	sp5kFree(embedding_name);
}

void similarity(float* vec, UINT16 size)  __attribute__((noinline));
void similarity(float* vec, UINT16 size) 
{
    // Calculate cosine similarity with registered vector
	float sum = 0;
	UINT16 i;
	for(i = 0; i < size; i++){
		sum += vec[i] * REGISTERED_EMBEDDING[i];
	}
	sim = sum;
}

void normalize(float* vec, UINT16 size)  __attribute__((noinline));
void normalize(float* vec, UINT16 size) 
{
	float norm = 0;
	UINT16 i;
	for(i = 0; i < size; i++){
		norm += vec[i] * vec[i];
	}

	norm = sqrtf(norm);

	for(i = 0; i < size; i++){
		vec[i] /= norm;
	}
}

static UINT32
nnModelFvInfer(
	UINT32 appNnModelHandle,
	void *pnnInput,
	UINT32 nnInputSz,
	void *pnnOutput,
	UINT32 nnOutputSz
)
{
	UINT32 t_infer = sp5kMsTimeGet();
	UINT32 ret = SUCCESS;
	
	if (detected_face.stage == DMS_FACE_VERIFICATION && dms_frame.pbuf != NULL) {
		UINT32 i, j;
		nnModelEmtObj_t *pnnModelObj = (nnModelEmtObj_t *)appNnModelHandle;
		sp5kNnFrameInfo_t *pnnImg = &dms_frame;
		sp5kNnModelNormalInput_t normalIn;
		UINT32 dw, dh;
		roi_box_t crop_box;

		UINT32 retTensorOutnum = 0;
		sp5kNnIOTensorInfo_t *ptensor;
		mat_f32_t mat_embedding;
		matZeros(&mat_embedding, 1, EMBEDDING_VECTOR_SIZE);
		float* out_addr_arr[] = {mat_embedding.data};

		mat_f32_t normalized_embedding;
		matZeros(&normalized_embedding, 1, EMBEDDING_VECTOR_SIZE);

		memset(&normalIn, 0, sizeof(normalIn));
		normalIn.image.pbuf = sp5kYuvBufferAlloc(IMG_W, IMG_H);

		dw = detected_face.bbox.w * 10/100;
		dh = detected_face.bbox.h * 10/100;
		crop_box.x = max(detected_face.bbox.x - dw, 0);
		crop_box.y = max(detected_face.bbox.y - dh, 0);
		crop_box.w = min(detected_face.bbox.w + dw, IMAGE_W - crop_box.x);
		crop_box.h = min(detected_face.bbox.h + dh, IMAGE_H - crop_box.y);
		DMS_DEBUG("crop bbox: (%d, %d, %d, %d)", crop_box.x, crop_box.y, crop_box.w, crop_box.h);

		preprocCropResizeImage(pnnImg, &normalIn.image, crop_box.x, crop_box.y, crop_box.w, crop_box.h, IMG_W, IMG_H);

		ret = sp5kNnRun(SP5K_NN_MODEL_NORMAL, pnnModelObj->sp5kNnModelHandle,
				(void *)&normalIn, sizeof(normalIn),
				(void *)pnnModelObj->nnTensorOutput, sizeof(pnnModelObj->nnTensorOutput), &retTensorOutnum);

		if (ret == SUCCESS) {
			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 1);

			#if CHECK_PERFORMANCE
			sp5kNnModelPerformanceInfo_t perf;
			memset(&perf, 0, sizeof(perf));
			sp5kNnModelControl(SP5K_NN_CTRL_MODEL_PERFORMANCE_GET, pnnModelObj->sp5kNnModelHandle, &perf, sizeof(perf));
			DMS_INFO("runtime_npu=%d us, runtime_preprocess=%d us, runtime_postprocess=%d us", perf.runtime_npu, perf.runtime_preprocess, perf.runtime_postprocess);
			DMS_INFO("bw_total=%d bps, bw_read=%d bps, bw_write=%d bps", perf.bw_total, perf.bw_read, perf.bw_write);
			#endif

			ptensor = pnnModelObj->nnTensorOutput;
			for (i=0; i<retTensorOutnum; i++) {
				sp5kNnModelControl(
					SP5K_NN_CTRL_MISC_CONVERT_TENSOR_TO_FP32_ARRAY,
					ptensor[i].buffer_addr,
					ptensor[i].element_sz * ptensor[i].element_total_num,
					ptensor[i].data_format,
					ptensor[i].quant_info_type, &ptensor[i].quant_info,
					out_addr_arr[i], sizeof(float) * ptensor[i].element_total_num
				);
			}

			normalize(mat_embedding.data, EMBEDDING_VECTOR_SIZE);
			similarity(mat_embedding.data, EMBEDDING_VECTOR_SIZE);

			if (Register_cnt > 0) {
				memcpy(&REGISTER_EMBEDDINGS[Register_cnt-1], mat_embedding.data, sizeof(float) * EMBEDDING_VECTOR_SIZE);
				Register_cnt--;
				if (Register_cnt == 0) {
					// Average the embeddings and save
					register_user();
				}
			}

			if (check_cnt > 0 && isChecked == FALSE) {
				memcpy(&REGISTER_EMBEDDINGS[check_cnt-1], mat_embedding.data, sizeof(float) * EMBEDDING_VECTOR_SIZE);
				check_cnt--;
				if (check_cnt == 0) {
					verify_user();
					isChecked = TRUE;
				}
			}

			detected_face.stage += 1;
			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 0);
		}
		sp5kYuvBufferFree(normalIn.image.pbuf);
		matFree(&mat_embedding);
		matFree(&normalized_embedding);
		DMS_INFO("STAGE [%d->%d / %d]: %d ms", DMS_FACE_VERIFICATION, detected_face.stage, DMS_LAST_STAGE, sp5kMsTimeGet() - t_infer);

		if (detected_face.stage==DMS_LAST_STAGE) {
			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 1);
			run_dms();
			reset_detected_face();
			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 0);
		} else if (detected_face.stage>DMS_LAST_STAGE) {
			DMS_WARN("DMS stage control error. Reset stage");
			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 1);
			reset_detected_face();
			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 0);
		}
	}
	return ret;
}



static UINT32
nnModelFvCreate(
	UINT32 appNnModelHandle
)
{
	ATAM_MSG_G("nnModelCreate");
	nnModelEmtObj_t *pnnModelObj = (nnModelEmtObj_t *)appNnModelHandle;
	if( pnnModelObj && !pnnModelObj->sp5kNnModelHandle){
		memset(pnnModelObj->nnTensorOutput, 0, sizeof(pnnModelObj->nnTensorOutput));

		pnnModelObj->sp5kNnModelHandle = appNnMiscModelCommonCreate(&pnnModelObj->nnModelCfg);
	}

	return pnnModelObj->sp5kNnModelHandle;
}


static UINT32
nnModelFvDestroy(
	UINT32 appNnModelHandle
)
{
	ATAM_MSG_G("nnModelDestroy");
	nnModelEmtObj_t *pnnModelObj = (nnModelEmtObj_t *)appNnModelHandle;

	if ( pnnModelObj ) {
		if ( pnnModelObj->sp5kNnModelHandle ) {
			sp5kNnModelDestroy(pnnModelObj->sp5kNnModelHandle);
		}

		pnnModelObj->sp5kNnModelHandle = 0;

		sp5kFree((void *)pnnModelObj);
		pnnModelObj = NULL;
	}

	return SUCCESS;
}


UINT32
appNnModelFvNew(
	UINT32 appNnModelType,
	appNnModelRuntimeObj_t *prunObj
)
{
	UINT32 i, isfind = 0;
	nnModelEmtObj_t *pnnModelObj = sp5kMalloc(sizeof(nnModelEmtObj_t));

	HOST_ASSERT( prunObj && pnnModelObj );
	memset( prunObj, 0, sizeof(*prunObj));
	memset( pnnModelObj, 0, sizeof(*pnnModelObj));

	for( i = 0; i < (sizeof(nnModelCfgList)/sizeof(nnModelCfgList[0])) ; i++ ){
		if ( appNnModelType == nnModelCfgList[i].appNnModelType ) {
			pnnModelObj->nnModelCfg = nnModelCfgList[i];
			isfind = 1;
			break;
		}
	}

	HOST_ASSERT_MSG( isfind , "appNnModelType %x not find \n", appNnModelType );
	appNnMiscModelRuntimeObjFill(&pnnModelObj->nnModelCfg, (UINT32)pnnModelObj, prunObj);

	return SUCCESS;
}

UINT32
appNnModelFvNew2(
	UINT32 appNnModelType,
	appNnModelRuntimeObj_t *prunObj
)
{
	UINT32 ret;
	nnModelEmtObj_t *pnnModelObj;

	ret = appNnModelFvNew(appNnModelType, prunObj);

	pnnModelObj = (nnModelEmtObj_t *)prunObj->appNnModelHandle;

	return ret;
}
