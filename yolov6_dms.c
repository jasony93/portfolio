/**
 * @file   yolov6_dms.c
 * @brief  implement yolov6 detection model
 * @author 20241113 Jason Yang
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
#include "sp5k_modesw_api.h"
#include "sp5k_global_api.h"
#include "sp5k_media_api.h"

#include "app_sys_cfg.h"
#include "app_dbg_api.h"
#include "app_res_def.h"
#include "app_util.h"
#include "app_openpv.h"
#include "app_com_def.h"
#include "app_com_api.h"
#include "app_osd_api.h"
#include "app_aealg_api.h"
#include "app_gps.h"
#include "app_video.h"

#include "app_nn_model.h"
#include "app_nn_model_def.h"

#include "app_nn_dms.h"
#include "app_nn_utils_gfx.h"

/**************************************************************************
 *                          C O N S T A N T S
 **************************************************************************/

#define	YOLO_NN_INPUT_W	(640)
#define	YOLO_NN_INPUT_H	(640)
#define	YOLO_NUM_ANCHOR	((YOLO_NN_INPUT_W/8) * (YOLO_NN_INPUT_H/8) + (YOLO_NN_INPUT_W/16) * (YOLO_NN_INPUT_H/16) + (YOLO_NN_INPUT_W/32) * (YOLO_NN_INPUT_H/32))

#define CHECK_PERFORMANCE	FALSE
#define DRAW_BOX 			FALSE
#define DRAW_LABEL 			FALSE

static enum CLASS_INDEX{
	CIGARETTE,
	FACE,
	PHONE,
	GLASSES,
	MASK,
	SEATBELT,
	NUM_CLASSES
};
static const char label_names_yolov6_dms[][LABEL_STRING_LENGTH] = {
	"cigarette", "face", "phone", "glasses", "mask", "seatbelt"
};

// ROI zoom multiples - left, right, top, bottom
static UINT8 zoom_grid[4] = {1, 1, 1, 2};

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

void float2uint16(float *from_data, UINT16 *to_data, UINT32 length) __attribute__((noinline));
void clear_info(yolov6_t *obj_info, UINT16 num) __attribute__((noinline));
void dist2box(float *bbox, float *reg_dist_list, UINT16 *anchor_points, UINT16 *stride_tensor) __attribute__((noinline));
void pred_candidates_(UINT16 *cls_score_list_, UINT16 *count, UINT16 *candidate_idx, UINT16 *candidate_conf, UINT8 * candidate_cls, nnModelYoloCfg_t *pyoloCfg) __attribute__((noinline));
void drawer(yolov6_t *obj_info, UINT16 valid_num) __attribute__((noinline));
void anchor_init() __attribute__((noinline));

static UINT32 nnModelYoloInfer(UINT32 appNnModelYoloObj, void *pnnInput, UINT32 nnInputSz, void *pnnOutput, UINT32 nnOutputSz, BOOL bRear);
static UINT32 nnModelYoloCreate(UINT32 appNnModelHandle);
static UINT32 nnModelYoloDestroy(UINT32 appNnModelYoloObj);

/**************************************************************************
 *                        G L O B A L   D A T A
 **************************************************************************/

global_face_t detected_face = {0,};
sp5kNnFrameInfo_t dms_frame = {0,};

yolov6_t *pyolo_objects=NULL;
yolov6_t *pyolo_faces;
yolov6_t *pyolo_phones;
yolov6_t *pyolo_cigarettes;
UINT16 num_objects=0;
UINT8 num_faces=0, num_phones=0, num_cigarettes=0, num_glasses=0;

static appNnDataObjectInfo_t* g_detected_object = NULL;
static UINT32 g_num_of_detected_object = 0;
static UINT16 g_anchor_points[YOLO_NUM_ANCHOR * 2];
static UINT16 g_stride_tensor[YOLO_NUM_ANCHOR];

static face_box_t face_boxes[5]={0,};
static roi_box_t crop_box={0,};
static struct mavg_box_s { SINT32 x; SINT32 y; SINT32 w; SINT32 h; } mavg_box = {0,};
static UINT32 frame_cnt=0, face_cnt=0;
static BOOL bFaceDetected, bZoom=FALSE;

static const nnModelYoloCfg_t nnModelYoloCfgList[] = {
	[0] = {
		.cfg.appNnModelType 			 = APP_NN_MODEL_YOLOV6,
		.cfg.nn_model_name				 = "yolov6m",
		.cfg.sp5kNnModelType			 = SP5K_NN_MODEL_NORMAL, /* sp5kNnModelList_t */
		.cfg.bin_name					 = OBJYOLOV6_PATH, /* specify Model Network Binaray File */

		.cfg.input_tensor_fmt			 = SP5K_NN_INPUT_TENSOR_FMT_RGB888_PLANE, /* sp5kNnInputTensorFmt_t : specify Input Tensor Format */
		.cfg.tensor_dim_order			 = SP5K_NN_TENSOR_DIM_ORDER_NCHW, /* sp5kNnTensorDimOrder_t : specify tensor dim order, based on framework */
		.cfg.mean[0]					 = 0, /* SP5K_NN_NORMALIZE_BASE_VALUE means 1.0 */
		.cfg.mean[1]					 = 0, /* SP5K_NN_NORMALIZE_BASE_VALUE means 1.0 */
		.cfg.mean[2]					 = 0, /* SP5K_NN_NORMALIZE_BASE_VALUE means 1.0 */
		.cfg.scale						 = 255 * SP5K_NN_NORMALIZE_BASE_VALUE, /* SP5K_NN_NORMALIZE_BASE_VALUE means 1.0 */

		.cfg.is_file_encrypted			 = 0, /* 1 means nb file is encrypted */
        .cfg.file_encrypt_key			 = {0x11u, 0x22u, 0x33u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u, 0x99u, 0x00u, 0xAAu, 0xBBu, 0xCCu, 0xDDu, 0xEEu, 0xFFu}, /* */

		.cfg.fp_create					 = nnModelYoloCreate, /* fp_appNnModelCreate_t */
		.cfg.fp_infer					 = nnModelYoloInfer, /* fp_appNnModelInfer_t */
		.cfg.fp_destroy 				 = nnModelYoloDestroy, /* fp_appNnModelDestory_t */

		.yolo_generation   = YOLO_V2,
		.output_tensor_num = 1,
		.anchor_box[0]     = { {572730, 677385}, {1874460, 2062530}, {3338430, 5474340}, {7882820, 3527780}, {9770520, 9168280} },  /* out[0] is 13*13 */
		.anchor_box_Num[0] = 5,
		.classes_total_num = NUM_CLASSES,
		.classes_label	   = (char * )label_names_yolov6_dms,
		.input_layer_w	   = YOLO_NN_INPUT_W,
		.input_layer_h	   = YOLO_NN_INPUT_H,
		.thresh_prob	   = ( 50 * SP5K_NN_DENOMINATOR_BASE ) / 100,  /* 0.5  * YOLO_OBJ_CONF_THRD_BASE */
		.thresh_nms 	   = ( 60 * SP5K_NN_DENOMINATOR_BASE ) / 100,
		.bInputImageuseLetterBox = 0,
	},
};

/**************************************************************************
 *                        F U C T I O N  L I S T
 **************************************************************************/

static UINT32 nnYoloBoxesIouCalc(yolov6_t *obj_info, UINT16 i, UINT16 j)
{
	SINT32 box1_w = obj_info[i].w;
	SINT32 box1_h = obj_info[i].h;
	SINT32 box2_w = obj_info[j].w;
	SINT32 box2_h = obj_info[j].h;

	SINT32 box1_x = obj_info[i].cx;
	SINT32 box1_y = obj_info[i].cy;
	SINT32 box2_x = obj_info[j].cx;
	SINT32 box2_y = obj_info[j].cy;
	SINT32 IOU = 0;

	SINT32 endx, startx, width, area1, area2, area;
	SINT32 endy, starty, height;

	endx   = MAX(box1_x + box1_w, box2_x + box2_w);
	startx = MIN(box1_x, box2_x);
	width  = box1_w + box2_w - (endx - startx );

	endy   = MAX(box1_y + box1_h, box2_y + box2_h);
	starty = MIN(box1_y, box2_y);
	height = box1_h + box2_h -(endy - starty );

	if ( width > 0 && height > 0 ) {
		area1 = box1_w * box1_h;
		area2 = box2_w * box2_h;
		area  = width * height;
		IOU   = (area * SP5K_NN_DENOMINATOR_BASE) / (area1 + area2 - area);
	}

	return IOU;
}

void float2uint16(float *from_data, UINT16 *to_data, UINT32 length)
{
	// * 10000
	UINT32 i;
	for(i=0; i<length; i++)
	{
		to_data[i] = (UINT16)(from_data[i] * 10000);
	}
}

void clear_info(yolov6_t *obj_info, UINT16 num)
{
	obj_info[num].idx = 65535;
	obj_info[num].cls = 255;
	obj_info[num].conf = 0;
	obj_info[num].cx = 0;
	obj_info[num].cy = 0;
	obj_info[num].w = 0;
	obj_info[num].h = 0;
}

void dist2box(float *bbox, float *reg_dist_list, UINT16 *anchor_points, UINT16 *stride_tensor)
{
	UINT16 i, j;
	for(i = 0; i < YOLO_NUM_ANCHOR; i++)
	{
		for(j = 0; j < 4; j++)
		{
			if(j == 0)
			{
				reg_dist_list[4 * i + j] = (((float)anchor_points[i] / 10.0) - reg_dist_list[4 * i + j]);
			}
			else if(j == 1)
			{
				reg_dist_list[4 * i + j] = (((float)anchor_points[YOLO_NUM_ANCHOR + i] / 10.0) - reg_dist_list[4 * i + j]);
			}
            else if(j == 2)
            {
                reg_dist_list[4 * i + j] = (((float)anchor_points[i] / 10.0) + reg_dist_list[4 * i + j]);
            }
            else
            {
                reg_dist_list[4 * i + j] = (((float)anchor_points[YOLO_NUM_ANCHOR + i] / 10.0) + reg_dist_list[4 * i + j]);
            }
		}
        bbox[4 * i] = ((reg_dist_list[4 * i] + reg_dist_list[4 * i + 2]) / 2.0) * (float)stride_tensor[i];
		bbox[4 * i + 1] = ((reg_dist_list[4 * i + 1] + reg_dist_list[4 * i + 3]) / 2.0) * (float)stride_tensor[i];
		bbox[4 * i + 2] = (reg_dist_list[4 * i + 2] - reg_dist_list[4 * i ] ) * (float)stride_tensor[i];
		bbox[4 * i + 3] = (reg_dist_list[4 * i + 3] - reg_dist_list[4 * i + 1]) * (float)stride_tensor[i];
	}
}


void pred_candidates_(UINT16 *cls_score_list_, UINT16 *count, UINT16 *candidate_idx, UINT16 *candidate_conf, UINT8 * candidate_cls, nnModelYoloCfg_t *pyoloCfg)
{
	UINT16 i, j;
	UINT16 max_val;
	UINT16 cnt;
	cnt=0;
	UINT8 num_class = pyoloCfg->classes_total_num;

	for(i=0; i<YOLO_NUM_ANCHOR; i++)
	{
		max_val = 0;

		for(j=0; j<num_class; j++)
		{
			if(cls_score_list_[num_class*i+j] > max_val && cls_score_list_[num_class*i+j] > 5000)
			{
				max_val = cls_score_list_[num_class*i+j];
				candidate_idx[cnt] = i;
				candidate_conf[cnt] = max_val;
				candidate_cls[cnt] = j;
				cnt++;
			}
			else
			{
				candidate_idx[cnt] = 0;
				candidate_conf[cnt] = 0;
				candidate_cls[cnt] = 255;
			}
		}
	}
	*count = cnt;
}

/* change scale */
void wraper(yolov6_t *summary, UINT16 *cls_score_list_, float *bbox, UINT16 *count, UINT16 *candidate_idx, UINT16 *candidate_conf, UINT8 *candidate_cls, nnModelYoloCfg_t *pyoloCfg)
{
	UINT16 i;
	UINT16 tmp_idx;

	if(face_cnt < 5){
		for(i = 0; i < *count; i++)
		{
			tmp_idx = candidate_idx[i];
			summary[i].idx = tmp_idx;
			summary[i].cls = candidate_cls[i];
			summary[i].conf = candidate_conf[i];
			summary[i].cx = (UINT32)(bbox[4 * tmp_idx] * IMAGE_W / YOLO_NN_INPUT_W);
			summary[i].cy = (UINT32)(bbox[4 * tmp_idx + 1] * IMAGE_H / YOLO_NN_INPUT_H);
			summary[i].w = (UINT32)(bbox[4 * tmp_idx + 2] * IMAGE_W / YOLO_NN_INPUT_W);
			summary[i].h = (UINT32)(bbox[4 * tmp_idx + 3] * IMAGE_H / YOLO_NN_INPUT_H);
		}
	}else{
		for(i = 0; i < *count; i++)
		{
			tmp_idx = candidate_idx[i];
			summary[i].idx = tmp_idx;
			summary[i].cls = candidate_cls[i];
			summary[i].conf = candidate_conf[i];
			summary[i].cx = (UINT32)(bbox[4*tmp_idx] * crop_box.w / YOLO_NN_INPUT_W) + crop_box.x;
			summary[i].cy = (UINT32)(bbox[4*tmp_idx+1] * crop_box.h / YOLO_NN_INPUT_H) + crop_box.y;
			summary[i].w = (UINT32)(bbox[4*tmp_idx+2] * crop_box.w / YOLO_NN_INPUT_W);
			summary[i].h = (UINT32)(bbox[4*tmp_idx+3] * crop_box.h / YOLO_NN_INPUT_H);
		}
	}

}

static yolov6_t* yolov6nms(yolov6_t *obj_info, UINT16 *valid_num, nnModelYoloCfg_t *pyoloCfg)
{
	UINT16 i,j;
	UINT16 val = *valid_num;
	UINT16 new_valid_num = 0;

	for(i = 0; i < val; i++)
	{
		if(obj_info[i].conf < pyoloCfg->thresh_prob)
		{
			clear_info(obj_info, i);
			continue;
		}

		for(j = i + 1; j < val; j++)
		{
			if(obj_info[j].conf < pyoloCfg->thresh_prob)
			{
				clear_info(obj_info, j);
				continue;
			}

			if(obj_info[i].cls == obj_info[j].cls)
			{
				UINT32 iou = nnYoloBoxesIouCalc(obj_info, i, j);

				if(iou > pyoloCfg->thresh_nms)
				{
					if(obj_info[i].conf > obj_info[j].conf)
					{
						clear_info(obj_info, j);
					}
					else
					{
						clear_info(obj_info, i);
						break;
					}
				}
			}
		}
	}

	for (i = 0; i < val; i++)
	{
		if (obj_info[i].conf > 0)
		{
			yolov6_t tmp;
			tmp = obj_info[i];
			clear_info(obj_info, i);
			obj_info[new_valid_num] = tmp;
			new_valid_num++;
		}
	}

	*valid_num = new_valid_num;

	return obj_info;

}

void drawer(yolov6_t *obj_info, UINT16 valid_num)
{
	drawStart();
	setDrawLineWidth(2);

	UINT32 i, w_2, h_2, x0, y0, x1, y1;
	for(i=0;i<valid_num;i++)
	{
		if(obj_info[i].conf != 0)
		{
			BOOL drawing = TRUE;
			switch (obj_info[i].cls) {
				case CIGARETTE	: setDrawColor(RGB565_RED,	128);	break;
				case FACE		: setDrawColor(RGB565_WHITE,128);	break;
				case PHONE		: setDrawColor(RGB565_BLUE,	128);	break;
				case GLASSES	: setDrawColor(RGB565_GREEN,128);	break;
				case MASK		: setDrawColor(RGB565_BLACK,128);	break;
				default			: drawing = FALSE;					break;
			}
			if (drawing) {
				w_2 = obj_info[i].w/2;
				h_2 = obj_info[i].h/2;
				x0 = obj_info[i].cx - w_2;
				y0 = obj_info[i].cy - h_2;
				x1 = obj_info[i].cx + w_2;
				y1 = obj_info[i].cy + h_2;
				drawBox(x0, y0, x1, y1);
#if DRAW_LABEL
				char label[LABEL_STRING_LENGTH];
				sprintf(&label, "%s: %d", label_names_yolov6_dms[obj_info[i].cls], obj_info[i].conf);
				drawString(x1, y1, label);
#endif
			}
		}
	}
	drawFinish();
}


void anchor_init()
{
	UINT16 i, j, k;
	UINT16 f11, f12, f21, f22, f31, f32;

	f11 = (UINT16)(YOLO_NN_INPUT_W / 8);
	f12 = (UINT16)(YOLO_NN_INPUT_H / 8);
	f21 = (UINT16)(YOLO_NN_INPUT_W / 16);
	f22 = (UINT16)(YOLO_NN_INPUT_H / 16);
	f31 = (UINT16)(YOLO_NN_INPUT_W / 32);
	f32 = (UINT16)(YOLO_NN_INPUT_H / 32);

    for(i=0; i < f12; i++)
    {
        for(j=0; j < f11; j++)
        {
            g_anchor_points[f11*i+j] = 10*j+5;
            g_anchor_points[(UINT16)YOLO_NUM_ANCHOR + f11*i+j] = 10*i+5;
        }

    }
    for(i=0; i < f22; i++)
    {
        for(j=0; j < f21; j++)
        {
            g_anchor_points[f11*f12 + f21*i+j] = 10*j+5;
            g_anchor_points[(UINT16)YOLO_NUM_ANCHOR + f11*f12 + f21*i+j] = 10*i+5;
        }

    }
    for(i=0; i < f32; i++)
    {
        for(j=0; j < f31; j++)
        {
            g_anchor_points[f11*f12 + f21*f22 + f31*i+j] = 10*j+5;
            g_anchor_points[(UINT16)YOLO_NUM_ANCHOR + f11*f12 + f21*f22 + f31*i+j] = 10*i+5;
        }

    }

    for(k=0; k < f11*f12; k++)
    {
        g_stride_tensor[k] = 8;
    }
    for(k=0; k < f21*f22; k++)
    {
        g_stride_tensor[f11*f12 + k] = 16;
    }
    for(k=0; k < f31*f32; k++)
    {
        g_stride_tensor[f11*f12 + f21*f22 + k] = 32;
    }
}

void set_object_information(yolov6_t *obj_info, UINT16 valid_num, BOOL b_rear)
{
	UINT32 i;
	memset(g_detected_object, 0, sizeof(appNnDataObjectInfo_t)*512);

	for(i = 0; i < valid_num; i++)
	{
		g_detected_object[i].label_idx = obj_info[i].cls;
		g_detected_object[i].conf = obj_info[i].conf;
		g_detected_object[i].objectBBox.roiX = obj_info[i].cx - obj_info[i].w/2;
		g_detected_object[i].objectBBox.roiY = obj_info[i].cy - obj_info[i].h/2;
		g_detected_object[i].objectBBox.roiW = obj_info[i].w;
		g_detected_object[i].objectBBox.roiH = obj_info[i].h;
	}

	g_num_of_detected_object = valid_num;
	num_objects = valid_num;
}

static UINT32
nnModelYoloInfer(
	UINT32 appNnModelYoloObj,
	void *pnnInput,
	UINT32 nnInputSz,
	void *pnnOutput,
	UINT32 nnOutputSz,
	BOOL bRear
)
{
	UINT32 t_infer = sp5kMsTimeGet();
	UINT32 ret = FAIL;

	nnModelYoloObj_t *pYoloObj = (nnModelYoloObj_t *)appNnModelYoloObj;
	appNnModelObjectDetInput_t *pnnObjIn, objectDetectInput;
	appNnDataObjectsList_t *pobjectsList = NULL;
	appNnDataCollectionInfo_t *pmeta;

	sp5kNnModelNormalInput_t normalIn;
	sp5kNnRoi_t validRoi;

	UINT32 retTensorOutnum = 0;
	nnModelYoloCfg_t *pyoloCfg;

	BOOL b_rear = bRear;		// rear cam flag

	if( pnnInput == NULL && nnInputSz == 0 && nnOutputSz == sizeof(appNnDataCollectionInfo_t) ){
		pmeta = (appNnDataCollectionInfo_t*)pnnOutput;

		memset(&objectDetectInput, 0, sizeof(objectDetectInput));
		objectDetectInput.image = pmeta->nnImage.img;
		pnnObjIn = &objectDetectInput;

		if( !pmeta->pobjects ){
			pobjectsList = sp5kMalloc(sizeof(*pmeta->pobjects));

			if( pobjectsList ){
				pobjectsList->validNum = 0;
				pmeta->pobjects = pobjectsList;
			}
		}else{
			pobjectsList = &pmeta->pobjects[pobjectsList->validNum];
		}

	}else{
		pnnObjIn = (appNnModelObjectDetInput_t *)pnnInput;
		pobjectsList = (appNnDataObjectsList_t *)pnnOutput;

		HOST_ASSERT_MSG( nnInputSz == sizeof(appNnModelObjectDetInput_t),
			"%d not match %d \n", nnInputSz, sizeof(appNnModelObjectDetInput_t) );

		HOST_ASSERT_MSG( nnOutputSz == sizeof(appNnDataObjectsList_t),
			"%d not match %d \n", nnOutputSz, sizeof(appNnDataObjectsList_t) );
	}
	HOST_ASSERT_MSG(pYoloObj, "%s not ready %p \n",__FUNCTION__, pYoloObj);

	if( pobjectsList && pYoloObj && pYoloObj->sp5kNnModelHandle && detected_face.stage == DMS_FACE_DETECTION ){
		UINT32 i;
		pyoloCfg = &pYoloObj->nnModelYoloCfg;
		copyImage(&pnnObjIn->image, &dms_frame);

        DMS_TRACE("frame size: (%d, %d) | roi: (%d, %d, %d, %d)", pnnObjIn->image.width, pnnObjIn->image.height,
            pnnObjIn->image.roiX, pnnObjIn->image.roiY, pnnObjIn->image.roiW, pnnObjIn->image.roiH);

        memset(&normalIn, 0, sizeof(normalIn));
        normalIn.image.pbuf = sp5kYuvBufferAlloc(YOLO_NN_INPUT_W, YOLO_NN_INPUT_H);// Select ROI for YOLO input

        // ROI for object detection - focuses on where the faces were detected from previous frames
        if (bZoom) {
            crop_box.x = max(mavg_box.x - (mavg_box.w * zoom_grid[0]), 0);
            crop_box.y = max(mavg_box.y - (mavg_box.h * zoom_grid[2]), 0);
            crop_box.w = min(mavg_box.w * (zoom_grid[1] + zoom_grid[0] + 1), IMAGE_W - crop_box.x);
            crop_box.h = min(mavg_box.h * (zoom_grid[2] + zoom_grid[3] + 1), IMAGE_H - crop_box.y);
            DMS_DEBUG("crop bbox: (%d, %d, %d, %d)", crop_box.x, crop_box.y, crop_box.w, crop_box.h);
        } else {
            crop_box.x = pnnObjIn->image.roiX;
            crop_box.y = pnnObjIn->image.roiY;
            crop_box.w = pnnObjIn->image.roiW;
            crop_box.h = pnnObjIn->image.roiH;
        }
        preprocCropResizeImage(pnnObjIn, &normalIn, crop_box.x, crop_box.y, crop_box.w, crop_box.h, YOLO_NN_INPUT_W, YOLO_NN_INPUT_H);

        DMS_DEBUG("input size: (%d, %d) | roi: (%d, %d, %d, %d)", normalIn.image.width, normalIn.image.height,
                    normalIn.image.roiX, normalIn.image.roiY, normalIn.image.roiW, normalIn.image.roiH);
    

		HOST_ASSERT_MSG(pYoloObj->nnModelYoloCfg.cfg.sp5kNnModelType == SP5K_NN_MODEL_NORMAL,
						"`yolov6 sp5kNnModelType` expected %d but found %d, ",
						SP5K_NN_MODEL_NORMAL, pYoloObj->nnModelYoloCfg.cfg.sp5kNnModelType);

		ret = sp5kNnRun(SP5K_NN_MODEL_NORMAL, pYoloObj->sp5kNnModelHandle,
				(void *)&normalIn, sizeof(normalIn), (void *)pYoloObj->nnTensorOutput,
				sizeof(pYoloObj->nnTensorOutput), &retTensorOutnum);

        // Postprocess
		if (ret == SUCCESS) {
			if( pnnObjIn->objectConfThrd && pnnObjIn->objectConfThrd <= YOLO_OBJ_CONF_THRD_BASE ){
				pyoloCfg->thresh_prob = pnnObjIn->objectConfThrd;
			}

			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 1);
			sp5kNnIOTensorInfo_t *ptensor = pYoloObj->nnTensorOutput;

			float *cls_score_list = sp5kMalloc(sizeof(float) * ptensor[0].element_total_num);
			float *reg_dist_list = sp5kMalloc(sizeof(float) * ptensor[1].element_total_num);
			float *out_addr_arr[] = {cls_score_list, reg_dist_list};

			float *bbox = sp5kMalloc(sizeof(float) * ptensor[1].element_total_num);

			UINT16 *cls_score_list_ = sp5kMalloc(sizeof(UINT16) * ptensor[0].element_total_num);
			UINT16 *candidate_idx = sp5kMalloc(sizeof(UINT16) * YOLO_NUM_ANCHOR);
			UINT16 *candidate_conf = sp5kMalloc(sizeof(UINT16) * YOLO_NUM_ANCHOR);

			UINT8 *candidate_cls = sp5kMalloc(sizeof(UINT8) * YOLO_NUM_ANCHOR);

			memset(candidate_idx, 0, sizeof(UINT16) * YOLO_NUM_ANCHOR);
			memset(candidate_conf, 0, sizeof(UINT16) * YOLO_NUM_ANCHOR);
			memset(candidate_cls, 255, sizeof(UINT8) * YOLO_NUM_ANCHOR);

			DMS_TRACE("retTensorOutnum==%d", retTensorOutnum);
			for (i=0; i<retTensorOutnum; i++) {
				sp5kNnModelControl(
					SP5K_NN_CTRL_MISC_CONVERT_TENSOR_TO_FP32_ARRAY,
					ptensor[i].buffer_addr,
					ptensor[i].element_sz * ptensor[i].element_total_num,
					ptensor[i].data_format,
					ptensor[i].quant_info_type, &ptensor[i].quant_info,
					out_addr_arr[i], sizeof(float) * ptensor[i].element_total_num
				);
				DMS_TRACE("output[%d] size = %d", i, ptensor[i].element_total_num);
			}
			float2uint16(cls_score_list, cls_score_list_, ptensor[0].element_total_num);

			UINT16 valid_num;
			dist2box(bbox, reg_dist_list, g_anchor_points, g_stride_tensor);
			pred_candidates_(cls_score_list_, &valid_num, candidate_idx, candidate_conf, candidate_cls, pyoloCfg);

			yolov6_t *obj_info = sp5kMalloc(sizeof(yolov6_t) * valid_num);

			wraper(obj_info, cls_score_list_, bbox, &valid_num, candidate_idx, candidate_conf, candidate_cls, pyoloCfg);
			pyolo_objects = yolov6nms(obj_info, &valid_num, pyoloCfg);

			// storing object info in global variables
			set_object_information(pyolo_objects, valid_num, b_rear);

            // Counting number of each object
			num_faces = 0;
			num_phones = 0;
			num_cigarettes = 0;
			num_glasses = 0;
			for(i=0; i < num_objects; i++)
			{
				if(pyolo_objects[i].cls == 0){
					num_cigarettes++;
				}else if (pyolo_objects[i].cls == 1){
					num_faces++;
				}else if (pyolo_objects[i].cls == 2){
					num_phones++;
				}else if (pyolo_objects[i].cls == 3){
					num_glasses++;
				}
			}

			pyolo_cigarettes = sp5kMalloc(sizeof(yolov6_t) * num_cigarettes);
			pyolo_faces = sp5kMalloc(sizeof(yolov6_t) * num_faces);
			pyolo_phones = sp5kMalloc(sizeof(yolov6_t) * num_phones);

			UINT8 cigarette_idx = 0;
			UINT8 face_idx = 0;
			UINT8 phone_idx = 0;

			for(i=0; i < num_objects; i++)
			{
				switch(pyolo_objects[i].cls){
					case CIGARETTE:
						pyolo_cigarettes[cigarette_idx] = pyolo_objects[i];
						cigarette_idx++;
						break;
					case FACE:
						pyolo_faces[face_idx] = pyolo_objects[i];
						face_idx++;
						break;
					case PHONE:
						pyolo_phones[phone_idx] = pyolo_objects[i];
						phone_idx++;
						break;
					case GLASSES:
					case MASK:
					case SEATBELT:
					default:
						break;
				}
				DMS_TRACE("%s detected | conf: %d", label_names_yolov6_dms[obj_info[i].cls], pyolo_objects[i].conf);
			}


            // Find the face with greatest area
			UINT32 max_idx=0, max_area=0;
			for(i=0; i < num_faces; i++)
			{
				UINT32 area = calculate_area(pyolo_faces[i]);
				if(area > max_area)
				{
					max_idx = i;
					max_area = area;
				}
			}

			yolov6_t *pyolo_face = &pyolo_faces[max_idx];
			face_box_t face_box;
			face_box.x = pyolo_face->cx - (pyolo_face->w / 2);
			face_box.y = pyolo_face->cy - (pyolo_face->h / 2);
			face_box.w = pyolo_face->w;
			face_box.h = pyolo_face->h;
			face_box.cx = pyolo_face->cx;
			face_box.cy = pyolo_face->cy;

			if (0 < pyolo_face->conf && pyolo_face->conf < ((1<<(sizeof(pyolo_face->conf)*8)) - 1)
				&& 0 < face_box.x && face_box.x <= IMAGE_W && 0 < face_box.y && face_box.y <= IMAGE_H
				&& 0 < face_box.w && face_box.w <= IMAGE_W && 0 < face_box.h && face_box.h <= IMAGE_H)
			{
				bFaceDetected = TRUE;
				if (detected_face.stage == DMS_FACE_DETECTION) {
					detected_face.bbox = face_box;
					detected_face.stage += 1;
				}
				// moving average for ROI
				UINT8 face_idx = face_cnt%5;
				DMS_TRACE("face bbox: (%d, %d, %d, %d)", face_box.x, face_box.y, face_box.w, face_box.h);
				DMS_TRACE("old%d bbox: (%d, %d, %d, %d)", face_idx, face_boxes[face_idx].x, face_boxes[face_idx].y, face_boxes[face_idx].w, face_boxes[face_idx].h);
				mavg_box.x += (face_box.x - face_boxes[face_idx].x)/5;
				mavg_box.y += (face_box.y - face_boxes[face_idx].y)/5;
				mavg_box.w += (SINT32)(face_box.w - face_boxes[face_idx].w)/5;
				mavg_box.h += (SINT32)(face_box.h - face_boxes[face_idx].h)/5;
				face_boxes[face_idx] = face_box;
				DMS_TRACE("new%d bbox: (%d, %d, %d, %d)", face_idx, face_boxes[face_idx].x, face_boxes[face_idx].y, face_boxes[face_idx].w, face_boxes[face_idx].h);
				face_cnt++;
				if (face_cnt > 5) {
					bZoom = TRUE;
				}
				DMS_TRACE("mavg bbox: (%d, %d, %d, %d)", mavg_box.x, mavg_box.y, mavg_box.w, mavg_box.h);
			}
			else
			{
				bFaceDetected = FALSE;
			}
			frame_cnt++;

			// if face is not detected for more than 2 frames, then reset face count
			if (frame_cnt - face_cnt > 2) {
				frame_cnt = 0;
				face_cnt = 0;
				memset(&mavg_box, 0, sizeof(mavg_box));
				memset(face_boxes, 0, sizeof(face_boxes));
				bZoom = FALSE;
			}

			#if DRAW_BOX
			drawer(pyolo_objects, valid_num);
			#endif

			sp5kSystemCfgSet(SP5K_SYS_VFP_ENABLE, 0);

			sp5kFree(cls_score_list);
			sp5kFree(cls_score_list_);
			sp5kFree(reg_dist_list);
			sp5kFree(bbox);
			sp5kFree(candidate_idx);
			sp5kFree(candidate_conf);
			sp5kFree(candidate_cls);
			sp5kFree(obj_info);
			sp5kFree(pyolo_faces);
			sp5kFree(pyolo_phones);
			sp5kFree(pyolo_cigarettes);
		}
		sp5kYuvBufferFree(normalIn.image.pbuf);

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
	APP_NN_YOLO_DBG_PRINT("E", "");
	return ret;
}

static UINT32
nnModelYoloCreate(
	UINT32 appNnModelHandle
)
{
	ATAM_MSG_G("nnModelYoloCreate");
	nnModelYoloObj_t *obj = (nnModelYoloObj_t *)appNnModelHandle;
	if( obj && !obj->sp5kNnModelHandle){
		memset(obj->nnTensorOutput, 0, sizeof(obj->nnTensorOutput));

		obj->sp5kNnModelHandle = appNnMiscModelCommonCreate(&obj->nnModelYoloCfg.cfg);

		obj->objectOutPoolMaxNum = YOLO_OBJECT_MAX_NUM;
		obj->objectOutPoolSz = sizeof( appNnDataObjectInfo_t ) * obj->objectOutPoolMaxNum;

		obj->pobjectInfoPool = sp5kMalloc(obj->objectOutPoolSz);

		HOST_ASSERT(obj->pobjectInfoPool);

		obj->pYoloBBoxList = sp5kMalloc(sizeof(nnYoloBoxeOutput_t)*YOLO_OBJECT_MAX_NUM);
		HOST_ASSERT( obj->pYoloBBoxList );

		memset(obj->pYoloBBoxList, 0, sizeof(nnYoloBoxeOutput_t)*YOLO_OBJECT_MAX_NUM);

		anchor_init();

		g_detected_object = sp5kMalloc(sizeof(appNnDataObjectInfo_t)*512);
		HOST_ASSERT( g_detected_object );
		memset(g_detected_object, 0, sizeof(appNnDataObjectInfo_t)*512);

		// DMS detected face
		init_detected_face();
		if (dms_frame.pbuf == NULL) {
			dms_frame.width = IMAGE_W;
			dms_frame.height = IMAGE_H;
			dms_frame.pbuf = sp5kYuvBufferAlloc(dms_frame.width, dms_frame.height);
		}
	}
	return obj->sp5kNnModelHandle;
}

UINT32
nnModelYoloDestroy(
	UINT32 appNnModelYoloObj
)
{
	ATAM_MSG_G("nnModelYoloDestroy");
    UINT32 i;
	nnModelYoloObj_t *pnnYoloObj = (nnModelYoloObj_t *)appNnModelYoloObj;

	if ( pnnYoloObj ) {
		if ( pnnYoloObj->sp5kNnModelHandle ) {
			sp5kNnModelDestroy(pnnYoloObj->sp5kNnModelHandle);
		}

		pnnYoloObj->sp5kNnModelHandle = 0;

		if (pnnYoloObj->pYoloBBoxList) {
			sp5kFree((void *)pnnYoloObj->pYoloBBoxList);
		}

		if (pnnYoloObj->pobjectInfoPool){
			sp5kFree((void *)pnnYoloObj->pobjectInfoPool);
		}

        if (pnnYoloObj->prevFrameInfo.pbuf){
            sp5kYuvBufferFree((void *)pnnYoloObj->prevFrameInfo.pbuf);
            pnnYoloObj->prevFrameInfo.pbuf = 0;
        }

		for( i = 0; i < pnnYoloObj->normLutvalidNum; i++){
			if(pnnYoloObj->pnormalizationLutList[i]){
				sp5kFree( pnnYoloObj->pnormalizationLutList[i] );
				pnnYoloObj->pnormalizationLutList[i] = NULL;
			}
		}

		pnnYoloObj->normLutvalidNum = 0;

		if (pnnYoloObj->pnnOutTmpBufF32){
			sp5kFree((void *)pnnYoloObj->pnnOutTmpBufF32);
		}

		for( i = 0; i < pnnYoloObj->outputLutsNum; i++){

			if( pnnYoloObj->outputLuts[i].pfp32LutList ){
				sp5kFree((void *)pnnYoloObj->outputLuts[i].pfp32LutList);
			}

			if( pnnYoloObj->outputLuts[i].paBoxIdxLuts ){
				sp5kFree((void *)pnnYoloObj->outputLuts[i].paBoxIdxLuts);
			}

			if( pnnYoloObj->outputLuts[i].paBoxIdxLutFcosIou ){
				sp5kFree((void *)pnnYoloObj->outputLuts[i].paBoxIdxLutFcosIou);
			}


		}

		sp5kFree((void *)pnnYoloObj);
		pnnYoloObj = NULL;

		if(g_detected_object != NULL)
		{
			sp5kFree((void *)g_detected_object);
			g_detected_object = NULL;
		}

		// DMS detected face
 		free_detected_face();
		if (dms_frame.pbuf != NULL) {
			sp5kYuvBufferFree(dms_frame.pbuf);
			dms_frame.pbuf = NULL;
		}
	}

	return SUCCESS;
}

UINT32
appNnModelYoloNew(
	UINT32 appNnModelType,
	appNnModelRuntimeObj_t *prunObj
)
{
	UINT32 i, isfind = 0;
	nnModelYoloObj_t *pnnModelObj = sp5kMalloc(sizeof(nnModelYoloObj_t));

	HOST_ASSERT( prunObj && pnnModelObj );
	memset( prunObj, 0, sizeof(*prunObj));
	memset( pnnModelObj, 0, sizeof(*pnnModelObj));

	for( i = 0; i < (sizeof(nnModelYoloCfgList)/sizeof(nnModelYoloCfgList[0])) ; i++ ){
		if ( appNnModelType == nnModelYoloCfgList[i].cfg.appNnModelType ) {
			pnnModelObj->nnModelYoloCfg = nnModelYoloCfgList[i];
			isfind = 1;
			break;
		}
	}

	HOST_ASSERT_MSG( isfind , "appNnModelType %x not find \n", appNnModelType );
	appNnMiscModelRuntimeObjFill(&pnnModelObj->nnModelYoloCfg.cfg, (UINT32)pnnModelObj, prunObj);

	return SUCCESS;
}

UINT32
appNnModelYoloNew2(
	UINT32 appNnModelType,
	appNnModelRuntimeObj_t *prunObj,
	int rearCam
)
{
	UINT32 ret;
	nnModelYoloObj_t *pnnModelObj;

	ret = appNnModelYoloNew(appNnModelType, prunObj);

	pnnModelObj = (nnModelYoloObj_t *)prunObj->appNnModelHandle;
	pnnModelObj->rearCam = rearCam;

	return ret;
}