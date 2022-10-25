/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsmeta.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)


static const int NUM_CLASSES_YOLO = 80;
static bool DICT_LPR_READY=false;
std::vector<std::string> DICT_LPR;
static bool DICT_VMN_READY=false;
std::vector<std::string> DICT_VMN;

void *set_metadata_ptr(std::array<float, 10> & arr)
{
    gfloat *user_metadata = (gfloat*)g_malloc0(10*sizeof(gfloat));

    for(int i = 0; i < 10; i++) {
       user_metadata[i] = arr[i];
    }
    return (void *)user_metadata;
}

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

struct ObjectPoint{
   float ctx;
   float cty;
   float width;
   float height;
   float confidence;
   int classId;
};


extern "C" bool NvDsInferParseCustomYolor(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* YOLOR implementations */
static NvDsInferParseObjectInfo convertBBoxYolor(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = clamp(bx1, 0, netW-1);
    float y1 = clamp(by1, 0, netH-1);
    float x2 = clamp(bx2, 0, netW-1);
    float y2 = clamp(by2, 0, netH-1);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW-1);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH-1);
    // std::cout << " left " << b.left << " width " << b.width << " top " << b.top << " height " << b.height << std::endl;
    return b;
}

static void addBBoxProposalYolor(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYolor(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

/* SCRFD implementations */
static NvDsInferParseObjectInfo convertBBoxSCRFD(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    const int margin = 0; // Add margin for more robust alignment 
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 - margin;
    float y1 = by1 - margin;
    float x2 = bx2 + margin;
    float y2 = by2 + margin;

    b.left = x1;
    b.width = x2 - x1;
    b.top = y1;
    b.height = y2 - y1;
    // std::cout << " left " << b.left << " width " << b.width << " top " << b.top << " height " << b.height << std::endl;
    return b;
}

static void addBBoxProposalSCRFD(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxSCRFD(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo> decodeYolorTensor(
    const float* boxes, const float* scores, const float* classes,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;


    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = clamp(boxes[bbox_location], 0, netW - 1);
        float by1 = clamp(boxes[bbox_location + 1], 0, netH - 1);
        float bx2 = clamp(boxes[bbox_location + 2], 0, netW - 1);
        float by2 = clamp(boxes[bbox_location + 3], 0, netH - 1);
        float maxProb = scores[score_location];
        int maxIndex = (int) classes[score_location];

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYolor(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += 1;
    }

    return binfo;
}

static std::vector<NvDsInferParseObjectInfo> decodeSCRFDTensor(
    const float* boxes, const float* scores, const float* classes, const float* landmarks,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;


    uint bbox_location = 0;
    uint score_location = 0;
    uint lmk_location = 0;
    // const float half_margin = 22.0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];
        float maxProb = scores[score_location];
        int maxIndex = (int) classes[score_location];

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalSCRFD(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
            // std::cout << "SCRFD: " << bx1 << " " << by1 << " " << bx2 << " " << by2 << " " << " " << netW << " " << " " << netH << " " << std::endl;
            // std::cout << "LMK  : " << landmarks[lmk_location] << " " << landmarks[lmk_location+2] << " " << landmarks[lmk_location+4] << " " << landmarks[lmk_location+6] << " " << landmarks[lmk_location+8] << std::endl;
            // std::cout << "LMK  : " << landmarks[lmk_location+1] << " " << landmarks[lmk_location+3] << " " << landmarks[lmk_location+5] << " " << landmarks[lmk_location+7] << " " << landmarks[lmk_location+9] << std::endl;
        }

        bbox_location += 4;
        score_location += 1;
        lmk_location += 10;
    }

    return binfo;
}

extern "C" bool NvDsInferParseCustomYolor(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{

    std::vector<NvDsInferParseObjectInfo> objects;
    const NvDsInferLayerInfo &n_bboxes   = outputLayersInfo[0];
    const NvDsInferLayerInfo &boxes      = outputLayersInfo[1]; // (num_boxes, 4)
    const NvDsInferLayerInfo &scores     = outputLayersInfo[2]; // (num_boxes, )
    const NvDsInferLayerInfo &classes    = outputLayersInfo[3]; // (num_boxes, )
    

    int num_bboxes = *(const int*)(n_bboxes.buffer);


    assert(boxes.inferDims.numDims == 2);
    assert(scores.inferDims.numDims == 1);
    assert(classes.inferDims.numDims == 1);

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYolorTensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseCustomSCRFD(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{


    std::vector<NvDsInferParseObjectInfo> objects;
    const NvDsInferLayerInfo &n_bboxes   = outputLayersInfo[0];
    const NvDsInferLayerInfo &boxes      = outputLayersInfo[1]; // (num_boxes, 4)
    const NvDsInferLayerInfo &scores     = outputLayersInfo[2]; // (num_boxes, )
    const NvDsInferLayerInfo &classes    = outputLayersInfo[3]; // (num_boxes, )
    const NvDsInferLayerInfo &landmarks  = outputLayersInfo[4]; // (num_boxes, )

    int num_bboxes = *(const int*)(n_bboxes.buffer);
    // std::cout << "Got from plugin " << std::to_string(num_bboxes) << " faces" << std::endl;

    assert(boxes.inferDims.numDims == 2);
    assert(scores.inferDims.numDims == 1);
    assert(classes.inferDims.numDims == 1);

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeSCRFDTensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), (const float*)(classes.buffer), (const float*)(landmarks.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseCustomFaceEmbedding(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
{


    float* embeddings         = (float*) outputLayersInfo[0].buffer; // (num_boxes, 512)

    const int number_features = 512;
    const int n_truncate = 43; // 43 truncates, each truncate have 12 value: 12, 12, 12, ... 12, 8
    const int truncate_size = 12;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    // Add to metadata
    int idx = 0;
    for (int k = 0; k < n_truncate; k++){
        std::string truncate_label = "";
        for (int kk = 0; kk < truncate_size; kk++){
            if (kk != 0) truncate_label += ",";
            truncate_label += std::to_string(embeddings[idx]);
            idx += 1;
            if (idx == number_features) {
                break;
            }
        }
        NvDsInferAttribute face_embed_truncate;
        face_embed_truncate.attributeIndex = k;
        face_embed_truncate.attributeValue = 1;
        face_embed_truncate.attributeConfidence = 1.0;
        face_embed_truncate.attributeLabel = strdup(truncate_label.c_str());
        // std::cout << "push back " << k << std::endl;
        attrList.push_back(face_embed_truncate);
    }
    return true;
}


/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolor);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSCRFD);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFaceEmbedding);