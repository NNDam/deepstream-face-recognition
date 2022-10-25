#ifndef _DAMONZZZ_EXTRACTOR_H_
#define _DAMONZZZ_EXTRACTOR_H_

#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "opencv2/opencv.hpp"

#define CLIP(a, min, max) (MAX(MIN(a, max), min))

static constexpr int LOCATIONS = 4;
static constexpr int LMKS = 10;
#define FACENET_WIDTH 160
#define FACENET_HEIGHT 320
#define FACENET_FF_WIDTH 640
#define FACENET_FF_HEIGHT 640
#define PLATENET_WIDTH 224
#define PLATENET_HEIGHT 224
struct alignas(float) FaceInfo{
    float bbox[LOCATIONS];
    float score;
    float lmk[LMKS];
};

namespace extractor_namespace {
class Extractor {
public:
    Extractor();
    ~Extractor();

    void facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res, bool is_fullframe);
    // cv::Mat AlignPlate(const cv::Mat & src, const cv::Mat & dst);
    
private:
    class Impl;
    Impl* impl_;
};

}// namespace extractor_namespace

#endif // !_DAMONZZZ_EXTRACTOR_H_