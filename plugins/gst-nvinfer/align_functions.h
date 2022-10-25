#ifndef _DAMONZZZ_ALIGNER_H_
#define _DAMONZZZ_ALIGNER_H_

#include "opencv2/opencv.hpp"


namespace align_namespace {
class Aligner {
public:
    Aligner();
    ~Aligner();

    cv::Mat AlignFace(const cv::Mat & dst);
    
private:
    class Impl;
    Impl* impl_;
};

} // namespace align_namespace

#endif // !_DAMONZZZ_ALIGNER_H_