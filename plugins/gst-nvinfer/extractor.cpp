#include "extractor.h"


namespace extractor_namespace {
class Extractor::Impl {
public:
	void facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res, bool is_fullframe);
	// cv::Mat AlignPlate(const cv::Mat& dst, const cv::Mat& src);

private:
    void parse_nms_output(std::vector<FaceInfo>& res, int num_detections, float *nmsed_boxes, float *nmsed_scores, float *nmsed_lmks, int width, int height);
};

Extractor::Extractor() {
    impl_ = new Impl();
}

Extractor::~Extractor() {
    if (impl_) {
        delete impl_;
    }
}

void Extractor::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res, bool is_fullframe) {
    return impl_->facelmks(l_user, res, is_fullframe);
}

void Extractor::Impl::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res, bool is_fullframe) {
    static guint use_host_mem = 1; // Process on GPU or CPU
    for (;l_user != NULL; l_user = l_user->next) { 
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
            continue; 
        }
        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        for (int i = 0; i < 5; i++){
            NvDsInferLayerInfo *info = &meta->output_layers_info[i];
            info->buffer = meta->out_buf_ptrs_host[i];
            if (use_host_mem && meta->out_buf_ptrs_dev[i]) {
                // get all data from gpu to cpu to access buffer
                cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                    info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
        }
        std::vector < NvDsInferLayerInfo > outputLayersInfo (meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);

        if(outputLayersInfo.size() != 5)
            {
                std::cerr << "Mismatch in the number of output buffers."
                        << "Expected 5 output buffers, detected in the network :"
                        << outputLayersInfo.size() << std::endl;
            }
        int num_detections    = *(int *)(outputLayersInfo[0].buffer);
        float *nmsed_boxes    = (float*)(outputLayersInfo[1].buffer);
        float *nmsed_scores   = (float*)(outputLayersInfo[2].buffer);
        float *nmsed_lmks     = (float*)(outputLayersInfo[4].buffer);

        if (!is_fullframe){
            parse_nms_output(res, num_detections, nmsed_boxes, nmsed_scores, nmsed_lmks, FACENET_WIDTH, FACENET_HEIGHT);
        }
        else{
            parse_nms_output(res, num_detections, nmsed_boxes, nmsed_scores, nmsed_lmks, FACENET_FF_WIDTH, FACENET_FF_HEIGHT);
        }
    }  
}

void Extractor::Impl::parse_nms_output(std::vector<FaceInfo>& res,
                                    int num_detections,
                                    float *nmsed_boxes,
                                    float *nmsed_scores,
                                    float *nmsed_lmks,
                                    int width,
                                    int height
                                    ) {
    
    int bbox_location = 0;
    int score_location = 0;
    int lmk_location = 0;
    const int margin = 0;
    for (int i = 0; i < num_detections; i++){  
        FaceInfo det;
        // memcpy(&det, &nmsed_boxes[bbox_location], 4 * sizeof(float));
        float bx1 = nmsed_boxes[bbox_location];
        float by1 = nmsed_boxes[bbox_location + 1];
        float bx2 = nmsed_boxes[bbox_location + 2];
        float by2 = nmsed_boxes[bbox_location + 3];

        float lx1 = nmsed_lmks[lmk_location];
        float ly1 = nmsed_lmks[lmk_location + 1];
        float lx2 = nmsed_lmks[lmk_location + 2];
        float ly2 = nmsed_lmks[lmk_location + 3];
        float lx3 = nmsed_lmks[lmk_location + 4];
        float ly3 = nmsed_lmks[lmk_location + 5];
        float lx4 = nmsed_lmks[lmk_location + 6];
        float ly4 = nmsed_lmks[lmk_location + 7];
        float lx5 = nmsed_lmks[lmk_location + 8];
        float ly5 = nmsed_lmks[lmk_location + 9];
        det.bbox[0] = bx1 - margin;
        det.bbox[1] = by1 - margin;
        det.bbox[2] = bx2 + margin;
        det.bbox[3] = by2 + margin;
        det.lmk[0] = lx1 + margin;
        det.lmk[1] = ly1 + margin;
        det.lmk[2] = lx2 + margin;
        det.lmk[3] = ly2 + margin;
        det.lmk[4] = lx3 + margin;
        det.lmk[5] = ly3 + margin;
        det.lmk[6] = lx4 + margin;
        det.lmk[7] = ly4 + margin;
        det.lmk[8] = lx5 + margin;
        det.lmk[9] = ly5 + margin;
        det.score   = nmsed_scores[score_location];
        // det
        res.push_back(det);
        bbox_location  += 4;
        score_location += 1;
        lmk_location   += 10;
    }
}
}