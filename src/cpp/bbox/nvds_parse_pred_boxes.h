#ifndef DETR_DEEPSTREAM_SDK_SRC_CPP_BBOX_NVDS_PARSE_PRED_BOXES_H
#define DETR_DEEPSTREAM_SDK_SRC_CPP_BBOX_NVDS_PARSE_PRED_BOXES_H

#include <vector>

#include <nvdsinfer_custom_impl.h>

extern "C"
auto NvDsInferParseCustomDETRResNet101(
    const std::vector<NvDsInferLayerInfo>& output_layers_info,
    const NvDsInferNetworkInfo& network_info,
    const NvDsInferParseDetectionParams& detection_params,
    std::vector<NvDsInferParseObjectInfo>& object_list
) -> bool;

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDETRResNet101);

#endif // DETR_DEEPSTREAM_SDK_SRC_CPP_BBOX_NVDS_PARSE_PRED_BOXES_H
