#include "bbox/nvds_parse_pred_boxes.h"

extern "C"
auto NvDsInferParseCustomDETRResNet101(
    const std::vector<NvDsInferLayerInfo>& output_layers_info,
    const NvDsInferNetworkInfo& network_info,
    const NvDsInferParseDetectionParams& detection_params,
    std::vector<NvDsInferParseObjectInfo>& object_list
) -> bool {
    const auto& out_logits_layer = output_layers_info[0];
    const auto& out_parse_boxes_layer = output_layers_info[1];

    

    return true;
}
