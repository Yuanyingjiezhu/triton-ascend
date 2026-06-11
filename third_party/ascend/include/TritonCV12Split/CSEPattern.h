//===- CSEPattern.h ----------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#ifndef TRITON_BUBBLE_UP_EXTRACT_SLICE_CSE_PATTERN_H
#define TRITON_BUBBLE_UP_EXTRACT_SLICE_CSE_PATTERN_H

namespace mlir {
class RewritePatternSet;
namespace triton::detail {
using namespace mlir;
void populateCSEPattern(RewritePatternSet &patterns);
} // namespace triton::detail
} // namespace mlir

#endif