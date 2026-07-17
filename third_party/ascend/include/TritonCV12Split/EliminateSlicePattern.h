//===- EliminateSlicePattern.h ------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#ifndef TRITON_CV12_SPLIT_ELIMINATE_SLICE_PATTERN_H
#define TRITON_CV12_SPLIT_ELIMINATE_SLICE_PATTERN_H

namespace mlir {
class RewritePatternSet;
namespace triton::detail {
using namespace mlir;
void populateEliminateSlicePattern(RewritePatternSet &patterns);
void populateEliminateSliceFallbackPattern(RewritePatternSet &patterns);
} // namespace triton::detail
} // namespace mlir

#endif
