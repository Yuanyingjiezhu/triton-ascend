//===- BubbleUpExtractSlicePass.cpp - Bubble Up ExtractSlice Pass ============//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#include "TritonCV12Split/CSEPattern.h"
#include "TritonCV12Split/EliminateSlicePattern.h"
#include "TritonCV12Split/BubbleUpExtractSlicePass.h"
#include "TritonCV12Split/BubbleUpPattern.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_BUBBLEUPEXTRACTSLICE
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "bubble-up-extract-slice"

using namespace mlir;

namespace {

static bool isMarkedBubbledSlice(tensor::ExtractSliceOp sliceOp) {
  return sliceOp->hasAttrOfType<UnitAttr>("to_be_bubbled_slice");
}

static bool isMarkedEliminatedSlice(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("to_be_eliminated_slice");
}

static bool isMarkedCVCommunication(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("cv_communication_slice");
}

static void markShouldKeptSlice(tensor::ExtractSliceOp sliceOp) {
  sliceOp->setAttr("should_kept_slice",
                   UnitAttr::get(sliceOp->getContext()));
}

static bool isUsedForDot(Value value);

static bool isDotUse(OpOperand &use) {
  Operation *user = use.getOwner();
  if (isa<triton::DotOp>(user))
    return true;
  if (auto transOp = dyn_cast<triton::TransOp>(user))
    return isUsedForDot(transOp.getResult());
  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    BlockArgument regionIterArg = forOp.getTiedLoopRegionIterArg(&use);
    if (regionIterArg)
      return isUsedForDot(regionIterArg);
  }
  if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
    auto *parentOp = yieldOp->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      auto operandIdx = use.getOperandNumber();
      return isUsedForDot(forOp->getResult(operandIdx));
    }
  }
  return false;
}

static bool isUsedForDot(Value value) {
  if (value.use_empty())
    return false;

  for (OpOperand &use : value.getUses()) {
    if (!isDotUse(use))
      return false;
  }
  return true;
}

static bool isUsedForDotIgnoringTopLevelExtractSlice(Value value) {
  bool hasDotUse = false;
  for (OpOperand &use : value.getUses()) {
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(use.getOwner())) {
      if (isMarkedBubbledSlice(extractSliceOp))
        continue;
      return false;
    }

    hasDotUse = true;
    if (!isDotUse(use))
      return false;
  }
  return hasDotUse;
}

static bool shouldAggressivelyBubbleUp(tensor::ExtractSliceOp extractSliceOp) {
  Value source = extractSliceOp.getSource();
  Operation *sourceOp = source.getDefiningOp();
  if (sourceOp && sourceOp->hasAttr("MetaUse"))
    return true;

  if (auto loadOp = source.getDefiningOp<triton::LoadOp>())
    return isUsedForDotIgnoringTopLevelExtractSlice(loadOp.getResult());

  return false;
}

static bool filterRemainingMarkedSlicesForAggressiveBubbleUp(ModuleOp moduleOp) {
  bool hasAggressiveCandidate = false;
  moduleOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
    if (!isMarkedBubbledSlice(extractSliceOp))
      return WalkResult::advance();

    if (shouldAggressivelyBubbleUp(extractSliceOp)) {
      hasAggressiveCandidate = true;
      return WalkResult::advance();
    }

    extractSliceOp->removeAttr("to_be_bubbled_slice");
    markShouldKeptSlice(extractSliceOp);
    return WalkResult::advance();
  });
  return hasAggressiveCandidate;
}

static bool hasUnexpectedLoopBlockArgShouldKeptSlice(ModuleOp moduleOp) {
  bool hasUnexpectedSlice = false;
  moduleOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
    if (!extractSliceOp->hasAttrOfType<UnitAttr>("should_kept_slice"))
      return WalkResult::advance();

    auto blockArg = dyn_cast<BlockArgument>(extractSliceOp.getSource());
    if (!blockArg ||
        !isa<scf::ForOp, scf::WhileOp>(blockArg.getOwner()->getParentOp()))
      return WalkResult::advance();

    extractSliceOp.emitError()
        << "loop was not tiled as expected: remaining "
           "should_kept_slice has a block argument source";
    hasUnexpectedSlice = true;
    return WalkResult::interrupt();
  });
  return hasUnexpectedSlice;
}

static bool hasRemainingSliceMarkerAttrs(ModuleOp moduleOp) {
  bool hasRemainingMarker = false;
  moduleOp.walk([&](Operation *op) {
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
    if ((extractSliceOp && isMarkedBubbledSlice(extractSliceOp)) ||
        isMarkedEliminatedSlice(op)) {
      hasRemainingMarker = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasRemainingMarker;
}

static bool hasUnsupportedCVCommunicationColSplit(ModuleOp moduleOp) {
  bool hasUnsupportedColSplit = false;
  moduleOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
    if (!isMarkedCVCommunication(extractSliceOp))
      return WalkResult::advance();

    auto sourceType =
        dyn_cast<RankedTensorType>(extractSliceOp.getSource().getType());
    auto resultType = dyn_cast<RankedTensorType>(extractSliceOp.getType());
    if (!sourceType || !resultType || sourceType.getRank() <= 1 ||
        resultType.getRank() <= 1)
      return WalkResult::advance();

    int64_t sourceDim = sourceType.getDimSize(1);
    int64_t resultDim = resultType.getDimSize(1);
    if (ShapedType::isDynamic(sourceDim) ||
        ShapedType::isDynamic(resultDim) || sourceDim == resultDim)
      return WalkResult::advance();

    if (sourceDim % 32 == 0)
      return WalkResult::advance();

    extractSliceOp.emitError()
        << "fixpipe col_split mode does not support non-32-multiple "
           "original shape on the second dimension (axis 1), got "
        << sourceDim;
    hasUnsupportedColSplit = true;
    return WalkResult::interrupt();
  });
  return hasUnsupportedColSplit;
}

struct BubbleUpExtractSlicePass
    : public mlir::triton::impl::BubbleUpExtractSliceBase<BubbleUpExtractSlicePass> {
  void runOnOperation() override;

private:
  SmallVector<std::shared_ptr<mlir::triton::detail::BubbleUpStrategy>>
  createStrategies() const;
};

SmallVector<std::shared_ptr<mlir::triton::detail::BubbleUpStrategy>>
BubbleUpExtractSlicePass::createStrategies() const {
  SmallVector<std::shared_ptr<mlir::triton::detail::BubbleUpStrategy>> strategies;
  strategies.push_back(std::make_shared<mlir::triton::detail::ElementwiseBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonBroadcastBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::IfBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::LoopInBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::LoopOutBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonDotBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonReduceBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonExpandDimsBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonSplatBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonLoadBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonMakeRangeBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonTransBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::TritonReshapeBubbleUpStrategy>());
  strategies.push_back(std::make_shared<mlir::triton::detail::InsertSliceBubbleUpStrategy>());
  return strategies;
}

void BubbleUpExtractSlicePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = moduleOp.getContext();
  GreedyRewriteConfig config;
  config.maxIterations = 50;

  auto strategies = createStrategies();

  RewritePatternSet patterns(context);
  patterns.add<mlir::triton::detail::BubbleUpPattern>(context, strategies, false);
  mlir::triton::detail::populateCSEPattern(patterns);

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), config))) {
    return signalPassFailure();
  }

  bool hasAggressiveCandidate =
      filterRemainingMarkedSlicesForAggressiveBubbleUp(moduleOp);

  if (hasAggressiveCandidate) {
    for (auto &strategy : strategies)
      strategy->setAggressive(true);

    RewritePatternSet aggressivePatterns(context);
    aggressivePatterns.add<mlir::triton::detail::BubbleUpPattern>(
        context, strategies, true);
    mlir::triton::detail::populateCSEPattern(aggressivePatterns);

    if (failed(applyPatternsAndFoldGreedily(
            moduleOp, std::move(aggressivePatterns), config))) {
      return signalPassFailure();
    }
  }

  PassManager pm(context);
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(moduleOp))) {
    return signalPassFailure();
  }

  RewritePatternSet eliminateSlicePatterns(context);
  mlir::triton::detail::populateEliminateSlicePattern(eliminateSlicePatterns);

  if (failed(applyPatternsAndFoldGreedily(
          moduleOp, std::move(eliminateSlicePatterns), config))) {
    return signalPassFailure();
  }

  RewritePatternSet fallbackSlicePatterns(context);
  mlir::triton::detail::populateEliminateSliceFallbackPattern(
      fallbackSlicePatterns);

  if (failed(applyPatternsAndFoldGreedily(
          moduleOp, std::move(fallbackSlicePatterns), config))) {
    return signalPassFailure();
  }

  if (hasUnsupportedCVCommunicationColSplit(moduleOp))
    return signalPassFailure();

  if (hasUnexpectedLoopBlockArgShouldKeptSlice(moduleOp))
    return signalPassFailure();

  // to be enhanced for should_kept_slice
  if (hasRemainingSliceMarkerAttrs(moduleOp))
    return signalPassFailure();
}

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createBubbleUpExtractSlicePass() {
  return std::make_unique<BubbleUpExtractSlicePass>();
}

} // namespace triton
} // namespace mlir
