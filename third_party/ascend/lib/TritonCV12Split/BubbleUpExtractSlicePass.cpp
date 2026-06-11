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

static bool isUsedByAddPtrOrLoad(Value value) {
  for (Operation *user : value.getUsers()) {
    if (isa<triton::AddPtrOp, triton::LoadOp>(user))
      return true;

    for (auto res : user->getResults()) {
      if (isUsedByAddPtrOrLoad(res))
        return true;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto &initOperand : forOp.getInitsMutable()) {
        if (initOperand.get() != value)
          continue;
        BlockArgument regionIterArg =
            forOp.getTiedLoopRegionIterArg(&initOperand);
        if (regionIterArg && isUsedByAddPtrOrLoad(regionIterArg))
          return true;
      }
    }
  }

  return false;
}

static bool isDirectlyFromLoad(tensor::ExtractSliceOp extractSliceOp) {
  return extractSliceOp.getSource().getDefiningOp<triton::LoadOp>() != nullptr;
}

static bool isMarkedBubbledSlice(tensor::ExtractSliceOp sliceOp) {
  return sliceOp->hasAttrOfType<UnitAttr>("to_be_bubbled_slice");
}

static bool isMarkedEliminatedSlice(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("to_be_eliminated_slice");
}

static bool areAllRemainingMarkedSlicesReadyForAggressiveBubbleUp(
    ModuleOp moduleOp) {
  bool allReady = true;
  moduleOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
    if (!isMarkedBubbledSlice(extractSliceOp))
      return WalkResult::advance();

    if (isUsedByAddPtrOrLoad(extractSliceOp.getSource()) ||
        isDirectlyFromLoad(extractSliceOp))
      return WalkResult::advance();

    allReady = false;
    return WalkResult::interrupt();
  });
  return allReady;
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

  if (!areAllRemainingMarkedSlicesReadyForAggressiveBubbleUp(moduleOp))
    return signalPassFailure();

  for (auto &strategy : strategies)
    strategy->setAggressive(true);

  RewritePatternSet aggressivePatterns(context);
  aggressivePatterns.add<mlir::triton::detail::BubbleUpPattern>(context, strategies, true);
  mlir::triton::detail::populateCSEPattern(aggressivePatterns);

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(aggressivePatterns), config))) {
    return signalPassFailure();
  }

  PassManager pm(context);
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(moduleOp))) {
    return signalPassFailure();
  }

  RewritePatternSet eliminateSlicePatterns(context);
  mlir::triton::detail::populateEliminateSlicePattern(eliminateSlicePatterns);

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(eliminateSlicePatterns), config))) {
    return signalPassFailure();
  }

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
