//===- BubbleUpPattern.cpp --------------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#include "TritonCV12Split/BubbleUpPattern.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bubble-up-extract-slice-pattern"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::detail {

static bool isElementwiseOp(Operation *op) {
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  if (isa<mlir::triton::FpToFpOp>(op))
    return true;
  return false;
}

static bool isMarkedBubbledSlice(tensor::ExtractSliceOp sliceOp) {
  return sliceOp->hasAttrOfType<UnitAttr>("to_be_bubbled_slice");
}

static void markBubbledSlice(PatternRewriter &rewriter,
                                        tensor::ExtractSliceOp op) {
  op->setAttr("to_be_bubbled_slice", UnitAttr::get(rewriter.getContext()));
}

static bool isMarkedEliminatedSlice(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("to_be_eliminated_slice");
}

static void markEliminatedSlice(PatternRewriter &rewriter, Operation *op) {
  op->setAttr("to_be_eliminated_slice", UnitAttr::get(rewriter.getContext()));
}

static bool isMarkedCVCommunication(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("cv_communication_slice");
}

static void markCVCommunication(PatternRewriter &rewriter,
                                 Operation *op) {
  op->setAttr("cv_communication_slice", UnitAttr::get(rewriter.getContext()));
}

static bool isMarkedVVCommunication(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("vv_communication");
}

static void markVVCommunication(PatternRewriter &rewriter, Operation *op) {
  op->setAttr("vv_communication", UnitAttr::get(rewriter.getContext()));
}

bool ElementwiseBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  if (!srcOp)
    return false;
  return isElementwiseOp(srcOp);
}

LogicalResult
ElementwiseBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                        PatternRewriter &rewriter) const {
  Operation *op = sliceOp.getSource().getDefiningOp();

  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto sliceType = dyn_cast<RankedTensorType>(sliceOp.getType());

  if (!resultType || !sliceType)
    return failure();

  Location loc = sliceOp.getLoc();
  rewriter.setInsertionPoint(op);

  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());

  for (Value operand : op->getOperands()) {
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType) {
      newOperands.push_back(operand);
      continue;
    }

    auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, operand, sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    markBubbledSlice(rewriter, newSlice);
    newOperands.push_back(newSlice);
  }

  OperationState state(loc, op->getName());
  state.addOperands(newOperands);
  state.addTypes(sliceOp.getType());
  for (auto attr : op->getAttrs())
    state.addAttribute(attr.getName(), attr.getValue());

  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(sliceOp, newOp->getResults());
  return success();
}

bool TritonBroadcastBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::BroadcastOp>(srcOp);
}

LogicalResult
TritonBroadcastBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                         PatternRewriter &rewriter) const {
  auto broadcastOp = cast<mlir::triton::BroadcastOp>(
      sliceOp.getSource().getDefiningOp());

  auto srcType = dyn_cast<RankedTensorType>(broadcastOp.getSrc().getType());
  auto dstType = dyn_cast<RankedTensorType>(broadcastOp.getType());
  if (!srcType || !dstType)
    return failure();

  SmallVector<OpFoldResult> inputOffsets, inputSizes, inputStrides;
  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto sliceSizes = sliceOp.getMixedSizes();
  auto sliceStrides = sliceOp.getMixedStrides();

  for (unsigned i = 0; i < dstType.getRank(); ++i) {
    if (srcType.getDimSize(i) == 1) {
      inputOffsets.push_back(rewriter.getIndexAttr(0));
      inputSizes.push_back(rewriter.getIndexAttr(1));
      inputStrides.push_back(rewriter.getIndexAttr(1));
    } else {
      inputOffsets.push_back(sliceOffsets[i]);
      inputSizes.push_back(sliceSizes[i]);
      inputStrides.push_back(sliceStrides[i]);
    }
  }

  Location loc = broadcastOp.getLoc();
  rewriter.setInsertionPoint(broadcastOp);

  auto newSlicedSrc = rewriter.create<tensor::ExtractSliceOp>(
      loc, broadcastOp.getSrc(), inputOffsets, inputSizes, inputStrides);
  markBubbledSlice(rewriter, newSlicedSrc);

  auto newBroadcastOp = rewriter.create<mlir::triton::BroadcastOp>(
      loc, sliceOp.getType(), newSlicedSrc.getResult());

  rewriter.replaceOp(sliceOp, newBroadcastOp.getResult());
  return success();
}

static bool sliceParamsMatch(tensor::ExtractSliceOp extractOp,
                             tensor::InsertSliceOp insertOp) {
  auto extractOffsets = extractOp.getMixedOffsets();
  auto extractSizes = extractOp.getMixedSizes();
  auto extractStrides = extractOp.getMixedStrides();

  auto insertOffsets = insertOp.getMixedOffsets();
  auto insertSizes = insertOp.getMixedSizes();
  auto insertStrides = insertOp.getMixedStrides();

  if (extractOffsets.size() != insertOffsets.size())
    return false;
  if (extractSizes.size() != insertSizes.size())
    return false;
  if (extractStrides.size() != insertStrides.size())
    return false;

  for (size_t i = 0; i < extractOffsets.size(); ++i) {
    if (extractOffsets[i] != insertOffsets[i])
      return false;
    if (extractSizes[i] != insertSizes[i])
      return false;
    if (extractStrides[i] != insertStrides[i])
      return false;
  }
  return true;
}

// TODO: support while loop case
bool LoopInBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto *sourceOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<scf::ForOp>(sourceOp);
}

// TODO: support while loop case
LogicalResult
LoopInBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                    PatternRewriter &rewriter) const {
  auto forOp = dyn_cast<scf::ForOp>(sliceOp.getSource().getDefiningOp());
  if (!forOp)
    return failure();

  auto yieldIndex = cast<OpResult>(sliceOp.getSource()).getResultNumber();
  Operation *yieldOp = forOp.getRegion().getBlocks().rbegin()->getTerminator();
  if (!isa<scf::YieldOp>(yieldOp))
    return failure();

  BlockArgument regionIterArg = forOp.getRegionIterArg(yieldIndex);
  OpOperand &forOpInit = forOp.getInitsMutable()[yieldIndex];

  auto initInsertOp = forOpInit.get().getDefiningOp<tensor::InsertSliceOp>();

  if (initInsertOp && isMarkedEliminatedSlice(initInsertOp)) {
    tensor::ExtractSliceOp iterExtractOp = nullptr;
    for (Operation *user : regionIterArg.getUsers()) {
      auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractOp && isMarkedEliminatedSlice(extractOp) &&
          sliceParamsMatch(extractOp, initInsertOp)) {
        iterExtractOp = extractOp;
        break;
      }
    }
    if (!iterExtractOp)
      return failure();

    auto slicedType = cast<RankedTensorType>(sliceOp.getType());
    Value initSource = initInsertOp.getSource();

    rewriter.modifyOpInPlace(forOp, [&]() {
      forOpInit.set(initSource);
    });

    rewriter.replaceAllUsesWith(iterExtractOp.getResult(), regionIterArg);
    regionIterArg.setType(slicedType);

    rewriter.setInsertionPoint(yieldOp);
    auto innerExtract = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), slicedType, yieldOp->getOperand(yieldIndex),
        sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    markBubbledSlice(rewriter, innerExtract);

    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp->setOperand(yieldIndex, innerExtract.getResult());
    });

    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp.getResult(yieldIndex).setType(slicedType);
    });

    rewriter.replaceAllUsesWith(sliceOp.getResult(), forOp.getResult(yieldIndex));

    rewriter.eraseOp(initInsertOp);
    rewriter.eraseOp(iterExtractOp);
    rewriter.eraseOp(sliceOp);

    return success();
  }

  Value valueToYield = forOp.getYieldedValues()[yieldIndex];

  rewriter.setInsertionPoint(yieldOp);

  auto innerExtract = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), sliceOp.getType(), valueToYield,
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());
  markBubbledSlice(rewriter, innerExtract);

  auto fullType = cast<RankedTensorType>(valueToYield.getType());
  auto empty =
      rewriter.create<tensor::EmptyOp>(sliceOp.getLoc(), fullType.getShape(),
                                        fullType.getElementType());
  auto insertBack = rewriter.create<tensor::InsertSliceOp>(
      sliceOp.getLoc(), innerExtract.getResult(), empty,
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());
  markEliminatedSlice(rewriter, insertBack);

  rewriter.modifyOpInPlace(yieldOp, [&]() {
    yieldOp->getOpOperand(yieldIndex).assign(insertBack.getResult());
  });

  if (sliceOp->hasAttr("to_be_bubbled_slice"))
    sliceOp->removeAttr("to_be_bubbled_slice");
  markEliminatedSlice(rewriter, sliceOp);

  return success();
}

// TODO: support while loop case
bool LoopOutBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  auto iterArg = dyn_cast<BlockArgument>(sliceOp.getSource());
  if (!iterArg)
    return false;

  auto forOp = dyn_cast<scf::ForOp>(iterArg.getOwner()->getParent()->getParentOp());
  if (!forOp)
    return false;

  auto yieldIndex = iterArg.getArgNumber() - forOp.getNumInductionVars();
  if (yieldIndex < 0 || yieldIndex >= static_cast<int>(forOp.getNumResults()))
    return false;

  if (!isMarkedBubbledSlice(sliceOp))
    return false;

  return true;
}

// TODO: support while loop case
LogicalResult
LoopOutBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                     PatternRewriter &rewriter) const {
  auto iterArg = cast<BlockArgument>(sliceOp.getSource());
  auto forOp = cast<scf::ForOp>(iterArg.getOwner()->getParent()->getParentOp());
  auto yieldIndex = iterArg.getArgNumber() - forOp.getNumInductionVars();

  Operation *yieldOp = forOp.getRegion().getBlocks().rbegin()->getTerminator();
  Value yieldValue = yieldOp->getOperand(yieldIndex);

  auto yieldInsertOp = yieldValue.getDefiningOp<tensor::InsertSliceOp>();

  if (yieldInsertOp && isMarkedEliminatedSlice(yieldInsertOp) &&
      sliceParamsMatch(sliceOp, yieldInsertOp)) {
    Value forResult = forOp.getResult(yieldIndex);
    tensor::ExtractSliceOp outerExtractOp = nullptr;
    for (Operation *user : forResult.getUsers()) {
      auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractOp && isMarkedEliminatedSlice(extractOp)) {
        outerExtractOp = extractOp;
        break;
      }
    }

    if (!outerExtractOp)
      return failure();

    OpOperand &forOpInit = forOp.getInitsMutable()[yieldIndex];
    rewriter.setInsertionPoint(forOp);
    auto initSlice = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), cast<RankedTensorType>(sliceOp.getType()),
        forOpInit.get(), sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        sliceOp.getMixedStrides());
    markBubbledSlice(rewriter, initSlice);
    forOpInit.set(initSlice.getResult());

    rewriter.modifyOpInPlace(forOp, [&]() {
      forOp.getResult(yieldIndex).setType(sliceOp.getType());
    });

    iterArg.setType(sliceOp.getType());

    rewriter.replaceAllUsesWith(sliceOp.getResult(), iterArg);

    Value compValue = yieldInsertOp.getSource();
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp->getOpOperand(yieldIndex).assign(compValue);
    });

    rewriter.eraseOp(sliceOp);
    rewriter.eraseOp(yieldInsertOp);
    rewriter.replaceOp(outerExtractOp, forOp->getResult(yieldIndex));

    return success();
  }

  OpOperand &forOpInit = forOp.getInitsMutable()[yieldIndex];
  auto slicedType = cast<RankedTensorType>(sliceOp.getType());
  auto fullType = cast<RankedTensorType>(forOpInit.get().getType());

  rewriter.setInsertionPoint(forOp);

  auto initExtract = rewriter.create<tensor::ExtractSliceOp>(
      sliceOp.getLoc(), slicedType, forOpInit.get(),
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());
  markBubbledSlice(rewriter, initExtract);

  auto empty = rewriter.create<tensor::EmptyOp>(
      sliceOp.getLoc(), fullType.getShape(), fullType.getElementType());

  auto initInsert = rewriter.create<tensor::InsertSliceOp>(
      sliceOp.getLoc(), initExtract.getResult(), empty,
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides());
  markEliminatedSlice(rewriter, initInsert);

  rewriter.modifyOpInPlace(forOp, [&]() {
    forOpInit.set(initInsert.getResult());
  });

  if (sliceOp->hasAttr("to_be_bubbled_slice"))
    sliceOp->removeAttr("to_be_bubbled_slice");
  markEliminatedSlice(rewriter, sliceOp);

  return success();
}

bool TritonDotBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::DotOp>(srcOp);
}

LogicalResult
TritonDotBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                       PatternRewriter &rewriter) const {
  sliceOp->removeAttr("to_be_bubbled_slice");
  markCVCommunication(rewriter, sliceOp);
  return success();
}

bool TritonReduceBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::ReduceOp>(srcOp);
}

LogicalResult
TritonReduceBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                      PatternRewriter &rewriter) const {
  auto reduceOp = cast<mlir::triton::ReduceOp>(sliceOp.getSource().getDefiningOp());

  auto srcs = reduceOp.getSrcs();
  if (srcs.size() != 1)
    return failure();

  auto inputType = cast<RankedTensorType>(srcs[0].getType());
  auto outputType = cast<RankedTensorType>(sliceOp.getType());
  auto rank = inputType.getRank();

  int32_t axis = reduceOp.getAxis();

  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto sliceSizes = sliceOp.getMixedSizes();
  auto sliceStrides = sliceOp.getMixedStrides();

  SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
  int out_idx = 0;
  for (int i = 0; i < rank; ++i) {
    if (i == axis) {
      newOffsets.push_back(rewriter.getIndexAttr(0));
      newSizes.push_back(rewriter.getIndexAttr(inputType.getDimSize(i)));
      newStrides.push_back(rewriter.getIndexAttr(1));
    } else {
      if (out_idx >= static_cast<int>(sliceOffsets.size()))
        return failure();
      newOffsets.push_back(sliceOffsets[out_idx]);
      newSizes.push_back(sliceSizes[out_idx]);
      newStrides.push_back(sliceStrides[out_idx]);
      out_idx++;
    }
  }

  Location loc = reduceOp.getLoc();
  rewriter.setInsertionPoint(reduceOp);

  auto newSlicedInput = rewriter.create<tensor::ExtractSliceOp>(
      loc, srcs[0], newOffsets, newSizes, newStrides);
  markBubbledSlice(rewriter, newSlicedInput);

  auto newReduceOp = rewriter.create<mlir::triton::ReduceOp>(
      loc, outputType, newSlicedInput.getResult(), axis);
  
  rewriter.cloneRegionBefore(reduceOp.getRegion(), newReduceOp.getRegion(),
                              newReduceOp.getRegion().begin());

  rewriter.replaceOp(sliceOp, newReduceOp.getResults());
  return success();
}

bool TritonExpandDimsBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::ExpandDimsOp>(srcOp);
}

LogicalResult
TritonExpandDimsBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                          PatternRewriter &rewriter) const {
  auto expandOp = cast<mlir::triton::ExpandDimsOp>(
      sliceOp.getSource().getDefiningOp());

  auto srcType = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
  auto dstType = dyn_cast<RankedTensorType>(expandOp.getType());
  if (!srcType || !dstType)
    return failure();

  int32_t axis = expandOp.getAxis();

  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto sliceSizes = sliceOp.getMixedSizes();
  auto sliceStrides = sliceOp.getMixedStrides();

  SmallVector<OpFoldResult> inputOffsets, inputSizes, inputStrides;
  int inputRank = srcType.getRank();
  int outputRank = dstType.getRank();

  for (int i = 0; i < outputRank; ++i) {
    if (i == axis) {
      continue;
    }
    int inputIdx = (i < axis) ? i : i - 1;
    if (inputIdx < inputRank) {
      inputOffsets.push_back(sliceOffsets[i]);
      inputSizes.push_back(sliceSizes[i]);
      inputStrides.push_back(sliceStrides[i]);
    }
  }

  Location loc = expandOp.getLoc();
  rewriter.setInsertionPoint(expandOp);

  auto newSlicedSrc = rewriter.create<tensor::ExtractSliceOp>(
      loc, expandOp.getSrc(), inputOffsets, inputSizes, inputStrides);
  markBubbledSlice(rewriter, newSlicedSrc);

  auto newExpandOp = rewriter.create<mlir::triton::ExpandDimsOp>(
      loc, sliceOp.getType(), newSlicedSrc.getResult(), axis);

  rewriter.replaceOp(sliceOp, newExpandOp.getResult());
  return success();
}

bool TritonSplatBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::SplatOp>(srcOp);
}

LogicalResult
TritonSplatBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                     PatternRewriter &rewriter) const {
  auto splatOp = cast<mlir::triton::SplatOp>(sliceOp.getSource().getDefiningOp());

  rewriter.setInsertionPoint(splatOp);
  auto newSplatOp = rewriter.create<mlir::triton::SplatOp>(
      splatOp.getLoc(), sliceOp.getType(), splatOp.getSrc());

  rewriter.replaceOp(sliceOp, newSplatOp.getResult());
  return success();
}

static Value createSlicedTensorPtr(tensor::ExtractSliceOp sliceOp,
                                   mlir::triton::MakeTensorPtrOp makeTensorPtrOp,
                                   mlir::triton::PointerType tensorPtrType,
                                   PatternRewriter &rewriter,
                                   Operation *insertionPoint) {
  Location loc = sliceOp.getLoc();
  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto slicedResultType = cast<RankedTensorType>(sliceOp.getType());
  auto tiledPtrType = mlir::triton::PointerType::get(
      slicedResultType, tensorPtrType.getAddressSpace());

  if (insertionPoint)
    rewriter.setInsertionPoint(insertionPoint);
  else
    rewriter.setInsertionPointAfter(makeTensorPtrOp);

  SmallVector<Value> newOffsets(makeTensorPtrOp.getOffsets());
  for (size_t i = 0; i < sliceOffsets.size(); ++i) {
    Value offsetValue =
        getValueOrCreateConstantIndexOp(rewriter, loc, sliceOffsets[i]);
    Value offsetI32 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), offsetValue);
    Value oldOffset = newOffsets[i];
    newOffsets[i] = rewriter.create<arith::AddIOp>(loc, oldOffset, offsetI32);
  }

  auto newMakeTensorPtrOp = rewriter.create<mlir::triton::MakeTensorPtrOp>(
      loc, tiledPtrType, makeTensorPtrOp.getBase(), makeTensorPtrOp.getShape(),
      makeTensorPtrOp.getStrides(), newOffsets, makeTensorPtrOp.getOrderAttr());
  return newMakeTensorPtrOp.getResult();
}

static Operation *createSlicedLoad(tensor::ExtractSliceOp sliceOp,
                                   mlir::triton::LoadOp loadOp, Value slicedPtr,
                                   PatternRewriter &rewriter) {
  Location loc = sliceOp.getLoc();
  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto sliceSizes = sliceOp.getMixedSizes();
  auto sliceStrides = sliceOp.getMixedStrides();

  rewriter.setInsertionPoint(loadOp);

  OperationState state(loc, loadOp->getName());
  state.addOperands(slicedPtr);

  if (loadOp.getMask()) {
    auto slicedMask = rewriter.create<tensor::ExtractSliceOp>(
        loc, loadOp.getMask(), sliceOffsets, sliceSizes, sliceStrides);
    markBubbledSlice(rewriter, slicedMask);
    state.addOperands(slicedMask.getResult());
  }

  if (loadOp.getOther()) {
    auto slicedOther = rewriter.create<tensor::ExtractSliceOp>(
        loc, loadOp.getOther(), sliceOffsets, sliceSizes, sliceStrides);
    markBubbledSlice(rewriter, slicedOther);
    state.addOperands(slicedOther.getResult());
  }

  state.addTypes(sliceOp.getType());
  for (auto attr : loadOp->getAttrs())
    state.addAttribute(attr.getName(), attr.getValue());

  return rewriter.create(state);
}

static bool isDefinedOutsideFor(Value value, scf::ForOp forOp) {
  auto defOp = value.getDefiningOp();
  if (!defOp)
    return true;
  return !forOp->isProperAncestor(defOp);
}

static LogicalResult
bubbleUpLoopCarriedTensorPtrLoad(tensor::ExtractSliceOp sliceOp,
                                 mlir::triton::LoadOp loadOp,
                                 mlir::triton::PointerType tensorPtrType,
                                 PatternRewriter &rewriter) {
  auto iterArg = dyn_cast<BlockArgument>(loadOp.getPtr());
  if (!iterArg)
    return failure();

  auto forOp = dyn_cast<scf::ForOp>(
      iterArg.getOwner()->getParent()->getParentOp());
  if (!forOp)
    return failure();

  int yieldIndex = iterArg.getArgNumber() - forOp.getNumInductionVars();
  if (yieldIndex < 0 || yieldIndex >= static_cast<int>(forOp.getNumResults()))
    return failure();

  OpOperand &forOpInit = forOp.getInitsMutable()[yieldIndex];
  auto makeTensorPtrOp =
      forOpInit.get().getDefiningOp<mlir::triton::MakeTensorPtrOp>();
  if (!makeTensorPtrOp)
    return failure();

  for (OpFoldResult offset : sliceOp.getMixedOffsets()) {
    if (auto value = dyn_cast<Value>(offset);
        value && !isDefinedOutsideFor(value, forOp))
      return failure();
  }

  Operation *yieldOp = forOp.getRegion().getBlocks().rbegin()->getTerminator();
  auto advanceOp =
      yieldOp->getOperand(yieldIndex).getDefiningOp<mlir::triton::AdvanceOp>();
  if (!advanceOp || advanceOp.getPtr() != iterArg)
    return failure();

  for (Operation *user : iterArg.getUsers()) {
    if (user != loadOp && user != advanceOp)
      return failure();
  }

  if (!forOp.getResult(yieldIndex).use_empty())
    return failure();

  Value slicedInitPtr =
      createSlicedTensorPtr(sliceOp, makeTensorPtrOp, tensorPtrType, rewriter,
                            forOp.getOperation());

  auto slicedPtrType = cast<mlir::triton::PointerType>(slicedInitPtr.getType());

  rewriter.modifyOpInPlace(forOp, [&]() { forOpInit.set(slicedInitPtr); });
  iterArg.setType(slicedPtrType);
  rewriter.modifyOpInPlace(advanceOp, [&]() {
    advanceOp.getResult().setType(slicedPtrType);
  });
  rewriter.modifyOpInPlace(forOp, [&]() {
    forOp.getResult(yieldIndex).setType(slicedPtrType);
  });

  Operation *newLoadOp = createSlicedLoad(sliceOp, loadOp, iterArg, rewriter);
  rewriter.replaceOp(sliceOp, newLoadOp->getResults());
  return success();
}

bool TritonLoadBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  if (!isa_and_nonnull<mlir::triton::LoadOp>(srcOp))
    return false;

  auto loadOp = cast<mlir::triton::LoadOp>(srcOp);
  
  auto ptrType = dyn_cast<RankedTensorType>(loadOp.getPtr().getType());
  if (ptrType && isa<mlir::triton::PointerType>(ptrType.getElementType()))
    return true;

  auto tensorPtrType = dyn_cast<mlir::triton::PointerType>(loadOp.getPtr().getType());
  if (tensorPtrType && isa<RankedTensorType>(tensorPtrType.getPointeeType()))
    return true;

  return false;
}

LogicalResult
TritonLoadBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                    PatternRewriter &rewriter) const {
  auto loadOp = cast<mlir::triton::LoadOp>(sliceOp.getSource().getDefiningOp());
  Location loc = sliceOp.getLoc();

  auto tensorPtrType = dyn_cast<mlir::triton::PointerType>(loadOp.getPtr().getType());
  if (tensorPtrType && isa<RankedTensorType>(tensorPtrType.getPointeeType())) {
    if (auto makeTensorPtrOp =
            loadOp.getPtr().getDefiningOp<mlir::triton::MakeTensorPtrOp>()) {
      Value slicedPtr = createSlicedTensorPtr(
          sliceOp, makeTensorPtrOp, tensorPtrType, rewriter, nullptr);
      Operation *newLoadOp =
          createSlicedLoad(sliceOp, loadOp, slicedPtr, rewriter);
      rewriter.replaceOp(sliceOp, newLoadOp->getResults());
      return success();
    }

    if (succeeded(bubbleUpLoopCarriedTensorPtrLoad(
            sliceOp, loadOp, tensorPtrType, rewriter)))
      return success();

    return failure();
  }

  auto ptrType = dyn_cast<RankedTensorType>(loadOp.getPtr().getType());
  if (!ptrType)
    return failure();

  rewriter.setInsertionPoint(loadOp);

  auto slicedPtr = rewriter.create<tensor::ExtractSliceOp>(
      loc, loadOp.getPtr(), sliceOp.getMixedOffsets(),
      sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  markBubbledSlice(rewriter, slicedPtr);

  Operation *newLoadOp =
      createSlicedLoad(sliceOp, loadOp, slicedPtr.getResult(), rewriter);

  rewriter.replaceOp(sliceOp, newLoadOp->getResults());
  return success();
}

bool TritonMakeRangeBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::MakeRangeOp>(srcOp);
}

LogicalResult
TritonMakeRangeBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                          PatternRewriter &rewriter) const {
  auto makeRangeOp = cast<mlir::triton::MakeRangeOp>(sliceOp.getSource().getDefiningOp());

  auto srcType = dyn_cast<RankedTensorType>(sliceOp.getSourceType());
  if (!srcType || srcType.getRank() != 1)
    return failure();

  auto offsets = sliceOp.getMixedOffsets();
  auto sizes = sliceOp.getMixedSizes();

  std::optional<int64_t> sizeOpt = getConstantIntValue(sizes[0]);
  if (!sizeOpt)
    return failure();

  int64_t size = *sizeOpt;
  int64_t oldStart = makeRangeOp.getStartAttr().getInt();

  Value offsetValue = getValueOrCreateConstantIndexOp(rewriter, sliceOp.getLoc(), offsets[0]);

  Location loc = makeRangeOp.getLoc();
  rewriter.setInsertionPoint(makeRangeOp);

  auto newResultType = RankedTensorType::get(size, srcType.getElementType());
  auto newMakeRangeOp = rewriter.create<mlir::triton::MakeRangeOp>(
      loc, newResultType,
      IntegerAttr::get(rewriter.getI32Type(), static_cast<int32_t>(oldStart)),
      IntegerAttr::get(rewriter.getI32Type(), static_cast<int32_t>(oldStart + size)));

  Value offsetI32 = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), offsetValue);

  auto newSplatOp = rewriter.create<mlir::triton::SplatOp>(
      loc, newResultType, offsetI32);

  auto newAddOp = rewriter.create<arith::AddIOp>(
      loc, newMakeRangeOp.getResult(), newSplatOp.getResult());

  rewriter.replaceOp(sliceOp, newAddOp.getResult());
  return success();
}

bool TritonTransBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::TransOp>(srcOp);
}

LogicalResult
TritonTransBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                     PatternRewriter &rewriter) const {
  auto transOp = cast<mlir::triton::TransOp>(sliceOp.getSource().getDefiningOp());

  auto srcType = dyn_cast<RankedTensorType>(transOp.getSrc().getType());
  auto dstType = dyn_cast<RankedTensorType>(transOp.getType());
  if (!srcType || !dstType)
    return failure();

  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto sliceSizes = sliceOp.getMixedSizes();
  auto sliceStrides = sliceOp.getMixedStrides();

  auto orderAttr = transOp.getOrderAttr();
  auto order = orderAttr.asArrayRef();
  auto rank = srcType.getRank();

  SmallVector<OpFoldResult> inputOffsets(rank);
  SmallVector<OpFoldResult> inputSizes(rank);
  SmallVector<OpFoldResult> inputStrides(rank);

  for (int64_t i = 0; i < rank; ++i) {
    inputOffsets[i] = rewriter.getIndexAttr(0);
    inputSizes[i] = rewriter.getIndexAttr(srcType.getDimSize(i));
    inputStrides[i] = rewriter.getIndexAttr(1);
  }

  for (int64_t i = 0; i < rank; ++i) {
    int64_t srcDim = order[i];
    inputOffsets[srcDim] = sliceOffsets[i];
    inputSizes[srcDim] = sliceSizes[i];
  }

  Location loc = transOp.getLoc();
  rewriter.setInsertionPoint(transOp);

  auto newSlicedSrc = rewriter.create<tensor::ExtractSliceOp>(
      loc, transOp.getSrc(), inputOffsets, inputSizes, inputStrides);
  markBubbledSlice(rewriter, newSlicedSrc);

  auto newTransOp = rewriter.create<mlir::triton::TransOp>(
      loc, sliceOp.getType(), newSlicedSrc.getResult(), transOp.getOrderAttr());

  rewriter.replaceOp(sliceOp, newTransOp.getResult());
  return success();
}

// TritonReshapeBubbleUpStrategy: Bubble up extract_slice through tt.reshape.
// Supported scenarios:
// 1. The extract_slice only slices along axis 0 of the reshape result
//    (other axes have offset=0 and full size)
// 2. The source tensor's axis 0 size must be even (for 1:2 splitting)
// The strategy computes the corresponding slice on the source tensor:
//   src_offset = dst_offset * src_dim0_size / dst_dim0_size
//   src_size = dst_size * src_dim0_size / dst_dim0_size
// Example: tensor<32x128xf32> reshape -> tensor<4096xf32>
//          slice {offset=2048, size=2048} on dst
//          maps to slice {offset=16, size=16} on src axis 0
//          resulting in tensor<16x128xf32> reshape -> tensor<2048xf32>
bool TritonReshapeBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<mlir::triton::ReshapeOp>(srcOp);
}

LogicalResult
TritonReshapeBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                       PatternRewriter &rewriter) const {
  auto reshapeOp =
      cast<mlir::triton::ReshapeOp>(sliceOp.getSource().getDefiningOp());

  auto srcType = dyn_cast<RankedTensorType>(reshapeOp.getSrc().getType());
  auto dstType = dyn_cast<RankedTensorType>(reshapeOp.getType());
  if (!srcType || !dstType)
    return failure();

  if (srcType.getRank() == 0 || dstType.getRank() == 0)
    return failure();

  auto sliceOffsets = sliceOp.getMixedOffsets();
  auto sliceSizes = sliceOp.getMixedSizes();
  auto sliceStrides = sliceOp.getMixedStrides();

  if (sliceOffsets.size() != dstType.getRank())
    return failure();

  for (size_t i = 1; i < sliceOffsets.size(); ++i) {
    std::optional<int64_t> offsetOpt = getConstantIntValue(sliceOffsets[i]);
    std::optional<int64_t> sizeOpt = getConstantIntValue(sliceSizes[i]);
    if (!offsetOpt || *offsetOpt != 0)
      return failure();
    if (!sizeOpt || *sizeOpt != dstType.getDimSize(i))
      return failure();
  }

  int64_t srcDim0Size = srcType.getDimSize(0);
  int64_t dstDim0Size = dstType.getDimSize(0);

  if (srcDim0Size % 2 != 0)
    return failure();

  if (ShapedType::isDynamic(srcDim0Size) || ShapedType::isDynamic(dstDim0Size))
    return failure();

  Location loc = reshapeOp.getLoc();
  rewriter.setInsertionPoint(reshapeOp);

  SmallVector<OpFoldResult> srcSliceOffsets, srcSliceSizes, srcSliceStrides;

  AffineExpr srcOffsetExpr = rewriter.getAffineSymbolExpr(0) * srcDim0Size;
  srcOffsetExpr = srcOffsetExpr.floorDiv(dstDim0Size);
  OpFoldResult srcOffsetOfr = affine::makeComposedFoldedAffineApply(
      rewriter, loc, srcOffsetExpr, {sliceOffsets[0]});

  std::optional<int64_t> sliceSizeOpt = getConstantIntValue(sliceSizes[0]);
  if (!sliceSizeOpt)
    return failure();

  int64_t srcSizeDim0 = *sliceSizeOpt * srcDim0Size / dstDim0Size;

  srcSliceOffsets.push_back(srcOffsetOfr);
  srcSliceSizes.push_back(rewriter.getIndexAttr(srcSizeDim0));
  srcSliceStrides.push_back(rewriter.getIndexAttr(1));

  for (int64_t i = 1; i < srcType.getRank(); ++i) {
    srcSliceOffsets.push_back(rewriter.getIndexAttr(0));
    srcSliceSizes.push_back(rewriter.getIndexAttr(srcType.getDimSize(i)));
    srcSliceStrides.push_back(rewriter.getIndexAttr(1));
  }

  SmallVector<int64_t> srcSlicedShape;
  srcSlicedShape.push_back(srcSizeDim0);
  for (int64_t i = 1; i < srcType.getRank(); ++i) {
    srcSlicedShape.push_back(srcType.getDimSize(i));
  }
  auto srcSlicedType = RankedTensorType::get(srcSlicedShape, srcType.getElementType());

  auto newSlicedSrc = rewriter.create<tensor::ExtractSliceOp>(
      loc, srcSlicedType, reshapeOp.getSrc(), srcSliceOffsets, srcSliceSizes,
      srcSliceStrides);
  markBubbledSlice(rewriter, newSlicedSrc);

  OperationState state(loc, rewriter.getStringAttr("tt.reshape"));
  state.addOperands(newSlicedSrc.getResult());
  state.addTypes(sliceOp.getType());
  if (reshapeOp.getAllowReorderAttr())
    state.addAttribute("allow_reorder", reshapeOp.getAllowReorderAttr());
  if (reshapeOp.getEfficientLayoutAttr())
    state.addAttribute("efficient_layout", reshapeOp.getEfficientLayoutAttr());
  for (auto attr : reshapeOp->getAttrs()) {
    if (attr.getName() != "allow_reorder" && attr.getName() != "efficient_layout")
      state.addAttribute(attr.getName(), attr.getValue());
  }
  Operation *newReshapeOp = rewriter.create(state);

  rewriter.replaceOp(sliceOp, newReshapeOp->getResults());
  return success();
}

bool InsertSliceBubbleUpStrategy::isSupportedOperation(
    tensor::ExtractSliceOp sliceOp) const {
  Operation *srcOp = sliceOp.getSource().getDefiningOp();
  return isa_and_nonnull<tensor::InsertSliceOp>(srcOp);
}

LogicalResult
InsertSliceBubbleUpStrategy::execute(tensor::ExtractSliceOp sliceOp,
                                     PatternRewriter &rewriter) const {
  auto insertSliceOp = cast<tensor::InsertSliceOp>(sliceOp.getSource().getDefiningOp());
  
  if (isMarkedVVCommunication(insertSliceOp)) {
    sliceOp->removeAttr("to_be_bubbled_slice");
    markVVCommunication(rewriter, sliceOp);
    return success();
  }
  
  return failure();
}

LogicalResult
BubbleUpPattern::matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                  PatternRewriter &rewriter) const {
  Value source = sliceOp.getSource();

  if (!sliceOp.hasUnitStride())
    return rewriter.notifyMatchFailure(sliceOp, "expected unit stride");

  if (!isMarkedBubbledSlice(sliceOp))
    return rewriter.notifyMatchFailure(
        sliceOp, "sliceOp needs to_be_bubbled_slice attr");

  int extractSliceCount = 0;
  bool allAllowedOperationUsage =
      llvm::all_of(source.getUsers(), [&extractSliceCount](Operation *user) {
        if (isa<tensor::ExtractSliceOp>(user)) {
          extractSliceCount++;
        }
        return isa<tensor::ExtractSliceOp>(user);
      });

  if (!aggressive && !allAllowedOperationUsage)
    return rewriter.notifyMatchFailure(
        sliceOp, "not all usages are extract slice");

  if (extractSliceCount != 1)
    return rewriter.notifyMatchFailure(
        sliceOp, "source has more than one extract slice usage.");

  auto blockArg = dyn_cast<BlockArgument>(source);
  if (aggressive && blockArg)
    return rewriter.notifyMatchFailure(
        sliceOp, "aggressive mode does not support block argument");

  for (const auto &strategy : bubbleUpStrategies) {
    if (aggressive && strategy->canHandleBlockArgument())
      continue;
    if (strategy->isSupportedOperation(sliceOp)) {
      LDBG("Picked strategy for sliceOp " << source);
      return strategy->execute(sliceOp, rewriter);
    }
  }

  return failure();
}

} // namespace mlir::triton::detail
