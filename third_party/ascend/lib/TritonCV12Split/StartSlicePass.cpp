//===- StartSlicePass.cpp - Slice tt.store along analyzer-selected axis ====//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#include "TritonCV12Split/StartSlicePass.h"

#include "Utils/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "Start-slice"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_STARTSLICE
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using DimensionAnalyzer = triton::DimensionAnalyzer;
using DimensionAnalyzerOptions = triton::DimensionAnalyzerOptions;

namespace {

static bool comesFromDot(Value value)
{
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
        return false;

    if (isa<triton::DotOp>(defOp))
        return true;

    if (isa<arith::TruncFOp>(defOp)) {
        auto truncOp = cast<arith::TruncFOp>(defOp);
        return comesFromDot(truncOp.getIn());
    }

    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
        auto resultIdx = cast<OpResult>(value).getResultNumber();
        Operation *yieldOp = forOp.getRegion().getBlocks().rbegin()->getTerminator();
        if (isa<scf::YieldOp>(yieldOp)) {
            Value yieldValue = yieldOp->getOperand(resultIdx);
            return comesFromDot(yieldValue);
        }
    }

    return false;
}

static bool isUsedByAddPtrOrLoad(Value value)
{
    for (Operation *user : value.getUsers()) {
        if (isa<triton::AddPtrOp, triton::LoadOp>(user))
            return true;

        for (auto res : user->getResults()) {
            if (isUsedByAddPtrOrLoad(res))
                return true;
        }

        if (auto forOp = dyn_cast<scf::ForOp>(user)) {
            for (auto &initOperand : forOp.getInitsMutable()) {
                if (initOperand.get() == value) {
                    BlockArgument regionIterArg = forOp.getTiedLoopRegionIterArg(&initOperand);
                    if (regionIterArg && isUsedByAddPtrOrLoad(regionIterArg))
                        return true;
                }
            }
        }
    }

    return false;
}

static bool isDotUse(OpOperand &use);

static bool isUsedForDot(Value value)
{
    if (value.use_empty())
        return false;
    for (OpOperand &use : value.getUses()) {
        if (!isDotUse(use))
            return false;
    }
    return true;
}

static bool isDotUse(OpOperand &use)
{
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

static bool isCubeStoreUse(OpOperand &use);

static bool isUsedForCubeStore(Value value)
{
    if (value.use_empty())
        return false;
    for (OpOperand &use : value.getUses()) {
        if (!isCubeStoreUse(use))
            return false;
    }
    return true;
}

static bool isCubeStoreUse(OpOperand &use)
{
    Operation *user = use.getOwner();
    if (isa<triton::StoreOp>(user) || isa<triton::AtomicRMWOp>(user))
        return true;
    if (auto truncOp = dyn_cast<arith::TruncFOp>(user))
        return isUsedForCubeStore(truncOp.getResult());
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        BlockArgument regionIterArg = forOp.getTiedLoopRegionIterArg(&use);
        if (regionIterArg)
            return isUsedForCubeStore(regionIterArg);
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        auto *parentOp = yieldOp->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
            auto operandIdx = use.getOperandNumber();
            return isUsedForCubeStore(forOp->getResult(operandIdx));
        }
    }
    return false;
}

static void markTiledOp(Operation *op, PatternRewriter &rewriter)
{
    op->setAttr("tiled_op", UnitAttr::get(rewriter.getContext()));
}

static void markBubbledSlice(tensor::ExtractSliceOp op, PatternRewriter &rewriter)
{
    op->setAttr("to_be_bubbled_slice", UnitAttr::get(rewriter.getContext()));
}

static void markCVCommunication(Operation *op, PatternRewriter &rewriter)
{
    op->setAttr("cv_communication_slice", UnitAttr::get(rewriter.getContext()));
}

static void markVVCommunication(Operation *op, PatternRewriter &rewriter)
{
    op->setAttr("vv_communication", UnitAttr::get(rewriter.getContext()));
}

struct StoreSlicePattern : public OpRewritePattern<triton::StoreOp> {
    DimensionAnalyzer &analyzer;

    StoreSlicePattern(MLIRContext *ctx, DimensionAnalyzer &analyzer)
        : OpRewritePattern(ctx), analyzer(analyzer)
    {
    }

    LogicalResult matchAndRewrite(triton::StoreOp storeOp, PatternRewriter &rewriter) const final
    {
        if (storeOp->hasAttrOfType<UnitAttr>("tiled_op"))
            return rewriter.notifyMatchFailure(storeOp, "already tiled");

        Value ptrVal = storeOp.getPtr();
        Value valueVal = storeOp.getValue();

        if (comesFromDot(valueVal))
            return rewriter.notifyMatchFailure(storeOp, "value comes from dot");

        auto valueType = dyn_cast<RankedTensorType>(valueVal.getType());
        if (!valueType)
            return rewriter.notifyMatchFailure(storeOp, "value is not a ranked tensor");

        if (valueType.getRank() == 0)
            return rewriter.notifyMatchFailure(storeOp, "0-d tensor");

        int64_t tilingDim = analyzer.getTilingDim(valueVal);
        if (tilingDim == -1)
            return rewriter.notifyMatchFailure(storeOp, "no tiling dim for value");

        LDBG("StoreSlicePattern: tiling dim = " << tilingDim);

        int64_t origSize = valueType.getDimSize(tilingDim);
        if (origSize <= 1 || ShapedType::isDynamic(origSize))
            return rewriter.notifyMatchFailure(storeOp, "cannot slice axis");

        int64_t tiledSize = origSize / 2;

        auto funcOp = storeOp->getParentOfType<FunctionOpInterface>();
        if (!funcOp)
            return rewriter.notifyMatchFailure(storeOp, "no enclosing function");

        Block &funcBody = funcOp.getFunctionBody().front();

        Value subBlockIdIndex;
        for (auto &op : funcBody) {
            if (auto existingOp = dyn_cast<arith::IndexCastOp>(&op)) {
                if (existingOp.getResult().getType() == rewriter.getIndexType()) {
                    auto srcOp = existingOp.getIn().getDefiningOp();
                    if (srcOp && isa<hivm::GetSubBlockIdxOp>(srcOp)) {
                        subBlockIdIndex = existingOp.getResult();
                        break;
                    }
                }
            }
        }
        if (!subBlockIdIndex)
            return rewriter.notifyMatchFailure(storeOp, "no subBlockIdIndex");

        Location loc = storeOp.getLoc();

        rewriter.setInsertionPointAfterValue(subBlockIdIndex);

        AffineExpr mulExpr = rewriter.getAffineSymbolExpr(0) * rewriter.getAffineSymbolExpr(1);
        OpFoldResult offsetOfr = affine::makeComposedFoldedAffineApply(
            rewriter, loc, mulExpr, {subBlockIdIndex, rewriter.getIndexAttr(tiledSize)});

        Value offsetIndex = getValueOrCreateConstantIndexOp(rewriter, loc, offsetOfr);

        Value offsetI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), offsetIndex);

        SmallVector<OpFoldResult> sliceOffsets, sliceSizes, sliceStrides;
        for (int64_t i = 0; i < valueType.getRank(); ++i) {
            if (i == tilingDim) {
                sliceOffsets.push_back(offsetOfr);
                sliceSizes.push_back(rewriter.getIndexAttr(tiledSize));
            } else {
                sliceOffsets.push_back(rewriter.getIndexAttr(0));
                sliceSizes.push_back(rewriter.getIndexAttr(valueType.getDimSize(i)));
            }
            sliceStrides.push_back(rewriter.getIndexAttr(1));
        }

        rewriter.setInsertionPoint(storeOp);

        SmallVector<int64_t> tiledShape;
        for (int64_t i = 0; i < valueType.getRank(); ++i)
            tiledShape.push_back(i == tilingDim ? tiledSize : valueType.getDimSize(i));
        auto slicedResultType = RankedTensorType::get(tiledShape, valueType.getElementType());

        auto slicedValueOp = rewriter.create<tensor::ExtractSliceOp>(loc, slicedResultType, valueVal, sliceOffsets,
                                                                     sliceSizes, sliceStrides);
        markBubbledSlice(slicedValueOp, rewriter);
        Value slicedValue = slicedValueOp.getResult();

        auto ptrType = dyn_cast<triton::PointerType>(ptrVal.getType());
        bool isTensorPtr = ptrType && isa<RankedTensorType>(ptrType.getPointeeType());

        Value newPtr;
        if (isTensorPtr) {
            auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(ptrVal.getDefiningOp());
            if (!makeTensorPtrOp)
                return rewriter.notifyMatchFailure(storeOp, "ptr not from make_tensor_ptr");

            OpBuilder::InsertionGuard ptrGuard(rewriter);
            rewriter.setInsertionPointAfter(makeTensorPtrOp);

            auto tiledPtrType = triton::PointerType::get(slicedResultType, ptrType.getAddressSpace());

            Value newOffset = rewriter.create<arith::AddIOp>(loc, makeTensorPtrOp.getOffsets()[tilingDim], offsetI32);

            SmallVector<Value> newOffsets(makeTensorPtrOp.getOffsets());
            newOffsets[tilingDim] = newOffset;

            auto newMakeTensorPtrOp = rewriter.create<triton::MakeTensorPtrOp>(
                loc, tiledPtrType, makeTensorPtrOp.getBase(), makeTensorPtrOp.getShape(), makeTensorPtrOp.getStrides(),
                newOffsets, makeTensorPtrOp.getOrderAttr());
            markTiledOp(newMakeTensorPtrOp, rewriter);

            newPtr = newMakeTensorPtrOp.getResult();
        } else {
            auto ptrTensorType = dyn_cast<RankedTensorType>(ptrVal.getType());
            if (!ptrTensorType)
                return rewriter.notifyMatchFailure(storeOp, "ptr is not a tensor");

            auto slicedPtrType = RankedTensorType::get(slicedResultType.getShape(), ptrTensorType.getElementType());

            auto slicedPtrOp = rewriter.create<tensor::ExtractSliceOp>(loc, slicedPtrType, ptrVal, sliceOffsets,
                                                                       sliceSizes, sliceStrides);
            markBubbledSlice(slicedPtrOp, rewriter);

            newPtr = slicedPtrOp.getResult();
        }

        auto newStoreOp =
            rewriter.create<triton::StoreOp>(loc, newPtr, slicedValue, storeOp.getCache(), storeOp.getEvict());
        markTiledOp(newStoreOp, rewriter);

        rewriter.eraseOp(storeOp);
        return success();
    }
};

struct AtomicRmwSlicePattern : public OpRewritePattern<triton::AtomicRMWOp> {
    DimensionAnalyzer &analyzer;

    AtomicRmwSlicePattern(MLIRContext *ctx, DimensionAnalyzer &analyzer)
        : OpRewritePattern(ctx), analyzer(analyzer)
    {
    }

    LogicalResult matchAndRewrite(triton::AtomicRMWOp atomicRmwOp, PatternRewriter &rewriter) const final
    {
        if (atomicRmwOp->hasAttrOfType<UnitAttr>("tiled_op"))
            return rewriter.notifyMatchFailure(atomicRmwOp, "already tiled");

        Value ptrVal = atomicRmwOp.getPtr();
        Value valVal = atomicRmwOp.getVal();

        if (comesFromDot(valVal))
            return rewriter.notifyMatchFailure(atomicRmwOp, "value comes from dot");

        Value maskVal = atomicRmwOp.getMask();

        auto valType = dyn_cast<RankedTensorType>(valVal.getType());
        if (!valType)
            return rewriter.notifyMatchFailure(atomicRmwOp, "val is not a ranked tensor");

        if (valType.getRank() == 0)
            return rewriter.notifyMatchFailure(atomicRmwOp, "0-d tensor");

        int64_t tilingDim = analyzer.getTilingDim(valVal);
        if (tilingDim == -1)
            return rewriter.notifyMatchFailure(atomicRmwOp, "no tiling dim for val");

        LDBG("AtomicRmwSlicePattern: tiling dim = " << tilingDim);

        int64_t origSize = valType.getDimSize(tilingDim);
        if (origSize <= 1 || ShapedType::isDynamic(origSize))
            return rewriter.notifyMatchFailure(atomicRmwOp, "cannot slice axis");

        int64_t tiledSize = origSize / 2;

        auto funcOp = atomicRmwOp->getParentOfType<FunctionOpInterface>();
        if (!funcOp)
            return rewriter.notifyMatchFailure(atomicRmwOp, "no enclosing function");

        Block &funcBody = funcOp.getFunctionBody().front();

        Value subBlockIdIndex;
        for (auto &op : funcBody) {
            if (auto existingOp = dyn_cast<arith::IndexCastOp>(&op)) {
                if (existingOp.getResult().getType() == rewriter.getIndexType()) {
                    auto srcOp = existingOp.getIn().getDefiningOp();
                    if (srcOp && isa<hivm::GetSubBlockIdxOp>(srcOp)) {
                        subBlockIdIndex = existingOp.getResult();
                        break;
                    }
                }
            }
        }
        if (!subBlockIdIndex)
            return rewriter.notifyMatchFailure(atomicRmwOp, "no subBlockIdIndex");

        Location loc = atomicRmwOp.getLoc();

        rewriter.setInsertionPointAfterValue(subBlockIdIndex);

        AffineExpr mulExpr = rewriter.getAffineSymbolExpr(0) * rewriter.getAffineSymbolExpr(1);
        OpFoldResult offsetOfr = affine::makeComposedFoldedAffineApply(
            rewriter, loc, mulExpr, {subBlockIdIndex, rewriter.getIndexAttr(tiledSize)});

        Value offsetIndex = getValueOrCreateConstantIndexOp(rewriter, loc, offsetOfr);

        Value offsetI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), offsetIndex);

        SmallVector<OpFoldResult> sliceOffsets, sliceSizes, sliceStrides;
        for (int64_t i = 0; i < valType.getRank(); ++i) {
            if (i == tilingDim) {
                sliceOffsets.push_back(offsetOfr);
                sliceSizes.push_back(rewriter.getIndexAttr(tiledSize));
            } else {
                sliceOffsets.push_back(rewriter.getIndexAttr(0));
                sliceSizes.push_back(rewriter.getIndexAttr(valType.getDimSize(i)));
            }
            sliceStrides.push_back(rewriter.getIndexAttr(1));
        }

        rewriter.setInsertionPoint(atomicRmwOp);

        SmallVector<int64_t> tiledShape;
        for (int64_t i = 0; i < valType.getRank(); ++i)
            tiledShape.push_back(i == tilingDim ? tiledSize : valType.getDimSize(i));
        auto slicedResultType = RankedTensorType::get(tiledShape, valType.getElementType());

        auto slicedValOp = rewriter.create<tensor::ExtractSliceOp>(loc, slicedResultType, valVal, sliceOffsets,
                                                                   sliceSizes, sliceStrides);
        markBubbledSlice(slicedValOp, rewriter);
        Value slicedVal = slicedValOp.getResult();

        Value newPtr;
        auto ptrTensorType = dyn_cast<RankedTensorType>(ptrVal.getType());
        if (ptrTensorType) {
            auto slicedPtrType = RankedTensorType::get(slicedResultType.getShape(), ptrTensorType.getElementType());

            auto slicedPtrOp = rewriter.create<tensor::ExtractSliceOp>(loc, slicedPtrType, ptrVal, sliceOffsets,
                                                                       sliceSizes, sliceStrides);
            markBubbledSlice(slicedPtrOp, rewriter);

            newPtr = slicedPtrOp.getResult();
        } else {
            return rewriter.notifyMatchFailure(atomicRmwOp, "ptr is not a tensor of pointers");
        }

        Value newMask;
        if (maskVal) {
            auto maskType = dyn_cast<RankedTensorType>(maskVal.getType());
            if (!maskType)
                return rewriter.notifyMatchFailure(atomicRmwOp, "mask is not a ranked tensor");

            auto slicedMaskType = RankedTensorType::get(slicedResultType.getShape(), maskType.getElementType());

            auto slicedMaskOp = rewriter.create<tensor::ExtractSliceOp>(loc, slicedMaskType, maskVal, sliceOffsets,
                                                                        sliceSizes, sliceStrides);
            markBubbledSlice(slicedMaskOp, rewriter);

            newMask = slicedMaskOp.getResult();
        }

        auto newAtomicRmwOp = rewriter.create<triton::AtomicRMWOp>(
            loc, slicedResultType, atomicRmwOp.getAtomicRmwOpAttr(), newPtr, slicedVal, newMask,
            atomicRmwOp.getSemAttr(), atomicRmwOp.getScopeAttr());
        markTiledOp(newAtomicRmwOp, rewriter);

        atomicRmwOp.getResult().replaceAllUsesWith(newAtomicRmwOp.getResult());
        rewriter.eraseOp(atomicRmwOp);
        return success();
    }
};

struct ReduceSlicePattern : public OpRewritePattern<triton::ReduceOp> {
    DimensionAnalyzer &analyzer;

    ReduceSlicePattern(MLIRContext *ctx, DimensionAnalyzer &analyzer)
        : OpRewritePattern(ctx), analyzer(analyzer)
    {
    }

    LogicalResult matchAndRewrite(triton::ReduceOp reduceOp, PatternRewriter &rewriter) const final
    {
        if (reduceOp->hasAttrOfType<UnitAttr>("tiled_op"))
            return rewriter.notifyMatchFailure(reduceOp, "already tiled");

        auto srcs = reduceOp.getSrcs();
        if (srcs.size() != 1)
            return rewriter.notifyMatchFailure(reduceOp, "only single source reduce supported");

        Value src = srcs[0];
        auto srcType = dyn_cast<RankedTensorType>(src.getType());
        if (!srcType)
            return rewriter.notifyMatchFailure(reduceOp, "source is not a ranked tensor");

        int32_t axis = reduceOp.getAxis();

        int64_t tilingDim = analyzer.getTilingDim(src);
        if (tilingDim == -1)
            return rewriter.notifyMatchFailure(reduceOp, "no tiling dim for src");

        if (tilingDim != axis)
            return rewriter.notifyMatchFailure(reduceOp, "reduce axis does not match tiling dim");

        LDBG("ReduceSlicePattern: tiling dim = " << tilingDim << " (matches reduce axis " << axis << ")");

        int64_t origSize = srcType.getDimSize(axis);
        if (origSize <= 1 || ShapedType::isDynamic(origSize))
            return rewriter.notifyMatchFailure(reduceOp, "cannot slice axis");

        int64_t tiledSize = origSize / 2;

        auto funcOp = reduceOp->getParentOfType<FunctionOpInterface>();
        if (!funcOp)
            return rewriter.notifyMatchFailure(reduceOp, "no enclosing function");

        Block &funcBody = funcOp.getFunctionBody().front();

        Value subBlockIdIndex;
        for (auto &op : funcBody) {
            if (auto existingOp = dyn_cast<arith::IndexCastOp>(&op)) {
                if (existingOp.getResult().getType() == rewriter.getIndexType()) {
                    auto srcOp = existingOp.getIn().getDefiningOp();
                    if (srcOp && isa<hivm::GetSubBlockIdxOp>(srcOp)) {
                        subBlockIdIndex = existingOp.getResult();
                        break;
                    }
                }
            }
        }
        if (!subBlockIdIndex)
            return rewriter.notifyMatchFailure(reduceOp, "no subBlockIdIndex");

        Location loc = reduceOp.getLoc();
        Type elementType = srcType.getElementType();
        Type resultType = reduceOp.getResult()[0].getType();
        auto rankedResultType = dyn_cast<RankedTensorType>(resultType);

        rewriter.setInsertionPointAfterValue(subBlockIdIndex);

        AffineExpr mulExpr = rewriter.getAffineSymbolExpr(0) * rewriter.getAffineSymbolExpr(1);
        OpFoldResult offsetOfr = affine::makeComposedFoldedAffineApply(
            rewriter, loc, mulExpr, {subBlockIdIndex, rewriter.getIndexAttr(tiledSize)});

        SmallVector<OpFoldResult> sliceOffsets, sliceSizes, sliceStrides;
        for (int64_t i = 0; i < srcType.getRank(); ++i) {
            if (i == axis) {
                sliceOffsets.push_back(offsetOfr);
                sliceSizes.push_back(rewriter.getIndexAttr(tiledSize));
            } else {
                sliceOffsets.push_back(rewriter.getIndexAttr(0));
                sliceSizes.push_back(rewriter.getIndexAttr(srcType.getDimSize(i)));
            }
            sliceStrides.push_back(rewriter.getIndexAttr(1));
        }

        rewriter.setInsertionPoint(reduceOp);

        SmallVector<int64_t> slicedShape;
        for (int64_t i = 0; i < srcType.getRank(); ++i)
            slicedShape.push_back(i == axis ? tiledSize : srcType.getDimSize(i));
        auto slicedSrcType = RankedTensorType::get(slicedShape, elementType);
        auto slicedSrcOp =
            rewriter.create<tensor::ExtractSliceOp>(loc, slicedSrcType, src, sliceOffsets, sliceSizes, sliceStrides);
        markBubbledSlice(slicedSrcOp, rewriter);

        auto partialReduceOp = rewriter.create<triton::ReduceOp>(loc, resultType, slicedSrcOp.getResult(), axis);
        rewriter.cloneRegionBefore(reduceOp.getRegion(), partialReduceOp.getRegion(), partialReduceOp.getRegion().begin());
        markTiledOp(partialReduceOp, rewriter);

        Value insertOffset = getValueOrCreateConstantIndexOp(rewriter, loc, subBlockIdIndex);
        Value intermediateTensor;

        if (!rankedResultType) {
            if (!isa<FloatType, IntegerType>(resultType))
                return rewriter.notifyMatchFailure(reduceOp, "result is not a scalar numeric type");

            intermediateTensor = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{2}, elementType).getResult();
            auto insertOp = rewriter.create<tensor::InsertOp>(loc, partialReduceOp.getResult()[0],
                                                              intermediateTensor, ValueRange{insertOffset});
            markVVCommunication(insertOp, rewriter);
            intermediateTensor = insertOp.getResult();
        } else {
            SmallVector<int64_t> intermediateShape;
            intermediateShape.push_back(2);
            for (int64_t i = 0; i < srcType.getRank(); ++i) {
                if (i != axis)
                    intermediateShape.push_back(srcType.getDimSize(i));
            }
            intermediateTensor = rewriter.create<tensor::EmptyOp>(loc, intermediateShape, elementType).getResult();

            SmallVector<OpFoldResult> insertOffsets;
            insertOffsets.push_back(insertOffset);
            for (int64_t i = 0; i < srcType.getRank(); ++i) {
                if (i != axis)
                    insertOffsets.push_back(rewriter.getIndexAttr(0));
            }

            SmallVector<OpFoldResult> insertSizes, insertStrides;
            insertSizes.push_back(rewriter.getIndexAttr(1));
            insertStrides.push_back(rewriter.getIndexAttr(1));
            for (int64_t i = 0; i < rankedResultType.getRank(); ++i) {
                insertSizes.push_back(rewriter.getIndexAttr(rankedResultType.getDimSize(i)));
                insertStrides.push_back(rewriter.getIndexAttr(1));
            }

            auto insertOp = rewriter.create<tensor::InsertSliceOp>(loc, partialReduceOp.getResult()[0], intermediateTensor,
                                                                   insertOffsets, insertSizes, insertStrides);
            markVVCommunication(insertOp, rewriter);
            intermediateTensor = insertOp.getResult();
        }

        auto finalReduceOp = rewriter.create<triton::ReduceOp>(loc, resultType, intermediateTensor, 0);
        rewriter.cloneRegionBefore(reduceOp.getRegion(), finalReduceOp.getRegion(), finalReduceOp.getRegion().begin());
        markTiledOp(finalReduceOp, rewriter);

        rewriter.replaceOp(reduceOp, finalReduceOp.getResults());
        return success();
    }
};

struct BroadcastSlicePattern : public OpRewritePattern<triton::BroadcastOp> {
    DimensionAnalyzer &analyzer;

    BroadcastSlicePattern(MLIRContext *ctx, DimensionAnalyzer &analyzer)
        : OpRewritePattern(ctx), analyzer(analyzer)
    {
    }

    LogicalResult matchAndRewrite(triton::BroadcastOp broadcastOp, PatternRewriter &rewriter) const final
    {
        if (broadcastOp->hasAttrOfType<UnitAttr>("tiled_op"))
            return rewriter.notifyMatchFailure(broadcastOp, "already tiled");

        Value src = broadcastOp.getSrc();
        auto srcType = dyn_cast<RankedTensorType>(src.getType());
        auto dstType = dyn_cast<RankedTensorType>(broadcastOp.getType());
        if (!srcType || !dstType)
            return rewriter.notifyMatchFailure(broadcastOp, "not ranked tensor types");

        if (srcType.getRank() != dstType.getRank())
            return rewriter.notifyMatchFailure(broadcastOp, "rank mismatch");

        int64_t dstTilingDim = analyzer.getTilingDim(broadcastOp.getResult());
        if (dstTilingDim == -1)
            return rewriter.notifyMatchFailure(broadcastOp, "no tiling dim for broadcast result");

        bool isBroadcastAxis = srcType.getDimSize(dstTilingDim) == 1 && dstType.getDimSize(dstTilingDim) > 1;
        if (!isBroadcastAxis)
            return rewriter.notifyMatchFailure(broadcastOp, "tiling dim is not a broadcast axis");

        int64_t srcTilingDim = -1;
        int64_t srcDimFromAnalyzer = analyzer.getTilingDim(src);
        if (srcDimFromAnalyzer >= 0 && srcType.getDimSize(srcDimFromAnalyzer) > 1) {
            srcTilingDim = srcDimFromAnalyzer;
        } else {
            for (int64_t i = 0; i < srcType.getRank(); ++i) {
                if (srcType.getDimSize(i) > 1) {
                    srcTilingDim = i;
                    break;
                }
            }
        }

        if (srcTilingDim < 0)
            return rewriter.notifyMatchFailure(broadcastOp, "no sliceable source axis");

        LDBG("BroadcastSlicePattern: dst tiling dim = " << dstTilingDim
            << " (broadcast axis), src slicing dim = " << srcTilingDim);

        int64_t srcAxisSize = srcType.getDimSize(srcTilingDim);
        if (srcAxisSize <= 1 || ShapedType::isDynamic(srcAxisSize))
            return rewriter.notifyMatchFailure(broadcastOp, "cannot slice source axis");

        int64_t tiledSize = srcAxisSize / 2;

        auto funcOp = broadcastOp->getParentOfType<FunctionOpInterface>();
        if (!funcOp)
            return rewriter.notifyMatchFailure(broadcastOp, "no enclosing function");

        Block &funcBody = funcOp.getFunctionBody().front();

        Value subBlockIdIndex;
        for (auto &op : funcBody) {
            if (auto existingOp = dyn_cast<arith::IndexCastOp>(&op)) {
                if (existingOp.getResult().getType() == rewriter.getIndexType()) {
                    auto srcOp = existingOp.getIn().getDefiningOp();
                    if (srcOp && isa<hivm::GetSubBlockIdxOp>(srcOp)) {
                        subBlockIdIndex = existingOp.getResult();
                        break;
                    }
                }
            }
        }
        if (!subBlockIdIndex)
            return rewriter.notifyMatchFailure(broadcastOp, "no subBlockIdIndex");

        Location loc = broadcastOp.getLoc();

        rewriter.setInsertionPointAfterValue(subBlockIdIndex);

        AffineExpr mulExpr = rewriter.getAffineSymbolExpr(0) * rewriter.getAffineSymbolExpr(1);
        OpFoldResult offsetOfr = affine::makeComposedFoldedAffineApply(
            rewriter, loc, mulExpr, {subBlockIdIndex, rewriter.getIndexAttr(tiledSize)});

        SmallVector<OpFoldResult> sliceOffsets, sliceSizes, sliceStrides;
        for (int64_t i = 0; i < srcType.getRank(); ++i) {
            if (i == srcTilingDim) {
                sliceOffsets.push_back(offsetOfr);
                sliceSizes.push_back(rewriter.getIndexAttr(tiledSize));
            } else {
                sliceOffsets.push_back(rewriter.getIndexAttr(0));
                sliceSizes.push_back(rewriter.getIndexAttr(srcType.getDimSize(i)));
            }
            sliceStrides.push_back(rewriter.getIndexAttr(1));
        }

        rewriter.setInsertionPoint(broadcastOp);

        SmallVector<int64_t> slicedShape;
        for (int64_t i = 0; i < srcType.getRank(); ++i)
            slicedShape.push_back(i == srcTilingDim ? tiledSize : srcType.getDimSize(i));
        auto slicedSrcType = RankedTensorType::get(slicedShape, srcType.getElementType());

        auto slicedSrcOp =
            rewriter.create<tensor::ExtractSliceOp>(loc, slicedSrcType, src, sliceOffsets, sliceSizes, sliceStrides);
        markBubbledSlice(slicedSrcOp, rewriter);

        auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, srcType.getShape(), srcType.getElementType());

        auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
            loc, slicedSrcOp.getResult(), emptyTensor.getResult(), sliceOffsets, sliceSizes, sliceStrides);
        markVVCommunication(insertSliceOp, rewriter);

        auto newBroadcastOp = rewriter.create<triton::BroadcastOp>(loc, broadcastOp.getType(), insertSliceOp.getResult());
        markTiledOp(newBroadcastOp, rewriter);

        rewriter.replaceOp(broadcastOp, newBroadcastOp.getResult());
        return success();
    }
};

struct DotSlicePattern : public OpRewritePattern<triton::DotOp> {
    DimensionAnalyzer &analyzer;

    DotSlicePattern(MLIRContext *ctx, DimensionAnalyzer &analyzer)
        : OpRewritePattern(ctx), analyzer(analyzer)
    {
    }

    LogicalResult matchAndRewrite(triton::DotOp dotOp, PatternRewriter &rewriter) const final
    {
        if (dotOp->hasAttrOfType<UnitAttr>("tiled_op"))
            return rewriter.notifyMatchFailure(dotOp, "already tiled");

        auto funcOp = dotOp->getParentOfType<FunctionOpInterface>();
        if (!funcOp)
            return failure();

        Block &funcBody = funcOp.getFunctionBody().front();

        Value subBlockIdIndex;
        for (auto &op : funcBody) {
            if (auto existingOp = dyn_cast<arith::IndexCastOp>(&op)) {
                if (existingOp.getResult().getType() == rewriter.getIndexType()) {
                    auto srcOp = existingOp.getIn().getDefiningOp();
                    if (srcOp && isa<hivm::GetSubBlockIdxOp>(srcOp)) {
                        subBlockIdIndex = existingOp.getResult();
                        break;
                    }
                }
            }
        }
        if (!subBlockIdIndex)
            return rewriter.notifyMatchFailure(dotOp, "no subBlockIdIndex");

        Location loc = dotOp.getLoc();

        rewriter.setInsertionPointAfterValue(subBlockIdIndex);

        AffineExpr mulExpr = rewriter.getAffineSymbolExpr(0) * rewriter.getAffineSymbolExpr(1);

        SmallVector<Value> newOperands;
        for (size_t idx = 0; idx < dotOp->getNumOperands(); ++idx) {
            Value operand = dotOp->getOperand(idx);

            if (idx == 2) {
                newOperands.push_back(operand);
                continue;
            }

            auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
            if (!tensorType) {
                newOperands.push_back(operand);
                continue;
            }

            Operation *defOp = operand.getDefiningOp();
            if (defOp && (isa<mlir::triton::LoadOp>(defOp) || isa<arith::ConstantOp>(defOp) || comesFromDot(operand)))
            {
                newOperands.push_back(operand);
                continue;
            }

            auto blockArg = dyn_cast<BlockArgument>(operand);
            if (blockArg) {
                auto *parentOp = blockArg.getOwner()->getParentOp();
                if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
                    auto argIdx = blockArg.getArgNumber();
                    auto numInductionVars = forOp.getNumInductionVars();
                    if (argIdx >= numInductionVars) {
                        auto initArgIdx = argIdx - numInductionVars;
                        Value initValue = forOp.getInits()[initArgIdx];
                        Operation *initDefOp = initValue.getDefiningOp();
                        if (initDefOp && (isa<mlir::triton::LoadOp>(initDefOp) || isa<arith::ConstantOp>(initDefOp) ||
                                          comesFromDot(initValue)))
                        {
                            newOperands.push_back(operand);
                            continue;
                        }
                    }
                } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
                    auto argIdx = blockArg.getArgNumber();
                    Value initValue = whileOp.getBeforeArguments()[argIdx];
                    Operation *initDefOp = initValue.getDefiningOp();
                    if (initDefOp && (isa<mlir::triton::LoadOp>(initDefOp) || isa<arith::ConstantOp>(initDefOp) ||
                                      comesFromDot(initValue)))
                    {
                        newOperands.push_back(operand);
                        continue;
                    }
                }
            }

            if (defOp && isa<mlir::triton::TransOp>(defOp)) {
                auto transOp = cast<mlir::triton::TransOp>(defOp);
                auto transSrc = transOp.getSrc();
                auto transSrcDefOp = transSrc.getDefiningOp();
                if (transSrcDefOp &&
                    (isa<mlir::triton::LoadOp>(transSrcDefOp) || isa<arith::ConstantOp>(transSrcDefOp) ||
                     comesFromDot(transSrc))) {
                    newOperands.push_back(operand);
                    continue;
                }

                auto transSrcType = dyn_cast<RankedTensorType>(transSrc.getType());
                if (!transSrcType)
                    return failure();

                int64_t transSrcTilingDim = analyzer.getTilingDim(transSrc);
                if (transSrcTilingDim == -1) {
                    newOperands.push_back(operand);
                    continue;
                }

                LDBG("DotSlicePattern: trans src tiling dim = " << transSrcTilingDim);

                int64_t transSrcOrigSize = transSrcType.getDimSize(transSrcTilingDim);
                if (transSrcOrigSize <= 1 || ShapedType::isDynamic(transSrcOrigSize))
                    return failure();

                int64_t transSrcTiledSize = transSrcOrigSize / 2;
                OpFoldResult offsetOfr = affine::makeComposedFoldedAffineApply(
                    rewriter, loc, mulExpr, {subBlockIdIndex, rewriter.getIndexAttr(transSrcTiledSize)});

                SmallVector<OpFoldResult> transSliceOffsets, transSliceSizes, transSliceStrides;
                for (int64_t i = 0; i < transSrcType.getRank(); ++i) {
                    if (i == transSrcTilingDim) {
                        transSliceOffsets.push_back(offsetOfr);
                        transSliceSizes.push_back(rewriter.getIndexAttr(transSrcTiledSize));
                    } else {
                        transSliceOffsets.push_back(rewriter.getIndexAttr(0));
                        transSliceSizes.push_back(rewriter.getIndexAttr(transSrcType.getDimSize(i)));
                    }
                    transSliceStrides.push_back(rewriter.getIndexAttr(1));
                }

                rewriter.setInsertionPoint(transOp);

                auto slicedSrc =
                    rewriter.create<tensor::ExtractSliceOp>(loc, transSrc, transSliceOffsets, transSliceSizes, transSliceStrides);
                markBubbledSlice(slicedSrc, rewriter);

                auto emptyTensor =
                    rewriter.create<tensor::EmptyOp>(loc, transSrcType.getShape(), transSrcType.getElementType());

                auto paddedSrc = rewriter.create<tensor::InsertSliceOp>(loc, slicedSrc.getResult(), emptyTensor,
                                                                        transSliceOffsets, transSliceSizes, transSliceStrides);
                markCVCommunication(paddedSrc, rewriter);

                auto newTransOp = rewriter.create<mlir::triton::TransOp>(loc, tensorType, paddedSrc.getResult(),
                                                                         transOp.getOrderAttr());

                newOperands.push_back(newTransOp.getResult());
                continue;
            }

            int64_t tilingDim = analyzer.getTilingDim(operand);
            if (tilingDim == -1) {
                newOperands.push_back(operand);
                continue;
            }

            LDBG("DotSlicePattern: operand " << idx << " tiling dim = " << tilingDim);

            int64_t origSize = tensorType.getDimSize(tilingDim);
            if (origSize <= 1 || ShapedType::isDynamic(origSize)) {
                newOperands.push_back(operand);
                continue;
            }

            int64_t tiledSize = origSize / 2;

            OpFoldResult offsetOfr = affine::makeComposedFoldedAffineApply(
                rewriter, loc, mulExpr, {subBlockIdIndex, rewriter.getIndexAttr(tiledSize)});

            SmallVector<OpFoldResult> sliceOffsets, sliceSizes, sliceStrides;
            for (int64_t i = 0; i < tensorType.getRank(); ++i) {
                if (i == tilingDim) {
                    sliceOffsets.push_back(offsetOfr);
                    sliceSizes.push_back(rewriter.getIndexAttr(tiledSize));
                } else {
                    sliceOffsets.push_back(rewriter.getIndexAttr(0));
                    sliceSizes.push_back(rewriter.getIndexAttr(tensorType.getDimSize(i)));
                }
                sliceStrides.push_back(rewriter.getIndexAttr(1));
            }

            rewriter.setInsertionPoint(dotOp);

            auto newSlice =
                rewriter.create<tensor::ExtractSliceOp>(loc, operand, sliceOffsets, sliceSizes, sliceStrides);
            markBubbledSlice(newSlice, rewriter);

            auto emptyTensor =
                rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(), tensorType.getElementType());

            auto newInsert = rewriter.create<tensor::InsertSliceOp>(loc, newSlice.getResult(), emptyTensor,
                                                                    sliceOffsets, sliceSizes, sliceStrides);
            markCVCommunication(newInsert, rewriter);

            newOperands.push_back(newInsert.getResult());
        }
        rewriter.setInsertionPoint(dotOp);
        auto newDot =
            rewriter.create<mlir::triton::DotOp>(loc, dotOp.getType(), newOperands[0], newOperands[1], newOperands[2]);
        markTiledOp(newDot, rewriter);
        dotOp.getResult().replaceAllUsesWith(newDot.getResult());
        rewriter.eraseOp(dotOp);
        return success();
    }
};

static LogicalResult tileAndSliceFunc(FunctionOpInterface funcOp)
{
    OpBuilder builder(funcOp->getContext());
    Block &funcBody = funcOp.getFunctionBody().front();

    Value subBlockId;
    bool hasSubBlockIdIndex = false;
    for (auto &op : funcBody) {
        if (auto idxOp = dyn_cast<hivm::GetSubBlockIdxOp>(&op)) {
            subBlockId = idxOp.getResult();
            continue;
        }
        if (auto castOp = dyn_cast<arith::IndexCastOp>(&op)) {
            if (castOp.getResult().getType() == builder.getIndexType()) {
                auto srcOp = castOp.getIn().getDefiningOp();
                if (srcOp && isa<hivm::GetSubBlockIdxOp>(srcOp)) {
                    hasSubBlockIdIndex = true;
                    break;
                }
            }
        }
    }

    if (!subBlockId) {
        builder.setInsertionPointToStart(&funcBody);
        subBlockId = builder.create<hivm::GetSubBlockIdxOp>(builder.getUnknownLoc(), builder.getI64Type());
    }
    if (!hasSubBlockIdIndex) {
        builder.setInsertionPointAfterValue(subBlockId);
        builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), builder.getIndexType(), subBlockId);
    }

    DimensionAnalyzerOptions options;
    options.isHeadOp = [](Operation *op) {
        if (isa<triton::LoadOp, triton::DotOp>(op))
            return true;
        if (isa<arith::ConstantOp, triton::MakeRangeOp, triton::SplatOp>(op))
            return !isUsedByAddPtrOrLoad(op->getResult(0));
        return false;
    };
    options.shouldSkipUse = [](Value value, OpOperand &use) -> bool {
        if (isDotUse(use))
            return true;
        Operation *defOp = value.getDefiningOp();
        if (defOp && isa<triton::DotOp>(defOp) && isCubeStoreUse(use))
            return true;
        Operation *user = use.getOwner();
        if (isa<triton::StoreOp>(user) || isa<triton::AtomicRMWOp>(user))
            return true;
        return false;
    };

    DimensionAnalyzer analyzer(funcOp, options);
    if (failed(analyzer.initialize()))
        return failure();

    bool isBroadcastAxisCase = analyzer.computeTilingDim();

    // // === DEBUG: print all broadcast op tiling dims ===
    // funcOp.walk([&](triton::BroadcastOp broadcastOp) {
    //     llvm::errs() << "[Start-slice-DEBUG] BroadcastOp: " << broadcastOp << "\n";
    //     Value src = broadcastOp.getSrc();
    //     auto srcType = dyn_cast<RankedTensorType>(src.getType());
    //     auto dstType = dyn_cast<RankedTensorType>(broadcastOp.getType());
    //     if (srcType && dstType) {
    //         llvm::errs() << "[Start-slice-DEBUG]   src type: " << srcType << ", dst type: " << dstType << "\n";
    //         int64_t srcTilingDim = analyzer.getTilingDim(src);
    //         int64_t dstTilingDim = analyzer.getTilingDim(broadcastOp.getResult());
    //         llvm::errs() << "[Start-slice-DEBUG]   src tiling dim = " << srcTilingDim
    //                      << ", dst tiling dim = " << dstTilingDim << "\n";
    //     } else {
    //         llvm::errs() << "[Start-slice-DEBUG]   src or dst is not a RankedTensorType\n";
    //     }
    // });
    // // === END DEBUG ===

    LLVM_DEBUG({
        DBGS() << "DimensionAnalyzer results:\n";
        funcOp.walk([&](Operation *op) {
            for (auto result : op->getResults()) {
                int64_t dim = analyzer.getTilingDim(result);
                if (dim >= 0)
                    LDBG("  tiling dim for " << result << " = axis " << dim);
            }
        });
        if (isBroadcastAxisCase)
            LDBG("  broadcast axis case detected");
    });

    RewritePatternSet patterns(funcOp->getContext());
    patterns.add<StoreSlicePattern, AtomicRmwSlicePattern, ReduceSlicePattern,
                 BroadcastSlicePattern, DotSlicePattern>(
        patterns.getContext(), analyzer);

    GreedyRewriteConfig config;
    config.maxIterations = 10;
    return applyPatternsAndFoldGreedily(funcOp, std::move(patterns), config);
}

struct StartSlicePass : public mlir::triton::impl::StartSliceBase<StartSlicePass> {
    void runOnOperation() override
    {
        ModuleOp moduleOp = getOperation();

        SmallVector<FunctionOpInterface> funcOps;
        moduleOp.walk([&](FunctionOpInterface funcOp) {
            funcOps.push_back(funcOp);
        });

        for (auto funcOp : funcOps) {
            if (failed(tileAndSliceFunc(funcOp))) {
                funcOp.emitError("StartSlice failed for function");
                return signalPassFailure();
            }
        }
    }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createStartSlicePass()
{
    return std::make_unique<StartSlicePass>();
}

} // namespace triton
} // namespace mlir
