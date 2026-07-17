//===- RewriteCommunicationSlicePass.cpp - Communication slice rewrite ===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "TritonCV12Split/RewriteCommunicationSlicePass.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_REWRITECOMMUNICATIONSLICE
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kCVCommunicationSliceAttr =
    "cv_communication_slice";
constexpr llvm::StringLiteral kVVCommunicationAttr = "vv_communication";

static MemRefType getMemRefType(RankedTensorType tensorType) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
}

static bool hasI1ElementType(RankedTensorType tensorType) {
  auto intType = dyn_cast<IntegerType>(tensorType.getElementType());
  return intType && intType.getWidth() == 1;
}

static RankedTensorType getI8StorageTensorType(RankedTensorType tensorType,
                                               MLIRContext *context) {
  if (!hasI1ElementType(tensorType))
    return tensorType;
  return RankedTensorType::get(tensorType.getShape(),
                               IntegerType::get(context, 8));
}

static Value castI1TensorToI8(OpBuilder &builder, Location loc, Value tensor) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType || !hasI1ElementType(tensorType))
    return tensor;

  auto storageType = getI8StorageTensorType(tensorType, builder.getContext());
  return builder.create<arith::ExtUIOp>(loc, storageType, tensor);
}

static Value castI8TensorToI1(OpBuilder &builder, Location loc, Value tensor,
                              RankedTensorType targetType) {
  if (!hasI1ElementType(targetType))
    return tensor;

  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType || tensorType == targetType)
    return tensor;

  return builder.create<arith::TruncIOp>(loc, targetType, tensor);
}

static SmallVector<Value> getDynamicTensorDims(OpBuilder &builder, Location loc,
                                               Value tensor,
                                               RankedTensorType tensorType) {
  SmallVector<Value> dynamicDims;
  for (auto [idx, dim] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(dim))
      dynamicDims.push_back(
          builder.create<tensor::DimOp>(loc, tensor, idx).getResult());
  }
  return dynamicDims;
}

static Value getOrCreateIndexValue(OpBuilder &builder, Location loc,
                                   OpFoldResult ofr) {
  if (auto value = dyn_cast<Value>(ofr))
    return value;
  auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt()).getResult();
}

static SmallVector<Value>
getDynamicDimsFromMixedSizes(OpBuilder &builder, Location loc,
                             RankedTensorType tensorType,
                             ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<Value> dynamicDims;
  for (auto [idx, dim] : llvm::enumerate(tensorType.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;
    dynamicDims.push_back(getOrCreateIndexValue(builder, loc, mixedSizes[idx]));
  }
  return dynamicDims;
}

static Value createAllocWorkspace(OpBuilder &builder, Location loc,
                                  MemRefType memrefType,
                                  ValueRange dynamicSizes = {}) {
  return builder
      .create<bishengir::memref_ext::AllocWorkspaceOp>(
          loc, memrefType, /*workspaceArg*/ Value(), dynamicSizes,
          /*offset*/ ValueRange{})
      .getMemref();
}

static Value expandSourceIfNeeded(OpBuilder &builder, Location loc,
                                 Value source, Value dest) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  auto destType = dyn_cast<MemRefType>(dest.getType());
  if (!sourceType || !destType)
    return source;

  unsigned srcRank = sourceType.getRank();
  unsigned dstRank = destType.getRank();
  if (srcRank >= dstRank)
    return source;

  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices firstGroup;
  for (unsigned i = 0; i <= dstRank - srcRank; i++)
    firstGroup.push_back(static_cast<int64_t>(i));
  reassociation.push_back(firstGroup);
  for (unsigned i = dstRank - srcRank + 1; i < dstRank; i++)
    reassociation.push_back(ReassociationIndices{static_cast<int64_t>(i)});

  SmallVector<OpFoldResult> outputShape;
  for (unsigned i = 0; i < dstRank; i++) {
    int64_t dim = destType.getDimSize(i);
    if (!ShapedType::isDynamic(dim))
      outputShape.push_back(builder.getIndexAttr(dim));
    else
      outputShape.push_back(
          builder.create<memref::DimOp>(loc, dest, i).getResult());
  }

  auto expandedType =
      RankedTensorType::get(destType.getShape(), sourceType.getElementType());
  return builder.create<tensor::ExpandShapeOp>(loc, expandedType, source,
                                                reassociation, outputShape);
}

static void materializeInDestination(OpBuilder &builder, Location loc,
                                     Value tensor, Value dest) {
  auto materialize =
      builder.create<bufferization::MaterializeInDestinationOp>(loc, tensor,
                                                                dest);
  materialize.setWritable(true);
}

static void createVVCommunicationSync(OpBuilder &builder, Location loc) {
  auto syncSubBlockMode =
      builder.getAttr<hivm::SyncBlockModeAttr>(
          hivm::SyncBlockMode::ALL_SUB_VECTOR);
  auto vectorPipeAttr =
      builder.getAttr<hivm::PipeAttr>(hivm::PIPE::PIPE_ALL);
  builder.create<hivm::SyncBlockOp>(loc, syncSubBlockMode, nullptr, Value{},
                                    hivm::PipeAttr{}, vectorPipeAttr);
}

static Value toWritableTensor(OpBuilder &builder, Location loc, Value memref,
                              RankedTensorType tensorType) {
  return builder.create<bufferization::ToTensorOp>(
      loc, tensorType, memref, true /* restrict */, true /* writable */);
}

static LogicalResult rewriteExtractSlice(OpBuilder &builder,
                                         tensor::ExtractSliceOp extractOp) {
  auto sourceType = dyn_cast<RankedTensorType>(extractOp.getSourceType());
  auto resultType = dyn_cast<RankedTensorType>(extractOp.getType());
  if (!sourceType || !resultType)
    return failure();

  Location loc = extractOp.getLoc();
  Operation *insertionAnchor = extractOp;
  DominanceInfo dominanceInfo;
  for (Operation *parent = extractOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp)
      continue;

    bool operandsDominateIf = true;
    for (Value operand : extractOp->getOperands()) {
      if (!dominanceInfo.dominates(operand, ifOp)) {
        operandsDominateIf = false;
        break;
      }
    }
    if (!operandsDominateIf)
      break;
    insertionAnchor = ifOp;
  }
  builder.setInsertionPoint(insertionAnchor);

  Value workspace =
      createAllocWorkspace(builder, loc, getMemRefType(sourceType),
                           getDynamicTensorDims(builder, loc,
                                                extractOp.getSource(),
                                                sourceType));
  materializeInDestination(builder, loc, extractOp.getSource(), workspace);

  Value subview =
      builder
          .create<memref::SubViewOp>(loc, workspace,
                                     extractOp.getMixedOffsets(),
                                     extractOp.getMixedSizes(),
                                     extractOp.getMixedStrides())
          .getResult();

  auto localType = getMemRefType(resultType);
  Value local = builder.create<memref::AllocOp>(
      loc, localType,
      getDynamicDimsFromMixedSizes(builder, loc, resultType,
                                   extractOp.getMixedSizes()));
  builder.create<memref::CopyOp>(loc, subview, local);

  Value tensor = toWritableTensor(builder, loc, local, resultType);
  extractOp.replaceAllUsesWith(tensor);
  extractOp.erase();
  return success();
}

static LogicalResult rewriteInsertSlice(OpBuilder &builder,
                                        tensor::InsertSliceOp insertOp,
                                        bool syncAfterSourceStore = false) {
  auto sourceType = dyn_cast<RankedTensorType>(insertOp.getSourceType());
  auto destType = dyn_cast<RankedTensorType>(insertOp.getDestType());
  auto resultType = dyn_cast<RankedTensorType>(insertOp.getType());
  if (!sourceType || !destType || !resultType)
    return failure();

  Location loc = insertOp.getLoc();
  builder.setInsertionPoint(insertOp);
  auto storageDestType =
      getI8StorageTensorType(destType, builder.getContext());
  auto storageResultType =
      getI8StorageTensorType(resultType, builder.getContext());

  Value workspace =
      createAllocWorkspace(builder, loc, getMemRefType(storageDestType),
                           getDynamicTensorDims(builder, loc, insertOp.getDest(),
                                                destType));

  Value subview =
      builder
          .create<memref::SubViewOp>(loc, workspace, insertOp.getMixedOffsets(),
                                     insertOp.getMixedSizes(),
                                     insertOp.getMixedStrides())
          .getResult();
  Value storageSource = castI1TensorToI8(builder, loc, insertOp.getSource());
  materializeInDestination(builder, loc,
                           expandSourceIfNeeded(builder, loc,
                                                storageSource, subview),
                           subview);
  if (syncAfterSourceStore)
    createVVCommunicationSync(builder, loc);

  auto localType = getMemRefType(storageResultType);
  Value local = builder.create<memref::AllocOp>(
      loc, localType,
      getDynamicTensorDims(builder, loc, insertOp.getDest(), resultType));
  builder.create<memref::CopyOp>(loc, workspace, local);

  Value tensor = toWritableTensor(builder, loc, local, storageResultType);
  tensor = castI8TensorToI1(builder, loc, tensor, resultType);
  insertOp.replaceAllUsesWith(tensor);
  insertOp.erase();
  return success();
}

static LogicalResult
rewriteVVCommInsertSlice(OpBuilder &builder,
                          tensor::InsertSliceOp insertOp) {
  SmallVector<tensor::ExtractSliceOp> pairedExtracts;
  for (Operation *user : insertOp->getUsers()) {
    auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (extractOp &&
        extractOp->hasAttrOfType<UnitAttr>(kVVCommunicationAttr))
      pairedExtracts.push_back(extractOp);
  }

  if (pairedExtracts.empty())
    return rewriteInsertSlice(builder, insertOp,
                              /*syncAfterSourceStore=*/true);

  auto destType = dyn_cast<RankedTensorType>(insertOp.getDestType());
  auto resultType = dyn_cast<RankedTensorType>(insertOp.getType());
  if (!destType || !resultType)
    return failure();

  Location loc = insertOp.getLoc();
  builder.setInsertionPoint(insertOp);
  auto storageDestType =
      getI8StorageTensorType(destType, builder.getContext());
  auto storageResultType =
      getI8StorageTensorType(resultType, builder.getContext());

  Value workspace =
      createAllocWorkspace(builder, loc, getMemRefType(storageDestType),
                           getDynamicTensorDims(builder, loc,
                                                insertOp.getDest(),
                                                destType));

  Value insertSubview =
      builder
          .create<memref::SubViewOp>(loc, workspace,
                                     insertOp.getMixedOffsets(),
                                     insertOp.getMixedSizes(),
                                     insertOp.getMixedStrides())
          .getResult();
  Value storageSource = castI1TensorToI8(builder, loc, insertOp.getSource());
  materializeInDestination(builder, loc,
                           expandSourceIfNeeded(builder, loc,
                                                storageSource, insertSubview),
                           insertSubview);
  createVVCommunicationSync(builder, loc);

  for (auto extractOp : pairedExtracts) {
    builder.setInsertionPoint(extractOp);
    auto extractResultType =
        dyn_cast<RankedTensorType>(extractOp.getType());
    if (!extractResultType)
      return failure();
    auto storageExtractResultType =
        getI8StorageTensorType(extractResultType, builder.getContext());

    Value extractSubview =
        builder
            .create<memref::SubViewOp>(
                extractOp.getLoc(), workspace,
                extractOp.getMixedOffsets(),
                extractOp.getMixedSizes(),
                extractOp.getMixedStrides())
            .getResult();

    auto localType = getMemRefType(storageExtractResultType);
    Value local = builder.create<memref::AllocOp>(
        extractOp.getLoc(), localType,
        getDynamicDimsFromMixedSizes(builder, extractOp.getLoc(),
                                     extractResultType,
                                     extractOp.getMixedSizes()));
    builder.create<memref::CopyOp>(extractOp.getLoc(), extractSubview,
                                   local);

    Value tensor =
        toWritableTensor(builder, extractOp.getLoc(), local,
                         storageExtractResultType);
    tensor = castI8TensorToI1(builder, extractOp.getLoc(), tensor,
                              extractResultType);
    extractOp.replaceAllUsesWith(tensor);
    extractOp.erase();
  }

  if (!insertOp->use_empty()) {
    builder.setInsertionPointAfter(insertOp);
    auto localType = getMemRefType(storageResultType);
    Value local = builder.create<memref::AllocOp>(
        loc, localType,
        getDynamicTensorDims(builder, loc, insertOp.getDest(),
                             resultType));
    builder.create<memref::CopyOp>(loc, workspace, local);
    Value tensor = toWritableTensor(builder, loc, local, storageResultType);
    tensor = castI8TensorToI1(builder, loc, tensor, resultType);
    insertOp.replaceAllUsesWith(tensor);
  }

  insertOp.erase();
  return success();
}

static LogicalResult rewriteVVCommInsert(OpBuilder &builder,
                                         tensor::InsertOp insertOp) {
  auto resultType = dyn_cast<RankedTensorType>(insertOp.getType());
  if (!resultType)
    return failure();

  Location loc = insertOp.getLoc();
  builder.setInsertionPoint(insertOp);

  Value workspace =
      createAllocWorkspace(builder, loc, getMemRefType(resultType),
                           getDynamicTensorDims(builder, loc,
                                                insertOp.getDest(),
                                                resultType));

  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<int64_t> sourceTensorShape(resultType.getRank(), 1);
  offsets.reserve(resultType.getRank());
  sizes.reserve(resultType.getRank());
  strides.reserve(resultType.getRank());
  for (Value index : insertOp.getIndices()) {
    offsets.push_back(index);
    sizes.push_back(builder.getIndexAttr(1));
    strides.push_back(builder.getIndexAttr(1));
  }

  auto sourceTensorType =
      RankedTensorType::get(sourceTensorShape, resultType.getElementType());
  Value sourceTensor = builder.create<tensor::FromElementsOp>(
      loc, sourceTensorType, ValueRange{insertOp.getScalar()});

  Value insertSubview =
      builder
          .create<memref::SubViewOp>(loc, workspace, offsets, sizes, strides)
          .getResult();
  materializeInDestination(builder, loc, sourceTensor, insertSubview);
  createVVCommunicationSync(builder, loc);

  auto localType = getMemRefType(resultType);
  Value local = builder.create<memref::AllocOp>(
      loc, localType,
      getDynamicTensorDims(builder, loc, insertOp.getDest(), resultType));
  builder.create<memref::CopyOp>(loc, workspace, local);

  Value tensor = toWritableTensor(builder, loc, local, resultType);
  insertOp.replaceAllUsesWith(tensor);
  insertOp.erase();
  return success();
}

struct RewriteCommunicationSlicePass
    : public mlir::triton::impl::RewriteCommunicationSliceBase<
          RewriteCommunicationSlicePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    SmallVector<tensor::InsertSliceOp> vvCommInsertOps;
    SmallVector<tensor::InsertOp> vvCommScalarInsertOps;
    SmallVector<Operation *> cvWorklist;
    moduleOp.walk([&](Operation *op) {
      if (op->hasAttrOfType<UnitAttr>(kVVCommunicationAttr)) {
        if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op))
          vvCommInsertOps.push_back(insertOp);
        else if (auto insertOp = dyn_cast<tensor::InsertOp>(op))
          vvCommScalarInsertOps.push_back(insertOp);
        return;
      }
      if (op->hasAttrOfType<UnitAttr>(kCVCommunicationSliceAttr)) {
        if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op))
          cvWorklist.push_back(op);
      }
    });

    for (auto insertOp : vvCommInsertOps) {
      if (failed(rewriteVVCommInsertSlice(builder, insertOp))) {
        insertOp->emitOpError("failed to rewrite vv communication slice");
        signalPassFailure();
        return;
      }
    }

    for (auto insertOp : vvCommScalarInsertOps) {
      if (failed(rewriteVVCommInsert(builder, insertOp))) {
        insertOp->emitOpError("failed to rewrite vv communication insert");
        signalPassFailure();
        return;
      }
    }

    for (Operation *op : cvWorklist) {
      if (!op->getBlock())
        continue;
      LogicalResult result = failure();
      if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op))
        result = rewriteExtractSlice(builder, extractOp);
      else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op))
        result = rewriteInsertSlice(builder, insertOp);

      if (failed(result)) {
        op->emitOpError("failed to rewrite cv communication slice");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createRewriteCommunicationSlicePass() {
  return std::make_unique<RewriteCommunicationSlicePass>();
}

} // namespace triton
} // namespace mlir
