/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

#include "Utils/DimensionGraphAnalyzer.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <queue>

#define DEBUG_TYPE "triton-dimension-graph-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();
  auto srcIt = srcShape.rbegin();
  auto dstIt = dstShape.rbegin();
  while (srcIt != srcShape.rend() && dstIt != dstShape.rend()) {
    if (*srcIt == 1 && *dstIt != 1)
      broadcastDims.push_back(
          static_cast<int64_t>(std::distance(dstShape.rbegin(), dstIt)));
    ++srcIt;
    ++dstIt;
  }
  while (dstIt != dstShape.rend()) {
    broadcastDims.push_back(
        static_cast<int64_t>(std::distance(dstShape.rbegin(), dstIt)));
    ++dstIt;
  }
  for (auto &dim : broadcastDims)
    dim = dstShape.size() - 1 - dim;
  return broadcastDims;
}

std::optional<std::pair<size_t, SmallVector<int64_t>>>
getValueShapeInfo(Value value) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType)
    return std::nullopt;
  return std::make_pair(shapedType.getRank(),
                        SmallVector<int64_t>(shapedType.getShape()));
}

SmallVector<int64_t> getShape(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return SmallVector<int64_t>(shapedType.getShape());
  return {};
}

bool hasElement(ArrayRef<int64_t> values, int64_t value) {
  return llvm::is_contained(values, value);
}

bool isDefinedByMetaUse(Value value) {
  Operation *defOp = value.getDefiningOp();
  return defOp && defOp->hasAttr("MetaUse");
}

void appendUnique(SmallVectorImpl<int64_t> &values, int64_t value) {
  if (!hasElement(values, value))
    values.push_back(value);
}

} // namespace

DimensionGraphAnalyzer::DimensionGraphAnalyzer(
    Operation *op, DimensionGraphAnalyzerOptions options)
    : op_(op), options_(std::move(options)) {}

LogicalResult DimensionGraphAnalyzer::initialize() {
  reset();

  if (failed(initializeStructures()))
    return failure();
  processBFS();
  removeSkippedDependencyEdges();
  rebuildComponents();
  dumpInitializeSummary();
  return success();
}

void DimensionGraphAnalyzer::reset() {
  valueIds_.clear();
  valuesById_.clear();
  argumentsRefPointer_.clear();
  axes_.clear();
  outEdges_.clear();
  inEdges_.clear();
  valueOutEdges_.clear();
  valueInEdges_.clear();
  tilingDim_.clear();
  headNodeSet_.clear();
  tailValues_.clear();
  tailValueSet_.clear();
  valueComponents_.clear();
  valueComponentIdByValue_.clear();
  axisComponents_.clear();
  axisComponentIdByAxis_.clear();
  nextValueId_ = 0;
}

void DimensionGraphAnalyzer::dumpInitializeSummary() const {
  LLVM_DEBUG({
    DBGS() << "initialize summary\n";
    DBGS() << "  values=" << valuesById_.size() << ", axes=" << axes_.size()
           << ", heads=" << headNodeSet_.size()
           << ", tails=" << tailValues_.size()
           << ", valueComponents=" << valueComponents_.size()
           << ", axisComponents=" << axisComponents_.size() << "\n";

    for (auto [idx, value] : llvm::enumerate(tailValues_))
      DBGS() << "  tail[" << idx << "] " << value << "\n";

    for (auto [componentId, component] : llvm::enumerate(valueComponents_)) {
      DBGS() << "  valueComponent[" << componentId << "] values=";
      for (int64_t valueId : component.valueIds)
        llvm::dbgs() << valueId << " ";
      llvm::dbgs() << "tails=";
      for (int64_t tailValueId : component.tailValueIds)
        llvm::dbgs() << tailValueId << " ";
      llvm::dbgs() << "axisComponents=";
      for (int64_t axisComponentId : component.axisComponentIds)
        llvm::dbgs() << axisComponentId << " ";
      llvm::dbgs() << "selectedAxisComponents=";
      for (int64_t axisComponentId : component.selectedAxisComponentIds)
        llvm::dbgs() << axisComponentId << " ";
      llvm::dbgs() << "\n";
    }

    for (auto [componentId, component] : llvm::enumerate(axisComponents_)) {
      DBGS() << "  axisComponent[" << componentId
             << "] axisCount=" << component.axisIds.size()
             << " tailAxes=";
      for (int64_t tailAxisId : component.tailAxisIds)
        llvm::dbgs() << tailAxisId << " ";
      llvm::dbgs() << "\n";
    }
  });
}

void DimensionGraphAnalyzer::dumpComputeSummary() const {
  LLVM_DEBUG({
    DBGS() << "computeTilingDim summary\n";
    for (auto [componentId, component] : llvm::enumerate(valueComponents_)) {
      DBGS() << "  valueComponent[" << componentId << "] selectedAxes=";
      for (int64_t axisId : component.selectedAxisIds)
        llvm::dbgs() << axisId << "(v="
                     << valueIds_.lookup(axes_[axisId].owner)
                     << ",d=" << axes_[axisId].dim
                     << ",shape=" << axes_[axisId].shape << ") ";
      llvm::dbgs() << "\n";

      for (int64_t valueId : component.valueIds) {
        Value value = valuesById_[valueId];
        auto it = tilingDim_.find(value);
        int64_t dim = it == tilingDim_.end() ? -1 : it->second;
        DBGS() << "    value[" << valueId << "] tilingDim=" << dim << " "
               << value << "\n";
      }
    }
  });
}

bool DimensionGraphAnalyzer::computeTilingDim() {
  for (const auto &entry : argumentsRefPointer_)
    tilingDim_[entry.first] = -1;
  LDBG("computeTilingDim begin");
  selectTilingAxes();
  dumpComputeSummary();
  return false;
}

int64_t DimensionGraphAnalyzer::getTilingDim(Value value) const {
  auto it = tilingDim_.find(value);
  if (it == tilingDim_.end())
    return -1;
  return it->second;
}

SmallVector<int64_t>
DimensionGraphAnalyzer::getArgumentRef(Value value) const {
  auto it = argumentsRefPointer_.find(value);
  if (it == argumentsRefPointer_.end())
    return {};
  return it->second;
}

SmallVector<int64_t>
DimensionGraphAnalyzer::getArgumentRefOrCreateDummy(Value value) {
  createDummyRefIfNotExist({value});
  return getArgumentRef(value);
}

LogicalResult DimensionGraphAnalyzer::initializeStructures() {
  op_->walk([&](Operation *operation) {
    bool isHead = false;
    if (options_.isHeadOp) {
      isHead = options_.isHeadOp(operation);
    } else {
      isHead = isa<triton::LoadOp, arith::ConstantOp, triton::SplatOp,
                   triton::MakeRangeOp>(operation);
    }
    if (!isHead)
      return;

    for (Value result : operation->getResults()) {
      if (!isa<RankedTensorType>(result.getType()))
        continue;
      createDummyRefIfNotExist({result});
      headNodeSet_.insert(result);
    }
  });

  return success();
}

void DimensionGraphAnalyzer::processBFS() {
  std::queue<Value> worklist;
  DenseSet<Value> visited;
  for (Value value : headNodeSet_) {
    worklist.push(value);
    visited.insert(value);
  }

  DenseSet<Operation *> processedOps;
  while (!worklist.empty()) {
    Value current = worklist.front();
    worklist.pop();

    bool hasSkippedUse = false;
    int64_t followedUseCount = 0;
    bool singleUseProducesScalar = false;
    for (OpOperand &use : current.getUses()) {
      if (isa<annotation::MarkOp>(use.getOwner()))
        continue;

      if (options_.shouldSkipUse && options_.shouldSkipUse(current, use)) {
        hasSkippedUse = true;
        continue;
      }

      ++followedUseCount;
      singleUseProducesScalar = useProducesScalar(use);

      Operation *user = use.getOwner();
      if (processedOps.insert(user).second)
        processOperation(user);

      for (Value result : user->getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        createDummyRefIfNotExist({result});
        if (visited.insert(result).second)
          worklist.push(result);
      }

      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        BlockArgument regionArg = forOp.getTiedLoopRegionIterArg(&use);
        if (!regionArg || !isa<RankedTensorType>(regionArg.getType()))
          continue;
        createDummyRefIfNotExist({regionArg});
        if (visited.insert(regionArg).second)
          worklist.push(regionArg);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
        unsigned operandNumber = use.getOperandNumber();
        if (operandNumber >= whileOp.getBeforeArguments().size())
          continue;
        Value beforeArg = whileOp.getBeforeArguments()[operandNumber];
        if (!isa<RankedTensorType>(beforeArg.getType()))
          continue;
        createDummyRefIfNotExist({beforeArg});
        if (visited.insert(beforeArg).second)
          worklist.push(beforeArg);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
        if (user->getParentOp() == nullptr ||
            use.getOperandNumber() == 0)
          continue;
        auto whileOp = dyn_cast<scf::WhileOp>(user->getParentOp());
        if (!whileOp)
          continue;
        unsigned argNumber = use.getOperandNumber() - 1;
        if (argNumber >= whileOp.getAfterArguments().size())
          continue;
        Value afterArg = whileOp.getAfterArguments()[argNumber];
        if (!isa<RankedTensorType>(afterArg.getType()))
          continue;
        createDummyRefIfNotExist({afterArg});
        if (visited.insert(afterArg).second)
          worklist.push(afterArg);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yieldOp->getParentOp();
        if (!parent)
          continue;
        unsigned operandNumber = use.getOperandNumber();
        if (operandNumber >= parent->getNumResults())
          continue;
        Value result = parent->getResult(operandNumber);
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        createDummyRefIfNotExist({result});
        if (visited.insert(result).second)
          worklist.push(result);
      }
    }

    bool isTail = hasSkippedUse || followedUseCount == 0 ||
                  (followedUseCount == 1 && singleUseProducesScalar);
    if (isTail)
      addTailValue(current);
  }
}

bool DimensionGraphAnalyzer::processOperation(Operation *operation) {
  return TypeSwitch<Operation *, bool>(operation)
      .Case<triton::BroadcastOp>([&](auto op) {
        processBroadcastOp(op);
        return true;
      })
      .Case<triton::ExpandDimsOp>([&](auto op) {
        processExpandDimsOp(op);
        return true;
      })
      .Case<triton::TransOp>([&](auto op) {
        processTransOp(op);
        return true;
      })
      .Case<triton::ReduceOp>([&](auto op) {
        processReduceOp(op);
        return true;
      })
      .Case<triton::ReshapeOp>([&](auto op) {
        processReshapeOp(op);
        return true;
      })
      .Case<triton::FpToFpOp>([&](auto op) {
        return processElementwiseOp(op);
      })
      .Case<scf::ForOp>([&](auto op) {
        processForOp(op);
        return true;
      })
      .Case<scf::WhileOp>([&](auto op) {
        processWhileOp(op);
        return true;
      })
      .Case<scf::ConditionOp>([&](auto op) {
        processConditionOp(op);
        return true;
      })
      .Case<scf::YieldOp>([&](auto op) {
        processYieldOp(op);
        return true;
      })
      .Default([&](Operation *op) {
        if (op->hasTrait<OpTrait::Elementwise>())
          return processElementwiseOp(op);
        return false;
      });
}

bool DimensionGraphAnalyzer::processElementwiseOp(Operation *operation) {
  SmallVector<Value> rankedResults;
  for (Value result : operation->getResults()) {
    if (isa<RankedTensorType>(result.getType()))
      rankedResults.push_back(result);
  }
  if (rankedResults.empty())
    return false;

  SmallVector<Value> operands(operation->getOperands().begin(),
                              operation->getOperands().end());
  createDummyRefIfNotExist(operands);
  createDummyRefIfNotExist(rankedResults);

  for (Value result : rankedResults) {
    auto resultArgs = getArgumentRef(result);
    auto resultRank = resultArgs.size();
    for (Value operand : operation->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || operandType.getRank() != resultRank)
        continue;
      auto operandArgs = getArgumentRef(operand);
      for (auto [dim, operandAxis] : llvm::enumerate(operandArgs))
        recordAxisDependency(operandAxis, resultArgs[dim]);
    }
  }

  return true;
}

void DimensionGraphAnalyzer::processForOp(scf::ForOp op) {
  auto initArgs = op.getInitArgs();
  auto regionArgs = op.getRegionIterArgs();

  for (auto [idx, initArg] : llvm::enumerate(initArgs)) {
    if (isDefinedByMetaUse(initArg))
      continue;
    if (idx < regionArgs.size())
      recordSameRankDependency(initArg, regionArgs[idx]);
  }
}

void DimensionGraphAnalyzer::processWhileOp(scf::WhileOp op) {
  auto initArgs = op.getInits();
  auto beforeArgs = op.getBeforeArguments();

  for (auto [idx, initArg] : llvm::enumerate(initArgs)) {
    if (isDefinedByMetaUse(initArg))
      continue;
    if (idx < beforeArgs.size())
      recordSameRankDependency(initArg, beforeArgs[idx]);
  }
}

void DimensionGraphAnalyzer::processConditionOp(scf::ConditionOp op) {
  Operation *parent = op->getParentOp();
  auto whileOp = dyn_cast_or_null<scf::WhileOp>(parent);
  if (!whileOp)
    return;

  auto afterArgs = whileOp.getAfterArguments();
  for (auto [idx, operand] : llvm::enumerate(op.getArgs())) {
    if (isDefinedByMetaUse(operand))
      continue;
    if (idx < afterArgs.size())
      recordSameRankDependency(operand, afterArgs[idx]);
  }
}

void DimensionGraphAnalyzer::processYieldOp(scf::YieldOp op) {
  Operation *parent = op->getParentOp();
  if (!parent)
    return;

  for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
    if (idx >= parent->getNumResults())
      break;
    if (isDefinedByMetaUse(operand))
      continue;
    recordSameRankDependency(operand, parent->getResult(idx));
  }
}

void DimensionGraphAnalyzer::recordSameRankDependency(Value src, Value dst) {
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  auto dstType = dyn_cast<RankedTensorType>(dst.getType());
  if (!srcType || !dstType || srcType.getRank() != dstType.getRank())
    return;

  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto dstArgs = getArgumentRefOrCreateDummy(dst);
  for (auto [dim, srcAxis] : llvm::enumerate(srcArgs))
    recordAxisDependency(srcAxis, dstArgs[dim]);
}

void DimensionGraphAnalyzer::processBroadcastOp(triton::BroadcastOp op) {
  Value src = op.getSrc();
  Value result = op.getResult();
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!srcType || !resultType)
    return;

  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resultArgs = getArgumentRefOrCreateDummy(result);
  auto broadcastDims = getBroadcastDims(srcType, resultType);
  DenseSet<int64_t> broadcastDimSet(broadcastDims.begin(),
                                    broadcastDims.end());

  if (srcArgs.size() == resultArgs.size()) {
    for (auto [idx, srcAxis] : llvm::enumerate(srcArgs))
      recordAxisDependency(srcAxis, resultArgs[idx]);
    return;
  }

  int64_t srcIdx = static_cast<int64_t>(srcArgs.size()) - 1;
  for (int64_t resultIdx = static_cast<int64_t>(resultArgs.size()) - 1;
       resultIdx >= 0; --resultIdx) {
    if (srcIdx < 0)
      break;
    recordAxisDependency(srcArgs[srcIdx], resultArgs[resultIdx]);
    if (!broadcastDimSet.contains(resultIdx))
      --srcIdx;
  }
}

void DimensionGraphAnalyzer::processExpandDimsOp(triton::ExpandDimsOp op) {
  auto srcArgs = getArgumentRefOrCreateDummy(op.getSrc());
  auto resultArgs = getArgumentRefOrCreateDummy(op.getResult());
  int64_t axis = op.getAxis();
  int64_t srcIdx = 0;
  for (int64_t resultIdx = 0; resultIdx < static_cast<int64_t>(resultArgs.size());
       ++resultIdx) {
    if (resultIdx == axis)
      continue;
    if (srcIdx >= static_cast<int64_t>(srcArgs.size()))
      return;
    recordAxisDependency(srcArgs[srcIdx++], resultArgs[resultIdx]);
  }
}

void DimensionGraphAnalyzer::processTransOp(triton::TransOp op) {
  auto srcArgs = getArgumentRefOrCreateDummy(op.getSrc());
  auto resultArgs = getArgumentRefOrCreateDummy(op.getResult());
  auto order = op.getOrder();
  for (int64_t resultIdx = 0; resultIdx < static_cast<int64_t>(resultArgs.size());
       ++resultIdx) {
    int64_t srcIdx = order[resultIdx];
    if (srcIdx >= 0 && srcIdx < static_cast<int64_t>(srcArgs.size()))
      recordAxisDependency(srcArgs[srcIdx], resultArgs[resultIdx]);
  }
}

void DimensionGraphAnalyzer::processReduceOp(triton::ReduceOp op) {
  SmallVector<Value> inputs(op.getOperands().begin(), op.getOperands().end());
  SmallVector<Value> outputs(op.getResults().begin(), op.getResults().end());
  if (inputs.empty())
    return;

  int64_t reduceAxis = op.getAxis();
  createDummyRefIfNotExist(inputs);
  createDummyRefIfNotExist(outputs);

  for (auto [outputIdx, output] : llvm::enumerate(outputs)) {
    if (!isa<RankedTensorType>(output.getType()))
      continue;
    Value input = inputs[std::min<size_t>(outputIdx, inputs.size() - 1)];
    auto inputArgs = getArgumentRef(input);
    auto outputArgs = getArgumentRef(output);
    int64_t resultDim = 0;
    for (int64_t inputDim = 0; inputDim < static_cast<int64_t>(inputArgs.size());
         ++inputDim) {
      if (inputDim == reduceAxis)
        continue;
      if (resultDim >= static_cast<int64_t>(outputArgs.size()))
        break;
      recordAxisDependency(inputArgs[inputDim], outputArgs[resultDim++]);
    }
  }
}

void DimensionGraphAnalyzer::processReshapeOp(triton::ReshapeOp op) {
  Value src = op.getSrc();
  Value result = op.getResult();
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resultArgs = getArgumentRefOrCreateDummy(result);
  auto srcShape = getShape(src.getType());
  auto resultShape = getShape(result.getType());
  size_t srcRank = srcArgs.size();
  size_t resultRank = resultArgs.size();

  size_t srcBegin = 0;
  size_t resultBegin = 0;
  while (srcBegin < srcRank || resultBegin < resultRank) {
    if (srcBegin < srcRank && resultBegin < resultRank &&
        srcShape[srcBegin] == resultShape[resultBegin]) {
      recordAxisDependency(srcArgs[srcBegin], resultArgs[resultBegin]);
      ++srcBegin;
      ++resultBegin;
      continue;
    }

    size_t srcEnd = srcBegin;
    size_t resultEnd = resultBegin;
    int64_t srcProduct = 1;
    int64_t resultProduct = 1;
    bool hasDynamicShape = false;

    do {
      if ((srcProduct <= resultProduct && srcEnd < srcRank) ||
          resultEnd == resultRank) {
        int64_t dim = srcShape[srcEnd++];
        if (ShapedType::isDynamic(dim))
          hasDynamicShape = true;
        else
          srcProduct *= dim;
      } else if (resultEnd < resultRank) {
        int64_t dim = resultShape[resultEnd++];
        if (ShapedType::isDynamic(dim))
          hasDynamicShape = true;
        else
          resultProduct *= dim;
      }
    } while (!hasDynamicShape && srcProduct != resultProduct &&
             (srcEnd < srcRank || resultEnd < resultRank));

    if (hasDynamicShape || srcProduct != resultProduct)
      return;

    auto srcNonUnitDims = getNonUnitDims(srcShape, srcBegin, srcEnd);
    auto resultNonUnitDims =
        getNonUnitDims(resultShape, resultBegin, resultEnd);

    if (!srcNonUnitDims.empty() && !resultNonUnitDims.empty())
      recordAxisDependency(srcArgs[srcNonUnitDims.front()],
                  resultArgs[resultNonUnitDims.front()]);

    srcBegin = srcEnd;
    resultBegin = resultEnd;
  }
}

int64_t DimensionGraphAnalyzer::getOrCreateValueId(Value value) {
  auto [it, inserted] = valueIds_.try_emplace(value, nextValueId_);
  if (inserted) {
    valuesById_.push_back(value);
    valueOutEdges_.push_back({});
    valueInEdges_.push_back({});
    ++nextValueId_;
  }
  return it->second;
}

SmallVector<int64_t> DimensionGraphAnalyzer::createAxesForValue(Value value) {
  auto shapeInfo = getValueShapeInfo(value).value_or(
      std::make_pair(0, SmallVector<int64_t>{}));
  auto rank = shapeInfo.first;
  auto shape = shapeInfo.second;
  getOrCreateValueId(value);

  SmallVector<int64_t> axisIds;
  axisIds.reserve(rank);
  for (size_t dim = 0; dim < rank; ++dim) {
    int64_t axisId = axes_.size();
    axes_.push_back(AxisInfo{value, static_cast<int64_t>(dim), shape[dim]});
    outEdges_.push_back({});
    inEdges_.push_back({});
    axisIds.push_back(axisId);
  }
  argumentsRefPointer_[value] = axisIds;
  tilingDim_[value] = -1;
  return axisIds;
}

void DimensionGraphAnalyzer::createDummyRefIfNotExist(ArrayRef<Value> values) {
  for (Value value : values) {
    if (argumentsRefPointer_.contains(value))
      continue;
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    createAxesForValue(value);
  }
}

void DimensionGraphAnalyzer::recordAxisDependency(int64_t srcAxis,
                                                 int64_t dstAxis) {
  if (srcAxis < 0 || dstAxis < 0 ||
      srcAxis >= static_cast<int64_t>(axes_.size()) ||
      dstAxis >= static_cast<int64_t>(axes_.size()))
    return;
  appendUnique(outEdges_[srcAxis], dstAxis);
  appendUnique(inEdges_[dstAxis], srcAxis);

  int64_t srcValue = getOrCreateValueId(axes_[srcAxis].owner);
  int64_t dstValue = getOrCreateValueId(axes_[dstAxis].owner);
  if (srcValue == dstValue)
    return;
  appendUnique(valueOutEdges_[srcValue], dstValue);
  appendUnique(valueInEdges_[dstValue], srcValue);
}

void DimensionGraphAnalyzer::removeSkippedDependencyEdges() {
  DenseSet<int64_t> skippedValueIds;
  DenseSet<int64_t> skippedAxisIds;

  for (auto [valueId, value] : llvm::enumerate(valuesById_)) {
    if (shouldSkipDependencyValue(value))
      skippedValueIds.insert(valueId);
  }

  for (auto [axisId, axis] : llvm::enumerate(axes_)) {
    if (shouldSkipDependencyValue(axis.owner))
      skippedAxisIds.insert(axisId);
  }

  auto removeSkippedAxes = [&](SmallVectorImpl<int64_t> &edges) {
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                               [&](int64_t axisId) {
                                 return skippedAxisIds.contains(axisId);
                               }),
                edges.end());
  };
  for (auto [axisId, edges] : llvm::enumerate(outEdges_)) {
    if (skippedAxisIds.contains(axisId))
      edges.clear();
    else
      removeSkippedAxes(edges);
  }
  for (auto [axisId, edges] : llvm::enumerate(inEdges_)) {
    if (skippedAxisIds.contains(axisId))
      edges.clear();
    else
      removeSkippedAxes(edges);
  }

  auto removeSkippedValues = [&](SmallVectorImpl<int64_t> &edges) {
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                               [&](int64_t valueId) {
                                 return skippedValueIds.contains(valueId);
                               }),
                edges.end());
  };
  for (auto [valueId, edges] : llvm::enumerate(valueOutEdges_)) {
    if (skippedValueIds.contains(valueId))
      edges.clear();
    else
      removeSkippedValues(edges);
  }
  for (auto [valueId, edges] : llvm::enumerate(valueInEdges_)) {
    if (skippedValueIds.contains(valueId))
      edges.clear();
    else
      removeSkippedValues(edges);
  }
}

void DimensionGraphAnalyzer::rebuildComponents() {
  rebuildValueComponents();
  rebuildAxisComponents();
  selectAxisComponentsForValueComponents();
}

void DimensionGraphAnalyzer::rebuildValueComponents() {
  valueComponents_.clear();
  valueComponentIdByValue_.assign(valuesById_.size(), -1);
  SmallVector<char> visited(valuesById_.size(), false);

  for (int64_t valueId = 0; valueId < static_cast<int64_t>(valuesById_.size());
       ++valueId) {
    if (visited[valueId])
      continue;

    int64_t componentId = valueComponents_.size();
    valueComponents_.push_back({});
    auto &component = valueComponents_.back();

    std::queue<int64_t> worklist;
    worklist.push(valueId);
    visited[valueId] = true;
    valueComponentIdByValue_[valueId] = componentId;

    while (!worklist.empty()) {
      int64_t current = worklist.front();
      worklist.pop();
      component.valueIds.push_back(current);

      for (int64_t next : valueOutEdges_[current]) {
        if (visited[next])
          continue;
        visited[next] = true;
        valueComponentIdByValue_[next] = componentId;
        worklist.push(next);
      }
      for (int64_t next : valueInEdges_[current]) {
        if (visited[next])
          continue;
        visited[next] = true;
        valueComponentIdByValue_[next] = componentId;
        worklist.push(next);
      }
    }

    for (int64_t valueId : component.valueIds) {
      if (tailValueSet_.contains(valuesById_[valueId]))
        component.tailValueIds.push_back(valueId);
    }
  }
}

void DimensionGraphAnalyzer::rebuildAxisComponents() {
  axisComponents_.clear();
  axisComponentIdByAxis_.assign(axes_.size(), -1);
  SmallVector<char> visited(axes_.size(), false);

  for (int64_t axisId = 0; axisId < static_cast<int64_t>(axes_.size());
       ++axisId) {
    if (visited[axisId])
      continue;

    int64_t componentId = axisComponents_.size();
    axisComponents_.push_back({});
    auto &component = axisComponents_.back();

    std::queue<int64_t> worklist;
    worklist.push(axisId);
    visited[axisId] = true;
    axisComponentIdByAxis_[axisId] = componentId;

    while (!worklist.empty()) {
      int64_t current = worklist.front();
      worklist.pop();
      component.axisIds.push_back(current);

      for (int64_t next : outEdges_[current]) {
        if (visited[next])
          continue;
        visited[next] = true;
        axisComponentIdByAxis_[next] = componentId;
        worklist.push(next);
      }
      for (int64_t next : inEdges_[current]) {
        if (visited[next])
          continue;
        visited[next] = true;
        axisComponentIdByAxis_[next] = componentId;
        worklist.push(next);
      }
    }
  }
}

void DimensionGraphAnalyzer::selectAxisComponentsForValueComponents() {
  for (auto &axisComponent : axisComponents_)
    axisComponent.tailAxisIds.clear();

  for (auto &valueComponent : valueComponents_) {
    valueComponent.axisComponentIds.clear();
    valueComponent.selectedAxisComponentIds.clear();

    for (int64_t valueId : valueComponent.valueIds) {
      Value value = valuesById_[valueId];
      for (int64_t axisId : getArgumentRef(value)) {
        if (axisId < 0 ||
            axisId >= static_cast<int64_t>(axisComponentIdByAxis_.size()))
          continue;
        appendUnique(valueComponent.axisComponentIds,
                     axisComponentIdByAxis_[axisId]);
      }
    }

    if (valueComponent.tailValueIds.empty())
      continue;

    for (int64_t axisComponentId : valueComponent.axisComponentIds) {
      bool coversAllTailValues = true;
      for (int64_t tailValueId : valueComponent.tailValueIds) {
        Value tailValue = valuesById_[tailValueId];
        // Loop results supplied by scf.yield do not require coverage.
        if (isa_and_nonnull<scf::ForOp, scf::WhileOp>(
                tailValue.getDefiningOp()))
          continue;

        bool coversTailValue = false;
        for (int64_t axisId : getArgumentRef(tailValue)) {
          if (axisComponentIdByAxis_[axisId] == axisComponentId) {
            coversTailValue = true;
            break;
          }
        }
        if (!coversTailValue) {
          coversAllTailValues = false;
          break;
        }
      }
      if (!coversAllTailValues)
        continue;

      appendUnique(valueComponent.selectedAxisComponentIds, axisComponentId);
      for (int64_t tailValueId : valueComponent.tailValueIds) {
        for (int64_t axisId : getArgumentRef(valuesById_[tailValueId])) {
          if (axisComponentIdByAxis_[axisId] == axisComponentId)
            appendUnique(axisComponents_[axisComponentId].tailAxisIds, axisId);
        }
      }
    }
  }
}

void DimensionGraphAnalyzer::addTailValue(Value value) {
  if (options_.isTailNode && !options_.isTailNode(value))
    return;
  if (tailValueSet_.insert(value).second)
    tailValues_.push_back(value);
}

bool DimensionGraphAnalyzer::useProducesScalar(OpOperand &use) const {
  Operation *user = use.getOwner();
  if (user->getNumResults() == 0)
    return false;

  for (Value result : user->getResults()) {
    auto shapedType = dyn_cast<ShapedType>(result.getType());
    if (shapedType && shapedType.getRank() > 0)
      return false;
  }
  return true;
}

bool DimensionGraphAnalyzer::shouldSkipDependencyValue(Value value) const {
  Operation *defOp = value.getDefiningOp();
  return defOp &&
         isa<arith::ConstantOp, triton::MakeRangeOp, triton::SplatOp>(defOp);
}

void DimensionGraphAnalyzer::selectTilingAxes() {
  for (auto &valueComponent : valueComponents_)
    valueComponent.selectedAxisIds.clear();

  for (auto [valueComponentId, valueComponent] :
       llvm::enumerate(valueComponents_)) {
    LDBG("selectTilingAxes valueComponent[" << valueComponentId
                                            << "] values="
                                            << valueComponent.valueIds.size()
                                            << " selectedAxisComponents="
                                            << valueComponent
                                                   .selectedAxisComponentIds
                                                   .size());
    AxisSetList candidates;
    for (int64_t axisComponentId : valueComponent.selectedAxisComponentIds) {
      auto visitedSets = traverseAxisComponent(axisComponentId);
      LDBG("  axisComponent[" << axisComponentId
                              << "] produced candidates="
                              << visitedSets.size());
      candidates.append(visitedSets.begin(), visitedSets.end());
    }

    if (candidates.empty()) {
      LDBG("  no candidates for valueComponent[" << valueComponentId << "]");
      continue;
    }

    AxisSet bestSet;
    bool hasBestSet = false;
    for (auto [candidateId, candidate] : llvm::enumerate(candidates)) {
      AxisSet filteredCandidate = removeUnitShapeAxes(candidate);
      LLVM_DEBUG({
        DBGS() << "  candidate[" << candidateId
               << "] rawSize=" << candidate.size()
               << " filteredSize=" << filteredCandidate.size()
               << " score=" << getAxisSetScore(filteredCandidate) << "\n";
        dumpAxisSet("    raw", candidate);
        dumpAxisSet("    filtered", filteredCandidate);
      });
      if (!hasBestSet || filteredCandidate.size() > bestSet.size() ||
          (filteredCandidate.size() == bestSet.size() &&
           getAxisSetScore(filteredCandidate) < getAxisSetScore(bestSet))) {
        bestSet = std::move(filteredCandidate);
        hasBestSet = true;
      }
    }
    if (!hasBestSet)
      continue;

    for (int64_t axisId : bestSet)
      valueComponent.selectedAxisIds.push_back(axisId);
    llvm::sort(valueComponent.selectedAxisIds);
    dumpAxisList("  selected", valueComponent.selectedAxisIds);

    for (int64_t valueId : valueComponent.valueIds) {
      Value value = valuesById_[valueId];
      tilingDim_[value] = -1;
      for (int64_t axisId : valueComponent.selectedAxisIds) {
        const AxisInfo &axis = axes_[axisId];
        if (axis.owner != value)
          continue;
        if (tilingDim_[value] == -1 || axis.dim < tilingDim_[value])
          tilingDim_[value] = axis.dim;
      }
    }
  }
}

DimensionGraphAnalyzer::AxisSetList
DimensionGraphAnalyzer::traverseAxisComponent(int64_t axisComponentId) const {
  AxisSetList visitedList;
  if (axisComponentId < 0 ||
      axisComponentId >= static_cast<int64_t>(axisComponents_.size()))
    return visitedList;

  SmallVector<int64_t> frontier;
  for (int64_t tailAxisId : axisComponents_[axisComponentId].tailAxisIds)
    appendUnique(frontier, tailAxisId);

  LLVM_DEBUG({
    DBGS() << "traverseAxisComponent[" << axisComponentId
           << "] axisCount="
           << axisComponents_[axisComponentId].axisIds.size()
           << " tailAxisCount=" << frontier.size() << "\n";
    dumpAxisList("  initialFrontier", frontier);
  });

  recursiveTraversal(frontier, AxisSet{}, AxisSet{}, visitedList, 0, 0,
                     axisComponentId);
  LDBG("traverseAxisComponent[" << axisComponentId
                                << "] finished candidates="
                                << visitedList.size());
  return visitedList;
}

void DimensionGraphAnalyzer::recursiveTraversal(
    SmallVector<int64_t> frontier, AxisSet visited, AxisSet selectedGroup,
    AxisSetList &visitedList, int64_t failNum, int64_t level,
    int64_t axisComponentId) const {
  int64_t iteration = 0;
  int64_t maxIterations =
      std::max<int64_t>(1024, static_cast<int64_t>(axes_.size()) * 64);
  while (!frontier.empty()) {
    ++iteration;
    if (iteration > maxIterations) {
      LLVM_DEBUG({
        DBGS() << "recursiveTraversal reached iteration limit "
               << maxIterations << " for axisComponent[" << axisComponentId
               << "] failNum=" << failNum << " level=" << level << "\n";
        dumpAxisList("  frontier", frontier);
        dumpAxisSet("  visited", visited);
        dumpAxisSet("  selectedGroup", selectedGroup);
      });
      visitedList.push_back(std::move(visited));
      return;
    }

    int64_t frontierSize = frontier.size();
    if (failNum == 2 * frontierSize && level == 1)
      level = 0;
    if (failNum == 2 * frontierSize && level == 2)
      level = 1;
    if (failNum == 3 * frontierSize && level == 1)
      level = 0;

    if (failNum >= frontierSize && level == 0) {
      visitedList.push_back(std::move(visited));
      return;
    }

    int64_t axis = frontier.front();
    frontier.erase(frontier.begin());
    LLVM_DEBUG({
      DBGS() << "recursiveTraversal axisComponent[" << axisComponentId
             << "] iter=" << iteration << " axis=" << axis
             << " frontierSize=" << frontierSize
             << " failNum=" << failNum << " level=" << level
             << " visited=" << visited.size()
             << " selectedGroup=" << selectedGroup.size() << "\n";
    });

    if (selectedGroup.contains(axis)) {
      LDBG("  axis " << axis << " is in selectedGroup");
      standardAction(axis, frontier, visited, selectedGroup, failNum, level,
                     axisComponentId);
      continue;
    }

    auto neighbors = getNeighborsInSameValue(axis);
    bool neighborInComponent = llvm::any_of(neighbors, [&](int64_t neighbor) {
      return isAxisInComponent(neighbor, axisComponentId);
    });

    if (neighborInComponent) {
      bool neighborInVisited = llvm::any_of(neighbors, [&](int64_t neighbor) {
        return visited.contains(neighbor);
      });
      bool neighborInFrontier = llvm::any_of(neighbors, [&](int64_t neighbor) {
        return hasElement(frontier, neighbor);
      });

      if (neighborInVisited) {
        LDBG("  axis " << axis << " has same-value neighbor in visited");
        frontier.push_back(axis);
        ++failNum;
        continue;
      }

      if (neighborInFrontier) {
        if (failNum < frontierSize) {
          LDBG("  axis " << axis
                         << " has same-value neighbor in frontier; defer");
          frontier.push_back(axis);
          ++failNum;
          level = std::max<int64_t>(level, 2);
          continue;
        }

        if (frontierSize <= failNum && failNum < 2 * frontierSize) {
          SmallVector<int64_t> frontierTmp = frontier;
          AxisSet visitedTmp = visited;
          AxisSet selectedGroupTmp = selectedGroup;
          int64_t failNumTmp = failNum;
          int64_t levelTmp = level;

          bool success =
              standardAction(axis, frontierTmp, visitedTmp, selectedGroupTmp,
                             failNumTmp, levelTmp, axisComponentId);
          if (success) {
            AxisSet selectedSet =
                generateSelectedGroup(axis, axisComponentId);
            LLVM_DEBUG({
              DBGS() << "  branch success for axis " << axis
                     << ", generated selectedSet size="
                     << selectedSet.size() << "\n";
              dumpAxisSet("    selectedSet", selectedSet);
            });
            for (int64_t selectedAxis : selectedSet)
              selectedGroupTmp.insert(selectedAxis);
            recursiveTraversal(std::move(frontierTmp), std::move(visitedTmp),
                               std::move(selectedGroupTmp), visitedList,
                               failNumTmp, levelTmp, axisComponentId);
          }

          frontier.push_back(axis);
          ++failNum;
          continue;
        }

        frontier.push_back(axis);
        ++failNum;
        continue;
      }

      if (failNum >= frontierSize && level == 1) {
        LDBG("  axis " << axis
                       << " has same-value neighbor in component; force action");
        standardAction(axis, frontier, visited, selectedGroup, failNum, level,
                       axisComponentId);
        continue;
      }

      frontier.push_back(axis);
      ++failNum;
      level = std::max<int64_t>(level, 1);
      continue;
    }

    standardAction(axis, frontier, visited, selectedGroup, failNum, level,
                   axisComponentId);
  }

  visitedList.push_back(std::move(visited));
}

void DimensionGraphAnalyzer::dumpAxisSet(StringRef label,
                                         const AxisSet &axisSet) const {
  LLVM_DEBUG({
    DBGS() << label << " axes=";
    for (int64_t axisId : axisSet) {
      if (axisId < 0 || axisId >= static_cast<int64_t>(axes_.size())) {
        llvm::dbgs() << axisId << "(invalid) ";
        continue;
      }
      const AxisInfo &axis = axes_[axisId];
      llvm::dbgs() << axisId << "(v=" << valueIds_.lookup(axis.owner)
                   << ",d=" << axis.dim << ",shape=" << axis.shape << ") ";
    }
    llvm::dbgs() << "\n";
  });
}

void DimensionGraphAnalyzer::dumpAxisList(StringRef label,
                                          ArrayRef<int64_t> axisIds) const {
  LLVM_DEBUG({
    DBGS() << label << " axes=";
    for (int64_t axisId : axisIds) {
      if (axisId < 0 || axisId >= static_cast<int64_t>(axes_.size())) {
        llvm::dbgs() << axisId << "(invalid) ";
        continue;
      }
      const AxisInfo &axis = axes_[axisId];
      llvm::dbgs() << axisId << "(v=" << valueIds_.lookup(axis.owner)
                   << ",d=" << axis.dim << ",shape=" << axis.shape << ") ";
    }
    llvm::dbgs() << "\n";
  });
}

bool DimensionGraphAnalyzer::standardAction(
    int64_t axis, SmallVectorImpl<int64_t> &frontier, AxisSet &visited,
    AxisSet &selectedGroup, int64_t &failNum, int64_t &level,
    int64_t axisComponentId) const {
  auto outEdges = getOutEdgesInAxisComponent(axis, axisComponentId);
  bool allOutVisited = llvm::all_of(outEdges, [&](int64_t outEdge) {
    return visited.contains(outEdge);
  });

  if (!allOutVisited) {
    LLVM_DEBUG({
      DBGS() << "  standardAction fail axis=" << axis
             << " outEdgesNotVisited=";
      for (int64_t outEdge : outEdges) {
        if (!visited.contains(outEdge))
          llvm::dbgs() << outEdge << " ";
      }
      llvm::dbgs() << "\n";
    });
    frontier.push_back(axis);
    ++failNum;
    return false;
  }

  auto inEdges = getInEdgesInAxisComponent(axis, axisComponentId);
  for (int64_t inEdge : inEdges) {
    if (!hasElement(frontier, inEdge))
      frontier.push_back(inEdge);
  }

  visited.insert(axis);
  selectedGroup.erase(axis);
  failNum = 0;
  level = 0;
  LLVM_DEBUG({
    DBGS() << "  standardAction success axis=" << axis
           << " addInEdges=" << inEdges.size()
           << " frontier=" << frontier.size()
           << " visited=" << visited.size() << "\n";
  });
  return true;
}

SmallVector<int64_t>
DimensionGraphAnalyzer::getNeighborsInSameValue(int64_t axisId) const {
  SmallVector<int64_t> neighbors;
  if (axisId < 0 || axisId >= static_cast<int64_t>(axes_.size()))
    return neighbors;

  auto it = argumentsRefPointer_.find(axes_[axisId].owner);
  if (it == argumentsRefPointer_.end())
    return neighbors;

  for (int64_t neighbor : it->second) {
    if (neighbor != axisId)
      neighbors.push_back(neighbor);
  }
  return neighbors;
}

SmallVector<int64_t> DimensionGraphAnalyzer::getInEdgesInAxisComponent(
    int64_t axisId, int64_t axisComponentId) const {
  SmallVector<int64_t> edges;
  if (axisId < 0 || axisId >= static_cast<int64_t>(inEdges_.size()))
    return edges;
  for (int64_t edge : inEdges_[axisId]) {
    if (isAxisInComponent(edge, axisComponentId))
      edges.push_back(edge);
  }
  return edges;
}

SmallVector<int64_t> DimensionGraphAnalyzer::getOutEdgesInAxisComponent(
    int64_t axisId, int64_t axisComponentId) const {
  SmallVector<int64_t> edges;
  if (axisId < 0 || axisId >= static_cast<int64_t>(outEdges_.size()))
    return edges;
  for (int64_t edge : outEdges_[axisId]) {
    if (isAxisInComponent(edge, axisComponentId))
      edges.push_back(edge);
  }
  return edges;
}

DimensionGraphAnalyzer::AxisSet
DimensionGraphAnalyzer::generateSelectedGroup(int64_t axisId,
                                              int64_t axisComponentId) const {
  AxisSet selectedGroup;
  SmallVector<int64_t> worklist;
  for (int64_t inEdge : getInEdgesInAxisComponent(axisId, axisComponentId)) {
    if (selectedGroup.insert(inEdge).second)
      worklist.push_back(inEdge);
  }

  while (!worklist.empty()) {
    int64_t current = worklist.pop_back_val();
    for (int64_t inEdge : getInEdgesInAxisComponent(current, axisComponentId)) {
      if (selectedGroup.insert(inEdge).second)
        worklist.push_back(inEdge);
    }
  }
  return selectedGroup;
}

bool DimensionGraphAnalyzer::isAxisInComponent(int64_t axisId,
                                               int64_t axisComponentId) const {
  if (axisId < 0 ||
      axisId >= static_cast<int64_t>(axisComponentIdByAxis_.size()))
    return false;
  return axisComponentIdByAxis_[axisId] == axisComponentId;
}

DimensionGraphAnalyzer::AxisSet
DimensionGraphAnalyzer::removeUnitShapeAxes(const AxisSet &axisSet) const {
  AxisSet filtered;
  for (int64_t axisId : axisSet) {
    if (axisId < 0 || axisId >= static_cast<int64_t>(axes_.size()))
      continue;
    if (axes_[axisId].shape != 1)
      filtered.insert(axisId);
  }
  return filtered;
}

int64_t
DimensionGraphAnalyzer::getAxisSetScore(const AxisSet &axisSet) {
  int64_t score = 0;
  for (int64_t axisId : axisSet)
    score += axisId;
  return score;
}

SmallVector<int64_t>
DimensionGraphAnalyzer::getNonUnitDims(ArrayRef<int64_t> shape, size_t begin,
                                       size_t end) const {
  SmallVector<int64_t> dims;
  for (size_t i = begin; i < end; ++i) {
    if (shape[i] != 1)
      dims.push_back(static_cast<int64_t>(i));
  }
  return dims;
}
