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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "Utils/DimensionAnalyzer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "triton-dimension-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {

SmallVector<int64_t> getBroadcastDims(RankedTensorType src, RankedTensorType dst) {
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

static bool isStaticUnitDim(Dimension dim) {
  auto shapedType = dyn_cast<ShapedType>(dim.first.getType());
  if (!shapedType || dim.second < 0 ||
      dim.second >= static_cast<int64_t>(shapedType.getRank()))
    return false;
  return shapedType.getDimSize(dim.second) == 1;
}

bool isElemwiseNaryOp(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>();
}

} // namespace

int UnionFindBase::find(int x) {
  allocateMinimum(x);
  if (parent_[x] < 0)
    return x;
  return parent_[x] = find(parent_[x]);
}

bool UnionFindBase::join(int a, int b) {
  allocateMinimum(std::max(a, b));
  a = find(a);
  b = find(b);
  if (a != b) {
    if (parent_[a] > parent_[b])
      std::swap(a, b);
    parent_[a] += parent_[b];
    parent_[b] = a;
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
  }
  return true;
}

void UnionFindBase::allocateMinimum(size_t n) {
  if (n + 1 > parent_.size()) {
    parent_.resize(n + 1, -1);
    size_t oldSize = minIndex.size();
    minIndex.resize(n + 1, -1);
    for (size_t i = oldSize; i < n + 1; ++i)
      minIndex[i] = static_cast<int>(i);
  }
}

bool ExtendedUnionFind::join(int a, int b) {
  assert(shape_[a] != kUndefinedShaped);
  assert(shape_[b] != kUndefinedShaped);
  assert(minParentIndex_.size() == parent_.size());
  a = find(a);
  b = find(b);
  if (a != b) {
    if (parent_[a] > parent_[b])
      std::swap(a, b);
    parent_[a] += parent_[b];
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
    minParentIndex_[a] = std::min(minParentIndex_[a], minParentIndex_[b]);
    assert(minParentIndex_[a] != kMaxDimPos);
    if (shape_[a] != ShapedType::kDynamic &&
        shape_[b] != ShapedType::kDynamic) {
      if (shape_[a] != shape_[b])
        return false;
    }
    if (shape_[b] != ShapedType::kDynamic)
      shape_[a] = shape_[b];
    parent_[b] = a;
  }
  return true;
}

void ExtendedUnionFind::allocateMinimum(size_t n) {
  UnionFindBase::allocateMinimum(n);
  if (n + 1 > shape_.size()) {
    shape_.resize(n + 1, kUndefinedShaped);
    minParentIndex_.resize(n + 1, kMaxDimPos);
  }
}

std::pair<DimensionPosition, int64_t>
ExtendedUnionFind::getMinParentAndShapePair(int a) {
  auto parent = find(a);
  return std::make_pair(minParentIndex_[parent], shape_[parent]);
}

DimensionAnalyzer::DimensionAnalyzer(Operation *op, DimensionAnalyzerOptions options)
    : op_(op), options(options) {}

LogicalResult DimensionAnalyzer::initialize() {
  bool hasFunctionCall = false;
  op_->walk([&](Operation *op) {
    if (isa<func::CallOp>(op))
      hasFunctionCall = true;
    return WalkResult::advance();
  });
  if (hasFunctionCall) {
    LDBG("Skipping function with function call inside");
    return failure();
  }

  solverGroup_ = std::make_unique<SimpleUnionFind>();

  initializeStructures();
  processBFS();
  unifyGroups();
  propagateConnection();
  spreadConnection();
  markDimensions();
  transferDimMark();
  return success();
}

bool DimensionAnalyzer::isAllowedType(Type type) {
  return isa_and_present<RankedTensorType>(type);
}

std::optional<size_t> DimensionAnalyzer::getShapeRankFromType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.getRank();
  return std::nullopt;
}

std::optional<std::pair<size_t, SmallVector<int64_t>>>
DimensionAnalyzer::getValueShapeInfo(Value v) {
  auto type = v.getType();
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return std::nullopt;
  size_t rank = shapedType.getRank();
  SmallVector<int64_t> shape(shapedType.getShape().begin(),
                             shapedType.getShape().end());
  return std::make_pair(rank, shape);
}

SmallVector<int64_t> DimensionAnalyzer::getShape(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return SmallVector<int64_t>(shapedType.getShape());
  return {};
}

void DimensionAnalyzer::initializeStructures()
{
    solverShapeElem_ = std::make_unique<ExtendedUnionFind>();
    solverCollapserElem_ = std::make_unique<SimpleUnionFind>();
    solverSegments_ = std::make_unique<SimpleUnionFind>();

    size_t sizeCount = 0;
    for (Block &block : op_->getRegion(0)) {
        sizeCount += block.getOperations().size();

        for (BlockArgument arg : block.getArguments()) {
            if (isAllowedType(arg.getType())) {
                headNodeSet.insert(arg);
                processArgument(arg);
            }
        }

        block.walk([&](Operation *op) {
            if (options.isHeadOp) {
                if (options.isHeadOp(op)) {
                    for (auto result : op->getResults()) {
                        if (isAllowedType(result.getType())) {
                            headNodeSet.insert(result);
                            processArgument(result);
                        }
                    }
                }
            } else {
                if (isa<tensor::EmptyOp, triton::LoadOp>(op)) {
                    for (auto result : op->getResults()) {
                        if (isAllowedType(result.getType())) {
                            headNodeSet.insert(result);
                            processArgument(result);
                        }
                    }
                }
            }
        });
    }

    solverSegments_->allocateMinimum(sizeCount);
    assert(dimensionAllocation_ == ssize_t(argumentList_.size()));
}

int64_t DimensionAnalyzer::allocateArguments(int rank,
                                             ArrayRef<int64_t> dimensionRef) {
  auto startingIdx = argumentTotalLength_;
  argumentTotalLength_ += rank + 1;
  isConnected_.resize(argumentTotalLength_);
  solverShapeElem_->allocateMinimum(argumentTotalLength_);
  solverCollapserElem_->allocateMinimum(argumentTotalLength_);

  for (int64_t i = 0; i < rank; ++i) {
    int64_t currentIndex = startingIdx + i;
    solverShapeElem_->minParentIndex_[currentIndex] = {dimensionAllocation_, i};
    solverShapeElem_->shape_[currentIndex] = dimensionRef[i];
    isConnected_[currentIndex].elementKind =
        dimensionRef[i] == 1 ? ConnectedLeftRight::ElementKind::Unit
                              : ConnectedLeftRight::ElementKind::NoMutation;
    if (i > 0)
      isConnected_[currentIndex].leftConnected = true;
    if (i + 1 < rank)
      isConnected_[currentIndex].rightConnected = true;
  }

  isConnected_[startingIdx + rank].leftConnected = false;
  isConnected_[startingIdx + rank].rightConnected = false;
  dimensionAllocation_++;

  return startingIdx;
}

void DimensionAnalyzer::processArgument(Value arg) {
  argumentList_.push_back(arg);

  auto shapeInfo = getValueShapeInfo(arg).value_or(
      std::make_pair(0, DimensionShape{}));
  auto rank = shapeInfo.first;
  auto shape = shapeInfo.second;

  LLVM_DEBUG(llvm::dbgs() << "Found args: " << arg << ' ' << rank << "\n");
  auto startingIdx = allocateArguments(rank, shape);
  initCollapseOrVerify(arg, argumentsRef_.size());
  argumentsRef_.push_back(DimensionShape(shape));
  std::iota(argumentsRef_.back().begin(), argumentsRef_.back().end(),
            startingIdx);
  LLVM_DEBUG({
    llvm::dbgs() << "[" DEBUG_TYPE "]: ";
    for (auto val : argumentsRef_.back())
      llvm::dbgs() << val << " ";
    llvm::dbgs() << "\n";
  });
}

void DimensionAnalyzer::processBFS() {
  std::queue<Value> bfsQueue;
  for (const auto &value : headNodeSet) {
    updatePreviousType(value);
    bfsQueue.push(value);
  }
  DenseSet<Value> visited(headNodeSet.begin(), headNodeSet.end());
  combineInferable();

  while (!bfsQueue.empty()) {
    Value current = bfsQueue.front();
    bfsQueue.pop();

    bool hasSkippedUse = false;
    for (auto &use : current.getUses()) {
      auto *user = use.getOwner();

      if (options.shouldSkipUse && options.shouldSkipUse(current, use)) {
        LDBG("Skipping use of " << current << " in " << *user);
        hasSkippedUse = true;
        continue;
      }

      processOperation(user, current);
      if (isa<ShapedType>(current.getType())) {
        createDummyRefIfNotExist({current});
        auto curRef = argumentsRefPointer_.at(current);
        if (auto forOp = dyn_cast<scf::ForOp>(user)) {
          auto regionArg = forOp.getTiedLoopRegionIterArg(&use);
          if (!regionArg)
            continue;
          auto res = forOp.getTiedLoopResult(&use);
          createDummyRefIfNotExist({regionArg, res});
          if (visited.insert(regionArg).second)
            bfsQueue.push(regionArg);
          solverGroup_->join(curRef, argumentsRefPointer_.at(regionArg));
          solverGroup_->join(curRef, argumentsRefPointer_.at(res));
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
          auto oprNum = use.getOperandNumber();
          auto arg = whileOp.getBeforeArguments()[oprNum];
          createDummyRefIfNotExist({arg});
          if (visited.insert(arg).second)
            bfsQueue.push(arg);
          solverGroup_->join(curRef, argumentsRefPointer_.at(arg));
        } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
          auto whileOp = cast<scf::WhileOp>(user->getParentOp());
          auto oprNum = use.getOperandNumber() - 1;
          for (auto arg :
               SmallVector<Value>{whileOp.getAfterArguments()[oprNum],
                                  whileOp->getResult(oprNum)}) {
            createDummyRefIfNotExist({arg});
            if (visited.insert(arg).second)
              bfsQueue.push(arg);
            solverGroup_->join(curRef, argumentsRefPointer_.at(arg));
          }
        } else {
          for (auto res : user->getResults()) {
            if (isa<ShapedType>(res.getType())) {
              createDummyRefIfNotExist({res});
              solverGroup_->join(curRef, argumentsRefPointer_.at(res));
            }
          }
        }
      }

      for (Value result : user->getResults()) {
        updatePreviousType(result);
        if (visited.insert(result).second)
          bfsQueue.push(result);
      }

      if (isa<scf::YieldOp, scf::ConditionOp>(user)) {
        auto *parentOp = user->getParentOp();
        processOperation(parentOp, current);

        if (isa<ShapedType>(current.getType())) {
          auto oprNum = use.getOperandNumber();
          if (isa<scf::ConditionOp>(user))
            oprNum -= 1;
          auto curRef = argumentsRefPointer_.at(current);
          auto res = parentOp->getResult(oprNum);
          createDummyRefIfNotExist({res});
          solverGroup_->join(curRef, argumentsRefPointer_.at(res));
        }
        for (Value result : parentOp->getResults()) {
          updatePreviousType(result);
          if (visited.insert(result).second)
            bfsQueue.push(result);
        }
        if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
          for (Value init : loopOp.getInits()) {
            updatePreviousType(init);
            if (visited.insert(init).second)
              bfsQueue.push(init);
          }
        }
      }
    }

    if (options.shouldSkipUse && hasSkippedUse)
      tailNodeSet.insert(current);
  }
}

bool DimensionAnalyzer::processOperation(Operation *op, Value current) {
  LDBG("Processing operation: " << *op);

  return TypeSwitch<Operation *, bool>(op)
      .Case<triton::SplatOp>([this](auto op) {
        processSplatOp(op);
        return true;
      })
      .Case<triton::BroadcastOp>([this](auto op) {
        processBroadcastOp(op);
        return true;
      })
      .Case<triton::ExpandDimsOp>([this](auto op) {
        processExpandDimsOp(op);
        return true;
      })
      .Case<triton::TransOp>([this](auto op) {
        processTransOp(op);
        return true;
      })
      .Case<triton::ReduceOp>([this](auto op) {
        processReduceOp(op);
        return true;
      })
      .Case<triton::DotOp>([this](auto op) {
        processDotOp(op);
        return true;
      })
      .Case<triton::DotScaledOp>([this](auto op) {
        processDotScaledOp(op);
        return true;
      })
      .Case<triton::CatOp>([this](auto op) {
        processCatOp(op);
        return true;
      })
      .Case<triton::ReshapeOp>([this](auto op) {
        processReshapeOp(op);
        return true;
      })
      .Case<triton::JoinOp>([this](auto op) {
        processJoinOp(op);
        return true;
      })
      .Case<triton::SplitOp>([this](auto op) {
        processSplitOp(op);
        return true;
      })
      .Case<tensor::ExpandShapeOp>([this](auto op) {
        processReshapeOp(op);
        return true;
      })
      .Case<tensor::CollapseShapeOp>([this](auto op) {
        processReshapeOp(op);
        return true;
      })
      .Case<scf::YieldOp>([this](auto op) {
        auto *parentOp = op->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          for (const auto &[regionArg, initArg] :
               llvm::zip_equal(forOp.getRegionIterArgs(),
                               forOp.getInitArgs())) {
            createDummyRefIfNotExist({regionArg, initArg});
            processValue(regionArg, initArg);
          }
          for (const auto &[resultOpr, yieldOpr] :
               llvm::zip_equal(forOp->getResults(),
                               forOp.getYieldedValues())) {
            createDummyRefIfNotExist({resultOpr, yieldOpr});
            processValue(resultOpr, yieldOpr);
          }
        } else {
          for (auto [parentResult, yieldOpResult] :
               llvm::zip_equal(parentOp->getResults(),
                               op.getOperands())) {
            if (isa<ShapedType>(parentResult.getType())) {
              SmallVector<Value> inputs{parentResult};
              SmallVector<Value> outputs{yieldOpResult};
              mergeValues(inputs, outputs);
            }
          }
        }
        return true;
      })
      .Case<scf::ForOp>([this](auto op) {
        for (const auto &[regionArg, initArg] :
             llvm::zip_equal(op.getRegionIterArgs(), op.getInitArgs())) {
          createDummyRefIfNotExist({regionArg, initArg});
          processValue(regionArg, initArg);
        }
        return true;
      })
      .Case<tensor::ConcatOp>([this](auto op) {
        processConcatOp(op);
        return true;
      })
      .Case<tensor::PadOp>([this](auto op) {
        processPadOp(op);
        return true;
      })
      .Case<tensor::ExtractSliceOp>([this](auto op) {
        processExtractSliceOp(op);
        return true;
      })
      .Case<tensor::InsertSliceOp>([this](auto op) {
        processInsertSliceOp(op);
        return true;
      })
      .Case<tensor::InsertOp>([this](auto op) {
        processInsertOp(op);
        return true;
      })
      .Case<scf::IfOp>([this](auto op) {
        processIfOp(op);
        return true;
      })
      .Default([&](Operation *op) {
        if (isElemwiseNaryOp(op) ||
            isa_and_nonnull<CopyOpInterface>(op)) {
          processParallelOp(op, current);
          return true;
        }
        processParallelOp(op, current);
        return true;
      });
}

void DimensionAnalyzer::processParallelOp(Operation *op, Value current) {
  LDBG("Processing parallel op " << *op);
  createDummyRefIfNotExist({current});
  for (Value v : op->getOperands())
    processValue(v, current);
  for (Value v : op->getResults())
    processValue(v, current);
}

void DimensionAnalyzer::processValue(Value v, Value current) {
  if (v == current)
    return;
  auto vRank = getShapeRankFromType(v.getType()).value_or(0);
  auto currentRank = getShapeRankFromType(current.getType()).value_or(0);
  if (vRank != currentRank)
    return;
  collapsePropagateOrVerify(v, current);
}

void DimensionAnalyzer::mergeValues(ArrayRef<Value> inputs,
                                    ArrayRef<Value> outputs,
                                    ArrayRef<int64_t> mutatedDims,
                                    bool mergeMutation) {
  if (outputs.empty())
    return;
  auto outputShape = getShape(outputs[0].getType());
  auto rank = outputShape.size();

  createDummyRefIfNotExist(inputs);
  createDummyRefIfNotExist(outputs);

  auto outputArgs = getArgumentRef(outputs[0]);
  for (auto input : inputs) {
    auto inputArgs = getArgumentRef(input);
    DenseSet<int64_t> mutatedSet(mutatedDims.begin(), mutatedDims.end());
    for (unsigned i = 0; i < rank; ++i) {
      if (mutatedSet.contains(i)) {
        isConnected_[outputArgs[i]].elementKind =
            ConnectedLeftRight::ElementKind::HasMutation;
        if (mergeMutation)
          joinCollapser(outputArgs[i], inputArgs[i]);
      } else {
        joinShape(outputArgs[i], inputArgs[i]);
      }
    }
  }
  for (auto output : drop_begin(outputs))
    processValue(outputs[0], output);
}

void DimensionAnalyzer::processSplatOp(triton::SplatOp op) {
  LDBG("Processing SplatOp " << op);
  Value src = op.getSrc();
  Value result = op.getResult();

  createDummyRefIfNotExist({src, result});
  auto srcType = dyn_cast<ShapedType>(src.getType());
  auto resultType = dyn_cast<ShapedType>(result.getType());
  if (!resultType)
    return;

  if (!srcType || srcType.getRank() == 0) {
    return;
  }

  SmallVector<Value> inputs{src};
  SmallVector<Value> outputs{result};
  mergeValues(inputs, outputs);
}

void DimensionAnalyzer::processBroadcastOp(triton::BroadcastOp op) {
  LDBG("Processing BroadcastOp " << op);
  Value src = op.getSrc();
  Value result = op.getResult();

  createDummyRefIfNotExist({src, result});
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  auto dstType = dyn_cast<RankedTensorType>(result.getType());
  if (!srcType || !dstType)
    return;

  auto broadcastDims = getBroadcastDims(srcType, dstType);
  auto dstArgs = getArgumentRef(result);
  for (int64_t dim : broadcastDims) {
    if (dim >= 0 && dim < static_cast<int64_t>(dstArgs.size()))
      broadcastAxisDimRefs.insert(dstArgs[dim]);
  }

  SmallVector<Value> inputs{src};
  SmallVector<Value> outputs{result};
  mergeValues(inputs, outputs, broadcastDims);
}

void DimensionAnalyzer::processExpandDimsOp(triton::ExpandDimsOp op) {
  LDBG("Processing ExpandDimsOp " << op);
  Value src = op.getSrc();
  Value result = op.getResult();
  int64_t axis = op.getAxis();

  createDummyRefIfNotExist({src, result});
  auto srcArgs = getArgumentRef(src);
  auto resultArgs = getArgumentRef(result);

  int64_t srcIdx = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(resultArgs.size()); ++i) {
    if (i == axis) {
      isConnected_[resultArgs[i]].elementKind =
          ConnectedLeftRight::ElementKind::Unit;
      continue;
    }
    joinCollapser(resultArgs[i], srcArgs[srcIdx]);
    srcIdx++;
  }
}

void DimensionAnalyzer::processTransOp(triton::TransOp op) {
  LDBG("Processing TransOp " << op);
  Value src = op.getSrc();
  Value result = op.getResult();
  auto order = op.getOrder();

  createDummyRefIfNotExist({src, result});
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resultArgs = getArgumentRefOrCreateDummy(result);

  SmallVector<int64_t> perm(order.size());
  for (size_t i = 0; i < order.size(); ++i)
    perm[i] = order[i];

  for (int i = 0; i < static_cast<int>(srcArgs.size()); ++i)
    joinCollapser(resultArgs[i], srcArgs[perm[i]]);
}

void DimensionAnalyzer::processReduceOp(triton::ReduceOp op) {
  LDBG("Processing ReduceOp " << op);
  SmallVector<Value> inputs(op.getOperands().begin(),
                             op.getOperands().end());
  SmallVector<Value> outputs(op.getResults().begin(),
                             op.getResults().end());

  SmallVector<int64_t> reduceDims = {op.getAxis()};
  createDummyRefIfNotExist(inputs);
  if (inputs.empty())
    return;
  for (auto [idx, output] : llvm::enumerate(outputs)) {
    if (!isa<ShapedType>(output.getType()))
      continue;
    Value input = inputs[std::min<size_t>(idx, inputs.size() - 1)];
    auto inputArgs = getArgumentRef(input);
    auto refIdx = processDecreasingDimensions(inputArgs, reduceDims, output);
    initCollapseOrVerify(output, refIdx);
  }
}

void DimensionAnalyzer::processDotOp(triton::DotOp op) {
  LDBG("Processing DotOp " << op);
  processMatmulOp(op.getOperation());
}

void DimensionAnalyzer::processDotScaledOp(triton::DotScaledOp op) {
  LDBG("Processing DotScaledOp " << op);
  processMatmulOp(op.getOperation());
}

void DimensionAnalyzer::processCatOp(triton::CatOp op) {
  LDBG("Processing CatOp " << op);
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value result = op.getResult();

  createDummyRefIfNotExist({lhs, rhs, result});
  auto lhsArgs = getArgumentRef(lhs);
  auto rhsArgs = getArgumentRef(rhs);
  auto resultArgs = getArgumentRef(result);
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType)
    return;

  auto resultShape = resultType.getShape();
  auto rank = resultShape.size();

  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType)
    return;

  int64_t catDim = resultType.getRank() - 1;

  if (lhsType.getRank() == resultType.getRank()) {
    auto lhsShape = lhsType.getShape();
    for (unsigned i = 0; i < rank; ++i) {
      if (i != catDim) {
        joinShape(resultArgs[i], lhsArgs[i]);
      } else {
        joinCollapser(resultArgs[i], lhsArgs[i]);
        isConnected_[resultArgs[i]].elementKind =
            ConnectedLeftRight::ElementKind::HasMutation;
      }
    }
  }

  if (rhsType.getRank() == resultType.getRank()) {
    auto rhsShape = rhsType.getShape();
    for (unsigned i = 0; i < rank; ++i) {
      if (i != catDim) {
        joinShape(resultArgs[i], rhsArgs[i]);
      } else {
        joinCollapser(resultArgs[i], rhsArgs[i]);
        isConnected_[resultArgs[i]].elementKind =
            ConnectedLeftRight::ElementKind::HasMutation;
      }
    }
  }
}

void DimensionAnalyzer::processReshapeOp(triton::ReshapeOp op) {
  LDBG("Processing ReshapeOp " << op);
  Value src = op.getSrc();
  Value result = op.getResult();

  createDummyRefIfNotExist({src, result});
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resultArgs = getArgumentRefOrCreateDummy(result);

  auto srcShape = getShape(src.getType());
  auto resultShape = getShape(result.getType());

  auto srcRank = srcArgs.size();
  auto resultRank = resultArgs.size();

  auto isStaticNonUnit = [](int64_t dim) {
    return !ShapedType::isDynamic(dim) && dim != 1;
  };
  auto getNonUnitDims = [](ArrayRef<int64_t> shape, size_t begin, size_t end) {
    SmallVector<size_t> dims;
    for (size_t i = begin; i < end; ++i) {
      if (shape[i] != 1)
        dims.push_back(i);
    }
    return dims;
  };
  auto markMutated = [&](ArrayRef<int64_t> args, ArrayRef<size_t> dims) {
    for (size_t dim : dims)
      isConnected_[args[dim]].elementKind =
          ConnectedLeftRight::ElementKind::HasMutation;
  };

  size_t srcBegin = 0;
  size_t resultBegin = 0;
  while (srcBegin < srcRank || resultBegin < resultRank) {
    if (srcBegin < srcRank && resultBegin < resultRank &&
        srcShape[srcBegin] == resultShape[resultBegin]) {
      joinShape(srcArgs[srcBegin], resultArgs[resultBegin]);
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
        if (ShapedType::isDynamic(dim)) {
          hasDynamicShape = true;
        } else {
          srcProduct *= dim;
        }
      } else if (resultEnd < resultRank) {
        int64_t dim = resultShape[resultEnd++];
        if (ShapedType::isDynamic(dim)) {
          hasDynamicShape = true;
        } else {
          resultProduct *= dim;
        }
      }
    } while (!hasDynamicShape && srcProduct != resultProduct &&
             (srcEnd < srcRank || resultEnd < resultRank));

    if (hasDynamicShape || srcProduct != resultProduct) {
      LDBG("Skipping reshape group with dynamic or unmatched product");
      return;
    }

    auto srcNonUnitDims = getNonUnitDims(srcShape, srcBegin, srcEnd);
    auto resultNonUnitDims =
        getNonUnitDims(resultShape, resultBegin, resultEnd);

    if (srcNonUnitDims.size() == 1 && resultNonUnitDims.size() == 1) {
      size_t srcDim = srcNonUnitDims.front();
      size_t resultDim = resultNonUnitDims.front();
      if (srcShape[srcDim] == resultShape[resultDim]) {
        joinShape(srcArgs[srcDim], resultArgs[resultDim]);
      } else if (isStaticNonUnit(srcShape[srcDim]) &&
                 isStaticNonUnit(resultShape[resultDim])) {
        joinCollapser(srcArgs[srcDim], resultArgs[resultDim]);
        markMutated(srcArgs, srcNonUnitDims);
        markMutated(resultArgs, resultNonUnitDims);
      }
    } else if (!srcNonUnitDims.empty() && !resultNonUnitDims.empty()) {
      joinCollapser(srcArgs[srcNonUnitDims.front()],
                    resultArgs[resultNonUnitDims.front()]);
      markMutated(srcArgs, srcNonUnitDims);
      markMutated(resultArgs, resultNonUnitDims);
    }

    srcBegin = srcEnd;
    resultBegin = resultEnd;
  }
}

void DimensionAnalyzer::processJoinOp(triton::JoinOp op) {
  LDBG("Processing JoinOp " << op);
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value result = op.getResult();

  createDummyRefIfNotExist({lhs, rhs, result});
  auto resultArgs = getArgumentRef(result);
  auto lhsArgs = getArgumentRef(lhs);
  auto rhsArgs = getArgumentRef(rhs);

  for (size_t i = 0; i < lhsArgs.size(); ++i)
    joinShape(resultArgs[i], lhsArgs[i]);

  if (resultArgs.size() == lhsArgs.size() + 1) {
    isConnected_[resultArgs.back()].elementKind =
        ConnectedLeftRight::ElementKind::Unit;
  }
}

void DimensionAnalyzer::processSplitOp(triton::SplitOp op) {
  LDBG("Processing SplitOp " << op);
  Value src = op.getSrc();
  SmallVector<Value> resultsVec(op.getResults().begin(),
                                op.getResults().end());

  createDummyRefIfNotExist({src});
  createDummyRefIfNotExist(resultsVec);

  auto srcArgs = getArgumentRef(src);

  for (auto result : resultsVec) {
    auto resultArgs = getArgumentRef(result);
    for (size_t i = 0; i < resultArgs.size() && i < srcArgs.size() - 1; ++i)
      joinShape(resultArgs[i], srcArgs[i]);
  }
}

template <typename T, typename>
void DimensionAnalyzer::processReshapeOp(T op) {
  if constexpr (std::is_same_v<T, tensor::ExpandShapeOp>) {
    LDBG("Processing ExpandShapeOp " << op);
  } else {
    LDBG("Processing CollapseShapeOp " << op);
  }
  auto input = op.getSrc();
  auto output = op.getResult();
  auto inputArgs = getArgumentRefOrCreateDummy(input);
  auto outputArgs = getArgumentRefOrCreateDummy(output);
  auto inputShape = getShape(input.getType());
  auto outputShape = getShape(output.getType());

  auto reassoc = op.getReassociationIndices();

  if constexpr (std::is_same_v<T, tensor::ExpandShapeOp>) {
    for (auto [inputIdx, indices] : llvm::enumerate(reassoc)) {
      if (indices.size() == 1) {
        joinCollapser(inputArgs[inputIdx], outputArgs[indices[0]]);
        continue;
      }
      for (auto idx : indices)
        isConnected_[outputArgs[idx]].elementKind =
            ConnectedLeftRight::ElementKind::HasMutation;

      SmallVector<int64_t> inputIdxVec{static_cast<int64_t>(inputIdx)};
      auto filteredInputIdx = to_vector(make_filter_range(
          inputIdxVec, [&inputShape](int64_t idx) {
            return inputShape[idx] != 1;
          }));
      auto filteredOutputIdx = to_vector(make_filter_range(
          indices, [&outputShape](int64_t idx) {
            return outputShape[idx] != 1;
          }));

      if (filteredOutputIdx.size() == 1) {
        isConnected_[outputArgs[filteredOutputIdx[0]]].elementKind =
            ConnectedLeftRight::ElementKind::Unit;
        joinCollapser(outputArgs[filteredOutputIdx[0]],
                      inputArgs[inputIdx]);
      }
    }
  } else {
    for (auto [outputIdx, indices] : llvm::enumerate(reassoc)) {
      if (indices.size() == 1) {
        joinCollapser(inputArgs[indices[0]], outputArgs[outputIdx]);
        continue;
      }
      for (auto idx : indices)
        isConnected_[inputArgs[idx]].elementKind =
            ConnectedLeftRight::ElementKind::HasMutation;

      SmallVector<int64_t> outputIdxVec{static_cast<int64_t>(outputIdx)};
      auto filteredOutputIdx = to_vector(make_filter_range(
          outputIdxVec, [&outputShape](int64_t idx) {
            return outputShape[idx] != 1;
          }));
      auto filteredInputIdx = to_vector(make_filter_range(
          indices, [&inputShape](int64_t idx) {
            return inputShape[idx] != 1;
          }));

      if (filteredInputIdx.size() == 1) {
        isConnected_[inputArgs[filteredInputIdx[0]]].elementKind =
            ConnectedLeftRight::ElementKind::Unit;
        joinCollapser(inputArgs[filteredInputIdx[0]],
                      outputArgs[outputIdx]);
      }
    }
  }
}

void DimensionAnalyzer::processConcatOp(tensor::ConcatOp concatOp) {
  LDBG("Processing ConcatOp " << concatOp);
  auto dim = concatOp.getDim();
  auto res = concatOp.getResult();
  auto resultShape = getShape(res.getType());
  auto rank = resultShape.size();

  for (auto opr : concatOp.getOperands()) {
    if (getShapeRankFromType(opr.getType()).value_or(0) !=
        resultShape.size())
      continue;
    auto oprArgRef = getArgumentRefOrCreateDummy(opr);
    auto argConcat = getArgumentRefOrCreateDummy(res);
    for (unsigned i = 0; i < rank; ++i) {
      if (i != dim) {
        isConnected_[argConcat[i]].elementKind =
            (resultShape[i] == 1 ? ConnectedLeftRight::ElementKind::Unit
                                  : ConnectedLeftRight::ElementKind::NoMutation);
        joinShape(argConcat[i], oprArgRef[i]);
      } else {
        isConnected_[argConcat[i]].elementKind =
            ConnectedLeftRight::ElementKind::HasMutation;
        joinCollapser(argConcat[i], oprArgRef[i]);
      }
    }
  }
}

void DimensionAnalyzer::processPadOp(tensor::PadOp padOp) {
  auto padSrc = padOp.getSource();
  auto padHigh = padOp.getStaticHigh();
  auto padLow = padOp.getStaticLow();
  auto padResult = padOp.getResult();
  auto rank = padOp.getType().getRank();
  auto argPadSrc = getArgumentRefOrCreateDummy(padSrc);
  auto argPadResult = getArgumentRefOrCreateDummy(padResult);
  auto srcShape = getShape(padSrc.getType());
  for (int i = 0; i < rank; i++) {
    if (padHigh[i] == 0 && padLow[i] == 0) {
      joinShape(argPadSrc[i], argPadResult[i]);
      isConnected_[argPadSrc[i]].elementKind =
          srcShape[i] == 1 ? ConnectedLeftRight::ElementKind::Unit
                            : ConnectedLeftRight::ElementKind::NoMutation;
    } else {
      joinCollapser(argPadSrc[i], argPadResult[i]);
      isConnected_[argPadSrc[i]].elementKind =
          ConnectedLeftRight::ElementKind::HasMutation;
    }
  }
}

template <class T, typename>
void DimensionAnalyzer::processSlicingOp(T slicingOp) {
  auto src = slicingOp.getSource();
  auto res = slicingOp.getResult();
  Value superview, subview;
  if (std::is_same_v<T, tensor::ExtractSliceOp>) {
    superview = src;
    subview = res;
  } else {
    superview = res;
    subview = src;
  }

  SmallVector<OpFoldResult> superviewShape;
  if (auto expandOp =
          superview.template getDefiningOp<tensor::ExpandShapeOp>()) {
    superviewShape = llvm::map_to_vector(
        getShape(expandOp.getResult().getType()),
        [&slicingOp](int64_t elementShape) -> OpFoldResult {
          return getAsIndexOpFoldResult(slicingOp.getContext(), elementShape);
        });
  } else {
    superviewShape = llvm::map_to_vector(
        getShape(superview.getType()),
        [&slicingOp](int64_t elementShape) -> OpFoldResult {
          return getAsIndexOpFoldResult(slicingOp.getContext(), elementShape);
        });
  }

  auto subviewShape = slicingOp.getMixedSizes();
  auto subviewStride = slicingOp.getMixedStrides();
  auto droppedDims = slicingOp.getDroppedDims();

  auto rank = superviewShape.size();
  auto superviewRef = getArgumentRef(superview);
  auto subviewRef = getArgumentRef(subview);

  SmallVector<int64_t> subviewShapeSize(superviewShape.size());
  SmallVector<int64_t> subviewStrides(superviewShape.size());
  int64_t subviewAxis = 0;
  for (size_t axis = 0; axis < rank; ++axis) {
    auto subviewStrideVal =
        getConstantIntValue(subviewStride[axis])
            .value_or(ShapedType::kDynamic);
    auto subviewShapeVal =
        getConstantIntValue(subviewShape[axis])
            .value_or(ShapedType::kDynamic);
    subviewStrides[axis] = subviewStrideVal;
    subviewShapeSize[axis] = subviewShapeVal;
    if (droppedDims[axis])
      continue;
    joinCollapser(superviewRef[axis], subviewRef[subviewAxis]);
    subviewAxis++;
  }

  if (rank <= 1)
    return;

  auto getContiguousAxes = [&]() -> BitVector {
    BitVector ret;
    ret.resize(rank, true);
    for (size_t axis = 1; axis < rank; ++axis) {
      int64_t staticRes;
      if (droppedDims[axis]) {
        staticRes = 1;
      } else {
        staticRes = getConstantIntValue(subviewShape[axis])
                        .value_or(ShapedType::kDynamic);
      }
      auto staticSrc = getConstantIntValue(superviewShape[axis])
                           .value_or(ShapedType::kDynamic);
      if (ShapedType::isDynamic(subviewStrides[axis]) ||
          ShapedType::isDynamic(subviewStrides[axis - 1]) ||
          ShapedType::isDynamic(staticRes) ||
          ShapedType::isDynamic(staticSrc)) {
        ret[axis] = false;
        continue;
      }
      if (!ret.empty())
        ret[0] = true;
      if (staticRes * subviewStrides[axis] !=
          staticSrc * subviewStrides[axis - 1])
        ret[axis] = false;
    }
    return ret;
  };

  auto contiguousMask = getContiguousAxes();
  separateGroup(superview, contiguousMask, subviewShapeSize);
}

void DimensionAnalyzer::processExtractSliceOp(
    tensor::ExtractSliceOp extractSliceOp) {
  auto res = extractSliceOp.getResult();
  auto src = extractSliceOp.getSource();
  createDummyRefIfNotExist({res, src});
  processSlicingOp(extractSliceOp);
}

void DimensionAnalyzer::processInsertSliceOp(
    tensor::InsertSliceOp insertSliceOp) {
  auto dest = insertSliceOp.getDest();
  auto res = insertSliceOp.getResult();
  auto src = insertSliceOp.getSource();
  createDummyRefIfNotExist({res, dest, src});
  assert(res.getType().getRank() == dest.getType().getRank());
  collapsePropagateOrVerify(res, dest);
  processSlicingOp(insertSliceOp);
}

void DimensionAnalyzer::processInsertOp(tensor::InsertOp insertOp) {
  auto dest = insertOp.getDest();
  auto res = insertOp.getResult();
  createDummyRefIfNotExist({res, dest});
  assert(res.getType().getRank() == dest.getType().getRank());
  collapsePropagateOrVerify(res, dest);
}

void DimensionAnalyzer::processIfOp(scf::IfOp op) {
  LDBG("Processing IfOp " << op);
  scf::YieldOp thenYield = op.thenYield();
  for (auto [ifResult, thenOperand] :
       llvm::zip_equal(op.getResults(), thenYield.getOperands())) {
    createDummyRefIfNotExist({thenOperand});
    processValue(ifResult, thenOperand);
  }
  if (!op.elseBlock())
    return;
  scf::YieldOp elseYield = op.elseYield();
  for (auto [ifResult, elseOperand] :
       llvm::zip_equal(op.getResults(), elseYield.getOperands())) {
    createDummyRefIfNotExist({elseOperand});
    processValue(ifResult, elseOperand);
  }
}

void DimensionAnalyzer::processForOp(scf::ForOp op) {
  LDBG("Processing ForOp " << op);
  for (const auto &[regionArg, initArg] :
       llvm::zip_equal(op.getRegionIterArgs(), op.getInitArgs())) {
    createDummyRefIfNotExist({regionArg, initArg});
    processValue(regionArg, initArg);
  }
  for (const auto &[resultOpr, yieldOpr] :
       llvm::zip_equal(op->getResults(), op.getYieldedValues())) {
    createDummyRefIfNotExist({resultOpr, yieldOpr});
    processValue(resultOpr, yieldOpr);
  }
}

size_t DimensionAnalyzer::processDecreasingDimensions(
    ArrayRef<int64_t> inputArgs, ArrayRef<int64_t> dimensions,
    const Value &output) {
  size_t outputRank = getShapeRankFromType(output.getType()).value_or(0);
  assert(inputArgs.size() == outputRank + dimensions.size());

  DimensionIndex outputArgs;
  outputArgs.reserve(outputRank);
  SmallVector<int64_t> sortedDimensions = {dimensions.begin(),
                                            dimensions.end()};
  llvm::sort(sortedDimensions);

  int64_t inputRank = static_cast<int64_t>(inputArgs.size());
  std::vector<bool> isReduced(inputRank, false);
  for (int64_t dim : dimensions) {
    if (dim >= 0 && dim < inputRank)
      isReduced[dim] = true;
  }

  for (int64_t i = 0; i < inputRank - 1; ++i) {
    if (isReduced[i] != isReduced[i + 1])
      disconnect(inputArgs[i], inputArgs[i + 1]);
  }

  const auto *dimPtr = sortedDimensions.begin();
  for (int64_t i = 0; i < static_cast<int64_t>(inputArgs.size()); ++i) {
    if (dimPtr != sortedDimensions.end() && *dimPtr == i) {
      ++dimPtr;
    } else {
      outputArgs.push_back(inputArgs[i]);
    }
  }

  assert(outputArgs.size() == outputRank);
  argumentsRef_.push_back(std::move(outputArgs));
  return argumentsRef_.size() - 1;
}

size_t DimensionAnalyzer::processPermutation(ArrayRef<int64_t> inputArgs,
                                              ArrayRef<int64_t> perm,
                                              const Value &output) {
  auto shapeInfo =
      getValueShapeInfo(output).value_or(std::make_pair(0, DimensionIndex{}));
  auto outputRank = shapeInfo.first;

  DimensionIndex outputArgs(outputRank, -1);
  for (int i = 0; i < static_cast<int>(inputArgs.size()); ++i)
    outputArgs[i] = inputArgs[perm[i]];

  argumentsRef_.push_back(std::move(outputArgs));

  std::vector<int> invPerm(perm.size());
  for (int i = 0; i < static_cast<int>(perm.size()); ++i)
    invPerm[perm[i]] = i;

  for (int j = 0; j < static_cast<int>(inputArgs.size()) - 1; ++j) {
    int posCurr = invPerm[j];
    int posNext = invPerm[j + 1];
    if (posNext != posCurr + 1)
      disconnect(inputArgs[j], inputArgs[j + 1]);
  }
  return argumentsRef_.size() - 1;
}

void DimensionAnalyzer::processMatmulOp(Operation *op, bool isTransposeA,
                                        bool isTransposeB) {
  auto matmulOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!matmulOp)
    return;

  auto inputs = matmulOp.getDpsInputs();
  assert(matmulOp.getDpsInits().size() == 1);
  Value output = matmulOp.getDpsInits()[0];

  auto arg0 = getArgumentRefOrCreateDummy(inputs[0]);
  auto arg1 = getArgumentRefOrCreateDummy(inputs[1]);

  int mDimIdx = isTransposeA ? 1 : 0;
  int nDimIdx = isTransposeB ? 0 : 1;

  auto aType = dyn_cast<ShapedType>(inputs[0].getType());
  auto bType = dyn_cast<ShapedType>(inputs[1].getType());

  if (aType && aType.getRank() == 3 && bType &&
      bType.getRank() == 3) {
    argumentsRef_.push_back(
        {arg0[0], arg0[mDimIdx], arg1[nDimIdx]});
    disconnect(arg0[0], arg0[1]);
    disconnect(arg1[0], arg1[1]);
  } else {
    argumentsRef_.push_back(
        {arg0[mDimIdx], arg1[nDimIdx]});
  }

  initCollapseOrVerify(output,
                       static_cast<int64_t>(argumentsRef_.size() - 1));
  for (Value result : op->getResults())
    processValue(result, output);
}

void DimensionAnalyzer::processBatchMatmulOp(
    linalg::BatchMatmulOp batchMatmulOp) {
  auto inputs = batchMatmulOp.getDpsInputs();
  assert(batchMatmulOp.getDpsInits().size() == 1);
  Value output = batchMatmulOp.getDpsInits()[0];

  auto arg0 = getArgumentRefOrCreateDummy(inputs[0]);
  auto arg1 = getArgumentRefOrCreateDummy(inputs[1]);
  argumentsRef_.push_back({arg0[0], arg0[1], arg1[2]});

  initCollapseOrVerify(output,
                       static_cast<int64_t>(argumentsRef_.size() - 1));
  for (Value result : batchMatmulOp->getResults())
    processValue(result, output);

  disconnect(arg0[0], arg0[1]);
  disconnect(arg1[0], arg1[1]);
}

void DimensionAnalyzer::combineInferable() {
  for (const auto &arg : argumentList_) {
    auto emptyOp = dyn_cast_if_present<tensor::EmptyOp>(arg.getDefiningOp());
    if (!emptyOp)
      continue;

    auto emptyRef = getArgumentRef(emptyOp.getResult());
    auto mixEmptyShape = emptyOp.getMixedSizes();

    for (auto [emptyIdx, el] : llvm::enumerate(mixEmptyShape)) {
      if (getConstantIntValue(el).has_value())
        continue;
      auto sizeDefiner = cast<Value>(el);
      auto dimOp =
          dyn_cast_if_present<tensor::DimOp>(sizeDefiner.getDefiningOp());
      if (!dimOp)
        continue;
      auto constantIndex = dimOp.getConstantIndex();
      if (!constantIndex.has_value() || !bindUsingTensorDim)
        continue;
      auto tensorSource = dimOp.getSource();
      auto tensorRef = getArgumentRefOrCreateDummy(tensorSource);
      joinShape(tensorRef[constantIndex.value()], emptyRef[emptyIdx]);
    }
  }
}

void DimensionAnalyzer::unifyGroups() {
  for (int ref = 0; ref < static_cast<int>(argumentsRef_.size()); ++ref) {
    auto parIdx = solverSegments_->find(ref);
    if (parIdx == ref)
      continue;

    const auto &par = argumentsRef_[parIdx];
    assert(par.size() == argumentsRef_[ref].size());
    for (const auto &[idx, u] : llvm::enumerate(argumentsRef_[ref]))
      joinShape(u, par[idx]);
  }
}

void DimensionAnalyzer::propagateConnection() {
  for (int i = 0; i < argumentTotalLength_; ++i) {
    int parent = solverCollapserElem_->find(i);
    isConnected_[parent].leftConnected =
        isConnected_[parent].leftConnected &&
        isConnected_[i].leftConnected;
    isConnected_[parent].rightConnected =
        isConnected_[parent].rightConnected &&
        isConnected_[i].rightConnected;
    isConnected_[parent].elementKind =
        std::min(isConnected_[parent].elementKind,
                 isConnected_[i].elementKind);
  }
}

void DimensionAnalyzer::spreadConnection() {
  for (int i = 0; i < argumentTotalLength_; ++i) {
    isConnected_[i] = isConnected_[solverCollapserElem_->find(i)];
  }
}

void DimensionAnalyzer::joinShape(int a, int b) {
  LDBG("Joining shape bind " << a << " " << b);
  solverShapeElem_->join(a, b);
  solverCollapserElem_->join(a, b);
}

void DimensionAnalyzer::joinCollapser(int a, int b) {
  LDBG("Joining collapser bind " << a << " " << b);
  solverCollapserElem_->join(a, b);
}

void DimensionAnalyzer::disconnect(int a, int b) {
  LDBG("Disconnecting " << a << " " << b);
  if (0 <= a && a < static_cast<int>(isConnected_.size())) {
    isConnected_[a].rightConnected = false;
    int parentOfA = solverCollapserElem_->find(a);
    isConnected_[parentOfA].rightConnected = false;
  }
  if (0 <= b && b < static_cast<int>(isConnected_.size())) {
    isConnected_[b].leftConnected = false;
    int parentOfB = solverCollapserElem_->find(b);
    isConnected_[parentOfB].leftConnected = false;
  }
}

bool DimensionAnalyzer::isConnected(int a, int b) {
  if (a + 1 != b)
    return false;
  return isConnected_[a].rightConnected && isConnected_[b].leftConnected;
}

void DimensionAnalyzer::separateGroup(Value val, BitVector contiguousMask,
                                      ArrayRef<int64_t> shape) {
  auto argRef = getArgumentRef(val);
  size_t rank = argRef.size();
  if (rank <= 1)
    return;

  struct Group {
    bool isAllContiguous;
    bool isAllUnit;
    int leftIndex;
    int rightIndex;
  };

  SmallVector<Group> groups;
  for (size_t i = 0; i < rank; ++i)
    groups.push_back({true, (shape[i] == 1),
                      static_cast<int>(i), static_cast<int>(i)});

  auto mergeGroups = [&contiguousMask](const Group &left,
                                       const Group &right) -> Group {
    bool boundary = contiguousMask[right.leftIndex];
    return {left.isAllContiguous && right.isAllContiguous && boundary,
            left.isAllUnit && right.isAllUnit,
            left.leftIndex, right.rightIndex};
  };

  SmallVector<Group> contiguousGroups;
  contiguousGroups.push_back(groups[0]);
  for (size_t i = 1; i < rank; ++i) {
    Group &lastGroup = contiguousGroups.back();
    bool boundary = contiguousMask[groups[i].leftIndex];
    if (boundary)
      lastGroup = mergeGroups(lastGroup, groups[i]);
    else
      contiguousGroups.push_back(groups[i]);
  }

  auto canConnect = [&argRef, this](const Group &left,
                                    const Group &right) -> bool {
    if (!isConnected(argRef[left.rightIndex],
                     argRef[right.leftIndex]))
      return false;
    if (left.isAllUnit || right.isAllUnit)
      return true;
    return false;
  };

  SmallVector<Group> mergedGroups;
  mergedGroups.push_back(contiguousGroups[0]);
  for (size_t i = 1; i < contiguousGroups.size(); ++i) {
    Group &lastGroup = mergedGroups.back();
    if (canConnect(lastGroup, contiguousGroups[i]))
      lastGroup = mergeGroups(lastGroup, contiguousGroups[i]);
    else
      mergedGroups.push_back(contiguousGroups[i]);
  }

  for (size_t g = 1; g < mergedGroups.size(); ++g) {
    disconnect(argRef[mergedGroups[g - 1].rightIndex],
               argRef[mergedGroups[g].leftIndex]);
  }
}

void DimensionAnalyzer::markDimensions() {
  auto processSlice = [this](auto sliceOp) {
    if (!argumentsRefPointer_.contains(sliceOp.getSource()))
      return;
    llvm::SmallBitVector droppedDimsMask = sliceOp.getDroppedDims();
    auto origType = dyn_cast<ShapedType>(sliceOp.getSource().getType());
    auto sliceType = dyn_cast<ShapedType>(sliceOp.getResult().getType());
    auto origRef = getArgumentRefOrCreateDummy(sliceOp.getSource());
    auto sliceRef = getArgumentRefOrCreateDummy(sliceOp.getResult());
    if (isa<tensor::InsertSliceOp>(sliceOp.getOperation())) {
      std::swap(origRef, sliceRef);
      std::swap(origType, sliceType);
    }
    size_t sliceIdx = 0;
    for (size_t i = 0; i < origRef.size(); ++i) {
      if (droppedDimsMask[i]) {
        tilingDimKindMapForCollapser[solverCollapserElem_->find(origRef[i])] =
            TilingDimensionKind::RankReduced;
        tilingDimKindMapForShape[solverShapeElem_->find(origRef[i])] =
            TilingDimensionKind::RankReduced;
      } else {
        if (isa<tensor::InsertSliceOp>(sliceOp.getOperation()) &&
            sliceType && sliceType.getDimSize(sliceIdx) == 1) {
          tilingDimKindMapForCollapser[solverCollapserElem_->find(origRef[i])] =
              TilingDimensionKind::Reduce;
          tilingDimKindMapForShape[solverShapeElem_->find(origRef[i])] =
              TilingDimensionKind::Reduce;
        }
        sliceIdx++;
      }
    }
  };

  op_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      auto reduceSrcRef = getArgumentRef(reduceOp.getOperands()[0]);
      if (reduceSrcRef.empty())
        return;
      int64_t reduceAxis = reduceOp.getAxis();
      if (reduceAxis >= static_cast<int64_t>(reduceSrcRef.size()))
        return;
      tilingDimKindMapForCollapser[solverCollapserElem_->find(
          reduceSrcRef[reduceAxis])] = TilingDimensionKind::Reduce;
      tilingDimKindMapForShape[solverShapeElem_->find(
          reduceSrcRef[reduceAxis])] = TilingDimensionKind::Reduce;
    } else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      processSlice(insertOp);
    } else if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      processSlice(extractOp);
    } else if (auto transOp = dyn_cast<triton::TransOp>(op)) {
      auto src = transOp.getSrc();
      auto result = transOp.getResult();
      auto srcType = dyn_cast<RankedTensorType>(src.getType());
      auto dstType = dyn_cast<RankedTensorType>(result.getType());
      if (!srcType || !dstType)
        return;
      auto order = transOp.getOrder();
      auto srcArgs = getArgumentRef(src);
      auto dstArgs = getArgumentRef(result);
      SmallVector<int64_t> perm(order.size());
      for (size_t i = 0; i < order.size(); ++i)
        perm[i] = order[i];
      for (auto [dimIdx, parentIdx] : llvm::enumerate(dstArgs)) {
        auto srcSolverIdx =
            solverShapeElem_->find(srcArgs[perm[dimIdx]]);
        auto dstSolverIdx = solverShapeElem_->find(parentIdx);
        if (auto it = transposedDimMap.find(srcSolverIdx);
            it != transposedDimMap.end()) {
          transposedDimMap[dstSolverIdx] = it->second;
        } else {
          transposedDimMap[dstSolverIdx] = perm[dimIdx];
        }
      }
    }
  });
}

void DimensionAnalyzer::transferDimMark() {
  op_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op))
      transferDimMarkImpl(expandOp);
    else if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op))
      transferDimMarkImpl(collapseOp);
    else if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op))
      transferDimMarkImpl(extractOp);
    else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op))
      transferDimMarkImpl(insertOp);
    else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(op))
      transferDimMarkImpl(broadcastOp);
    else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(op))
      transferDimMarkImpl(expandDimsOp);
    else if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op))
      transferDimMarkImpl(reshapeOp);
    else if (auto transOp = dyn_cast<triton::TransOp>(op))
      transferDimMarkImpl(transOp);
  });
}

template <typename IntegerRange>
void DimensionAnalyzer::transferDimMarkImpl(Value input, Value output,
                                            const IntegerRange &mutated) {
  auto inputArgs = getArgumentRefOrCreateDummy(input);
  auto outputArgs = getArgumentRefOrCreateDummy(output);
  for (auto idx : mutated) {
    auto srcDim = solverShapeElem_->find(inputArgs[idx]);
    auto resDim = solverShapeElem_->find(outputArgs[idx]);
    if (auto it = transposedDimMap.find(srcDim);
        it != transposedDimMap.end())
      transposedDimMap[resDim] = it->second;
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::ExpandShapeOp op) {
  auto input = op.getSrc();
  auto output = op.getResult();
  auto outputType = op.getType();
  auto inputArgs = getArgumentRefOrCreateDummy(input);
  auto outputArgs = getArgumentRefOrCreateDummy(output);
  auto reassoc = op.getReassociationIndices();

  DenseMap<int64_t, int64_t> dimMap;
  SmallVector<std::pair<int64_t, int64_t>> toBeTransferred;
  for (auto [inputIdx, indices] : llvm::enumerate(reassoc)) {
    int64_t targetIdx = indices[0];
    for (auto outputIdx : indices) {
      if (outputType.getDimSize(targetIdx) == 1)
        targetIdx = outputIdx;
    }
    dimMap[static_cast<int64_t>(inputIdx)] = targetIdx;
    toBeTransferred.emplace_back(static_cast<int64_t>(inputIdx), targetIdx);
  }

  for (auto [inputIdx, targetIdx] : toBeTransferred) {
    auto srcDim = inputArgs[inputIdx];
    auto resDim = outputArgs[targetIdx];
    if (solverCollapserElem_->find(srcDim) !=
        solverCollapserElem_->find(resDim))
      continue;
    srcDim = solverShapeElem_->find(srcDim);
    resDim = solverShapeElem_->find(resDim);
    if (auto it = transposedDimMap.find(srcDim);
        it != transposedDimMap.end()) {
      auto mapped = dimMap.find(it->second);
      transposedDimMap[resDim] =
          mapped == dimMap.end() ? it->second : mapped->second;
    }
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::CollapseShapeOp op) {
  auto input = op.getSrc();
  auto output = op.getResult();
  auto inputType = op.getSrcType();
  auto inputArgs = getArgumentRefOrCreateDummy(input);
  auto outputArgs = getArgumentRefOrCreateDummy(output);
  auto reassoc = op.getReassociationIndices();

  DenseMap<int64_t, int64_t> dimMap;
  SmallVector<std::pair<int64_t, int64_t>> toBeTransferred;
  for (auto [outputIdx, indices] : llvm::enumerate(reassoc)) {
    int64_t targetIdx = indices[0];
    for (auto inputIdx : indices) {
      dimMap[inputIdx] = static_cast<int64_t>(outputIdx);
      if (inputType.getDimSize(targetIdx) == 1)
        targetIdx = inputIdx;
    }
    dimMap[targetIdx] = static_cast<int64_t>(outputIdx);
    toBeTransferred.emplace_back(targetIdx, static_cast<int64_t>(outputIdx));
  }

  for (auto [targetIdx, outputIdx] : toBeTransferred) {
    auto srcDim = inputArgs[targetIdx];
    auto resDim = outputArgs[outputIdx];
    if (solverCollapserElem_->find(srcDim) !=
        solverCollapserElem_->find(resDim))
      continue;
    srcDim = solverShapeElem_->find(srcDim);
    resDim = solverShapeElem_->find(resDim);
    if (auto it = transposedDimMap.find(srcDim);
        it != transposedDimMap.end()) {
      auto mapped = dimMap.find(it->second);
      transposedDimMap[resDim] =
          mapped == dimMap.end() ? it->second : mapped->second;
    }
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::ExtractSliceOp op) {
  if (op.getDroppedDims().any())
    return;
  auto src = op.getSource();
  auto res = op.getResult();
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resArgs = getArgumentRefOrCreateDummy(res);
  auto rank = srcArgs.size();
  for (size_t i = 0; i < rank; ++i) {
    auto srcDim = solverShapeElem_->find(srcArgs[i]);
    auto resDim = solverShapeElem_->find(resArgs[i]);
    if (auto it = transposedDimMap.find(srcDim);
        it != transposedDimMap.end())
      transposedDimMap[resDim] = it->second;
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::InsertSliceOp op) {
  if (op.getDroppedDims().any())
    return;
  auto src = op.getSource();
  auto res = op.getResult();
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resArgs = getArgumentRefOrCreateDummy(res);
  auto rank = srcArgs.size();
  for (size_t i = 0; i < rank; ++i) {
    auto srcDim = solverShapeElem_->find(srcArgs[i]);
    auto resDim = solverShapeElem_->find(resArgs[i]);
    if (auto it = transposedDimMap.find(srcDim);
        it != transposedDimMap.end())
      transposedDimMap[resDim] = it->second;
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

void DimensionAnalyzer::transferDimMarkImpl(triton::BroadcastOp op) {
  auto src = op.getSrc();
  auto result = op.getResult();
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  auto dstType = dyn_cast<RankedTensorType>(result.getType());
  if (!srcType || !dstType || srcType.getRank() != dstType.getRank())
    return;
  transferDimMarkImpl(src, result, getBroadcastDims(srcType, dstType));
}

void DimensionAnalyzer::transferDimMarkImpl(triton::ExpandDimsOp op) {
  auto src = op.getSrc();
  auto result = op.getResult();
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resultArgs = getArgumentRefOrCreateDummy(result);
  int64_t axis = op.getAxis();

  DenseMap<int64_t, int64_t> dimMap;
  SmallVector<std::pair<int64_t, int64_t>> toBeTransferred;
  int64_t srcIdx = 0;
  for (int64_t resultIdx = 0;
       resultIdx < static_cast<int64_t>(resultArgs.size()); ++resultIdx) {
    if (resultIdx == axis)
      continue;
    dimMap[srcIdx] = resultIdx;
    toBeTransferred.emplace_back(srcIdx, resultIdx);
    ++srcIdx;
  }

  for (auto [srcIdx, resultIdx] : toBeTransferred) {
    auto srcDim = srcArgs[srcIdx];
    auto resDim = resultArgs[resultIdx];
    if (solverCollapserElem_->find(srcDim) !=
        solverCollapserElem_->find(resDim))
      continue;
    srcDim = solverShapeElem_->find(srcDim);
    resDim = solverShapeElem_->find(resDim);
    if (auto it = transposedDimMap.find(srcDim);
        it != transposedDimMap.end()) {
      auto mapped = dimMap.find(it->second);
      transposedDimMap[resDim] =
          mapped == dimMap.end() ? it->second : mapped->second;
    }
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

void DimensionAnalyzer::transferDimMarkImpl(triton::ReshapeOp op) {
  auto src = op.getSrc();
  auto result = op.getResult();
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resultArgs = getArgumentRefOrCreateDummy(result);
  auto srcShape = getShape(src.getType());
  auto resultShape = getShape(result.getType());
  auto srcRank = srcArgs.size();
  auto resultRank = resultArgs.size();

  auto getNonUnitDims = [](ArrayRef<int64_t> shape, size_t begin,
                           size_t end) {
    SmallVector<size_t> dims;
    for (size_t i = begin; i < end; ++i) {
      if (shape[i] != 1)
        dims.push_back(i);
    }
    return dims;
  };

  DenseMap<int64_t, int64_t> dimMap;
  SmallVector<std::pair<int64_t, int64_t>> toBeTransferred;
  size_t srcBegin = 0;
  size_t resultBegin = 0;
  while (srcBegin < srcRank || resultBegin < resultRank) {
    if (srcBegin < srcRank && resultBegin < resultRank &&
        srcShape[srcBegin] == resultShape[resultBegin]) {
      dimMap[static_cast<int64_t>(srcBegin)] =
          static_cast<int64_t>(resultBegin);
      toBeTransferred.emplace_back(static_cast<int64_t>(srcBegin),
                                   static_cast<int64_t>(resultBegin));
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

    if (srcNonUnitDims.size() == 1 && resultNonUnitDims.size() == 1) {
      dimMap[static_cast<int64_t>(srcNonUnitDims.front())] =
          static_cast<int64_t>(resultNonUnitDims.front());
      toBeTransferred.emplace_back(
          static_cast<int64_t>(srcNonUnitDims.front()),
          static_cast<int64_t>(resultNonUnitDims.front()));
    } else if (!srcNonUnitDims.empty() && !resultNonUnitDims.empty()) {
      dimMap[static_cast<int64_t>(srcNonUnitDims.front())] =
          static_cast<int64_t>(resultNonUnitDims.front());
      toBeTransferred.emplace_back(
          static_cast<int64_t>(srcNonUnitDims.front()),
          static_cast<int64_t>(resultNonUnitDims.front()));
    }

    srcBegin = srcEnd;
    resultBegin = resultEnd;
  }

  for (auto [srcIdx, resultIdx] : toBeTransferred) {
    auto srcDim = srcArgs[srcIdx];
    auto resDim = resultArgs[resultIdx];
    if (solverCollapserElem_->find(srcDim) !=
        solverCollapserElem_->find(resDim))
      continue;
    srcDim = solverShapeElem_->find(srcDim);
    resDim = solverShapeElem_->find(resDim);
    if (auto it = transposedDimMap.find(srcDim);
        it != transposedDimMap.end()) {
      auto mapped = dimMap.find(it->second);
      transposedDimMap[resDim] =
          mapped == dimMap.end() ? it->second : mapped->second;
    }
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

void DimensionAnalyzer::transferDimMarkImpl(triton::TransOp op) {
  auto src = op.getSrc();
  auto result = op.getResult();
  auto order = op.getOrder();
  auto srcArgs = getArgumentRefOrCreateDummy(src);
  auto resultArgs = getArgumentRefOrCreateDummy(result);
  for (int i = 0; i < static_cast<int>(srcArgs.size()); ++i) {
    auto srcDim = solverShapeElem_->find(srcArgs[order[i]]);
    auto resDim = solverShapeElem_->find(resultArgs[i]);
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end())
      tilingDimKindMapForShape[resDim] = it->second;
  }
}

bool DimensionAnalyzer::isParallelDim(Dimension dim) {
  if (isStaticUnitDim(dim))
    return false;
  auto args = getArgumentRefOrCreateDummy(dim.first);
  if (dim.second >= static_cast<int64_t>(args.size()))
    return true;
  auto solverCollapserIndex = solverCollapserElem_->find(args[dim.second]);
  auto solverShapeIndex = solverShapeElem_->find(args[dim.second]);

  auto tilingDimKindVal =
      tilingDimKindMapForCollapser.find(solverCollapserIndex);
  if (tilingDimKindVal != tilingDimKindMapForCollapser.end()) {
    if (tilingDimKindVal->getSecond() != TilingDimensionKind::Parallel &&
        broadcastAxisCaseCandidate.find(solverCollapserIndex) !=
            broadcastAxisCaseCandidate.end()) {
      if (auto it = tilingDimKindMapForShape.find(solverShapeIndex);
          it != tilingDimKindMapForShape.end())
        return it->getSecond() == TilingDimensionKind::Parallel ||
               it->getSecond() == TilingDimensionKind::Reduce;
      return true;
    }
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Parallel ||
           tilingDimKindVal->getSecond() == TilingDimensionKind::Reduce;
  }
  return true;
}

bool DimensionAnalyzer::isReduceDim(Dimension dim) {
  if (isStaticUnitDim(dim))
    return false;
  auto args = getArgumentRefOrCreateDummy(dim.first);
  if (dim.second >= static_cast<int64_t>(args.size()))
    return false;
  auto solverCollapserIndex = solverCollapserElem_->find(args[dim.second]);
  auto solverShapeIndex = solverShapeElem_->find(args[dim.second]);

  auto tilingDimKindVal =
      tilingDimKindMapForCollapser.find(solverCollapserIndex);
  if (tilingDimKindVal != tilingDimKindMapForCollapser.end()) {
    if (tilingDimKindVal->getSecond() != TilingDimensionKind::Parallel &&
        broadcastAxisCaseCandidate.find(solverCollapserIndex) !=
            broadcastAxisCaseCandidate.end()) {
      if (auto it = tilingDimKindMapForShape.find(solverShapeIndex);
          it != tilingDimKindMapForShape.end())
        return it->getSecond() == TilingDimensionKind::Reduce;
      return false;
    }
    return tilingDimKindVal->getSecond() == TilingDimensionKind::Reduce;
  }
  return false;
}

bool DimensionAnalyzer::isBroadcastAxisDim(Dimension dim) {
  if (isStaticUnitDim(dim))
    return false;
  auto args = getArgumentRefOrCreateDummy(dim.first);
  if (dim.second >= static_cast<int64_t>(args.size()))
    return false;

  auto parentIndex = solverCollapserElem_->find(args[dim.second]);
  for (int64_t ref : broadcastAxisDimRefs) {
    if (solverCollapserElem_->find(ref) == parentIndex)
      return true;
  }
  return false;
}

std::pair<int64_t, int64_t>
DimensionAnalyzer::getDimPriority(Dimension dim) {
  int64_t bucket = 0;
  if (isReduceDim(dim))
    bucket = 1;
  if (isBroadcastAxisDim(dim))
    bucket = 2;

  int64_t order = dim.second;
  auto args = getArgumentRef(dim.first);
  auto solverIndex = solverShapeElem_->find(args[dim.second]);
  if (auto it = transposedDimMap.find(solverIndex);
      it != transposedDimMap.end())
    order = it->second;
  return {bucket, order};
}

bool DimensionAnalyzer::computeTilingDim() {
  DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>> parallelDimMaps;
  DenseMap<int64_t, int> numStoreOps;

  for (auto [value, _] : argumentsRefPointer_)
    tilingDim_[value] = -1;

  if (!tailNodeSet.empty()) {
    for (auto value : tailNodeSet) {
      auto src = value;
      if (!argumentsRefPointer_.contains(src))
        continue;
      auto rank = getShapeRankFromType(src.getType()).value_or(0);
      if (rank == 0)
        continue;
      auto args = getArgumentRefOrCreateDummy(src);
      auto groupIndex =
          solverGroup_->find(argumentsRefPointer_.at(src));
      numStoreOps[groupIndex]++;
      auto shape = getShape(src.getType());
      DenseSet<int> usedParentIdx;
      for (size_t i = 0; i < rank; i++) {
        auto parentIndex = solverCollapserElem_->find(args[i]);
        if (!usedParentIdx.insert(parentIndex).second)
          broadcastAxisCaseCandidate.insert(parentIndex);
      }
      usedParentIdx.clear();
      for (size_t i = 0; i < rank; i++) {
        Dimension dim(src, i);
        if (isParallelDim(dim)) {
          if (ShapedType::isDynamic(shape[i]) || shape[i] == 1)
            continue;
          auto parentIndex = solverCollapserElem_->find(args[i]);
          if (usedParentIdx.insert(parentIndex).second) {
            parallelDimMaps[groupIndex][parentIndex].push_back(dim);
          } else {
            auto &otherDim = parallelDimMaps[groupIndex][parentIndex].back();
            if (getDimPriority(dim) < getDimPriority(otherDim))
              otherDim = dim;
          }
        }
      }
    }
  } else {
    computeTilingDimImpl<triton::StoreOp>(parallelDimMaps, numStoreOps);
  }

  DenseMap<int64_t, int> selectedTilingParIdxMap;
  for (const auto &[groupIndex, parallelDimMap] : parallelDimMaps) {
    auto numStoreOp = numStoreOps.at(groupIndex);
    for (const auto &[parentIndex, candidate] : parallelDimMap) {
      if (static_cast<int64_t>(candidate.size()) == numStoreOp) {
        int64_t higherDimCnt = 0;
        for (auto [store, cDim] : candidate) {
          int64_t curDim = tilingDim_[store];
          auto candPriority = getDimPriority(Dimension(store, cDim));
          if (curDim == -1 ||
              getDimPriority(Dimension(store, curDim)) > candPriority)
            higherDimCnt++;
        }
        if (2 * higherDimCnt >= numStoreOp) {
          selectedTilingParIdxMap[groupIndex] = parentIndex;
          for (auto [store, dim] : candidate)
            tilingDim_[store] = dim;
        }
      }
    }
  }

  bool isBroadcastAxisCase = false;
  for (auto [_, parIdx] : selectedTilingParIdxMap) {
    selectedTilingParIdx.insert(parIdx);
    isBroadcastAxisCase |= broadcastAxisCaseCandidate.contains(parIdx);
  }
  return isBroadcastAxisCase;
}

template <typename StoreOpTy>
void DimensionAnalyzer::computeTilingDimImpl(
    DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
        &parallelDimMap,
    DenseMap<int64_t, int> &numStoreOps) {
  op_->walk<WalkOrder::PreOrder>([&](StoreOpTy op) {
    auto src = op.getValue();
    auto rank = getShapeRankFromType(src.getType()).value_or(0);
    if (rank == 0)
      return;
    auto args = getArgumentRefOrCreateDummy(src);
    auto groupIndex = solverGroup_->find(argumentsRefPointer_.at(src));
    numStoreOps[groupIndex]++;
    auto shape = getShape(src.getType());
    DenseSet<int> usedParentIdx;
    for (size_t i = 0; i < rank; i++) {
      auto parentIndex = solverCollapserElem_->find(args[i]);
      if (!usedParentIdx.insert(parentIndex).second)
        broadcastAxisCaseCandidate.insert(parentIndex);
    }
    usedParentIdx.clear();
    for (size_t i = 0; i < rank; i++) {
      Dimension dim(src, i);
      if (isParallelDim(dim)) {
        if (ShapedType::isDynamic(shape[i]) || shape[i] == 1)
          continue;
        auto parentIndex = solverCollapserElem_->find(args[i]);
        if (usedParentIdx.insert(parentIndex).second) {
          parallelDimMap[groupIndex][parentIndex].push_back(dim);
        } else {
          auto &otherDim = parallelDimMap[groupIndex][parentIndex].back();
          if (getDimPriority(dim) < getDimPriority(otherDim))
            otherDim = dim;
        }
      }
    }
  });
}

int64_t DimensionAnalyzer::getTilingDim(Value v) {
  if (!argumentsRefPointer_.contains(v))
    return -1;
  auto rank = getShapeRankFromType(v.getType()).value_or(0);
  int64_t tilingDim = -1;
  std::pair<int64_t, int64_t> bestPriority = {0, 0};
  auto args = getArgumentRef(v);
  for (size_t i = 0; i < rank; i++) {
    auto parentIndex = solverCollapserElem_->find(args[i]);
    if (selectedTilingParIdx.contains(parentIndex) &&
        isParallelDim(Dimension(v, i))) {
      auto candPriority = getDimPriority(Dimension(v, i));
      if (tilingDim == -1 || bestPriority > candPriority) {
        tilingDim = static_cast<int64_t>(i);
        bestPriority = candPriority;
      }
    }
  }
  return tilingDim;
}

SmallVector<int64_t> DimensionAnalyzer::getDimShape(Value v) {
  assert(argumentsRefPointer_.count(v));
  auto vRef = getArgumentRef(v);
  SmallVector<int64_t> ret(vRef.size());
  for (size_t i = 0; i < ret.size(); ++i)
    ret[i] = solverShapeElem_->getMinParentAndShapePair(vRef[i]).second;
  return ret;
}

Dimension DimensionAnalyzer::getEarliestDimension(Dimension dim) {
  auto firstParentIndex =
      solverShapeElem_->minIndex[solverShapeElem_->find(
          getArgumentRef(dim.first)[dim.second])];
  return getDimension(firstParentIndex);
}

Dimension DimensionAnalyzer::getDimension(int64_t parentIndex) {
  if (reverseShapeElem_.empty())
    computeReverseElementMap();
  parentIndex = solverShapeElem_->find(parentIndex);
  auto value = reverseShapeElem_.at(parentIndex);
  auto vRef = getArgumentRef(value);
  for (size_t i = 0; i < vRef.size(); ++i) {
    auto currentElIdx = solverShapeElem_->find(vRef[i]);
    if (currentElIdx == parentIndex)
      return Dimension(value, i);
  }
  llvm_unreachable("Element shape index cannot be inferred");
}

void DimensionAnalyzer::computeReverseElementMap() {
  auto markShapes = [this](Value val) -> void {
    if (!argumentsRefPointer_.contains(val))
      return;
    auto vRef = getArgumentRef(val);
    for (auto el : vRef) {
      auto currentElIdx = solverShapeElem_->find(el);
      if (!reverseShapeElem_.contains(currentElIdx))
        reverseShapeElem_[currentElIdx] = val;
    }
  };
  for (auto arg : argumentList_)
    markShapes(arg);
  for (Block &block : op_->getRegion(0)) {
    block.walk([&markShapes](Operation *op) {
      for (auto res : op->getResults())
        markShapes(res);
    });
  }
}

SmallVector<int64_t> DimensionAnalyzer::getArgumentRef(Value v) const {
  auto it = argumentsRefPointer_.find(v);
  if (it == argumentsRefPointer_.end())
    return SmallVector<int64_t>();
  return argumentsRef_[it->second];
}

SmallVector<int64_t>
DimensionAnalyzer::getArgumentRefOrCreateDummy(Value v) {
  createDummyRefIfNotExist({v});
  return getArgumentRef(v);
}

bool DimensionAnalyzer::areDimensionsEqual(Dimension lhs, Dimension rhs,
                                           bool isStrict) {
  auto lhsRef = getArgumentRefOrCreateDummy(lhs.first);
  auto rhsRef = getArgumentRefOrCreateDummy(rhs.first);
  if (isStrict)
    return solverShapeElem_->find(lhsRef[lhs.second]) ==
           solverShapeElem_->find(rhsRef[rhs.second]);
  return solverCollapserElem_->find(lhsRef[lhs.second]) ==
         solverCollapserElem_->find(rhsRef[rhs.second]);
}

void DimensionAnalyzer::createDummyRefIfNotExist(ArrayRef<Value> values) {
  for (auto curVal : values) {
    if (argumentsRefPointer_.contains(curVal))
      continue;
    auto shapeInfo = getValueShapeInfo(curVal).value_or(
        std::make_pair(0, DimensionShape{}));
    auto rank = shapeInfo.first;
    auto shape = shapeInfo.second;
    int startingIdx = allocateArguments(rank, shape);
    argumentsRef_.push_back(DimensionShape(shape));
    std::iota(argumentsRef_.back().begin(),
              argumentsRef_.back().end(), startingIdx);
    initCollapseOrVerify(curVal,
                         static_cast<int64_t>(argumentsRef_.size() - 1));
  }
}

void DimensionAnalyzer::updatePreviousType(const Value &val) {
  auto curType = dyn_cast<ShapedType>(val.getType());
  if (curType)
    updatePreviousType(val, curType);
}

void DimensionAnalyzer::updatePreviousType(const Value &val,
                                           const ShapedType &curType) {
  if (previousType_.contains(val))
    assert(previousType_[val] == curType);
  previousType_[val] = curType;
}

void DimensionAnalyzer::collapsePropagateOrVerify(Operation *op,
                                                  const Value &refVal) {
  auto tmpVal = argumentsRefPointer_.at(refVal);
  for (const Value newVal : op->getResults()) {
    if (argumentsRefPointer_.contains(newVal)) {
      solverSegments_->join(argumentsRefPointer_.at(newVal), tmpVal);
      return;
    }
    argumentsRefPointer_[newVal] = tmpVal;
  }
}

void DimensionAnalyzer::collapsePropagateOrVerify(const Value &newVal,
                                                  const Value &arg) {
  auto tmpVal = argumentsRefPointer_.at(arg);
  if (argumentsRefPointer_.contains(newVal)) {
    solverSegments_->join(argumentsRefPointer_.at(newVal), tmpVal);
    return;
  }
  argumentsRefPointer_[newVal] = tmpVal;
}

void DimensionAnalyzer::initCollapseOrVerify(const Value &val,
                                             int64_t refPtr) {
  if (argumentsRefPointer_.contains(val)) {
    solverSegments_->join(argumentsRefPointer_.at(val), refPtr);
    return;
  }
  argumentsRefPointer_[val] = refPtr;
}
