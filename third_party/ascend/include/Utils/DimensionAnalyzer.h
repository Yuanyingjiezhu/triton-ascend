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

#ifndef TRITONNPU_UTILS_DIMENSION_ANALYZER_H
#define TRITONNPU_UTILS_DIMENSION_ANALYZER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace mlir {
namespace triton {

class UnionFindBase {
public:
  UnionFindBase(size_t n = 0) : minIndex(n), parent_(n, -1) {
    std::iota(minIndex.begin(), minIndex.end(), 0);
  }
  virtual ~UnionFindBase() = default;

  int find(int x);
  virtual bool join(int a, int b);
  virtual void allocateMinimum(size_t n);

  std::vector<int> minIndex;

protected:
  std::vector<int> parent_;
};

class SimpleUnionFind : public UnionFindBase {
public:
  SimpleUnionFind(size_t n = 0) : UnionFindBase(n) {}
  virtual ~SimpleUnionFind() = default;
};

using DimensionPosition = std::pair<int64_t, int64_t>;
static constexpr int64_t kUndefinedShaped = -1;
static constexpr DimensionPosition kMaxDimPos = {
    std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()};

class ExtendedUnionFind : public UnionFindBase {
public:
  ExtendedUnionFind(size_t n = 0)
      : UnionFindBase(n), shape_(n, ShapedType::kDynamic),
        minParentIndex_(n, kMaxDimPos) {}
  bool join(int a, int b) override;
  void allocateMinimum(size_t n) override;
  std::pair<DimensionPosition, int64_t> getMinParentAndShapePair(int a);

  SmallVector<int64_t> shape_;
  SmallVector<std::pair<int64_t, int64_t>> minParentIndex_;
};

struct ConnectedLeftRight {
  bool leftConnected = true;
  bool rightConnected = true;
  enum ElementKind {
    NoMutation = 0,
    Unit = 1,
    HasMutation = 2,
  };
  ElementKind elementKind = ElementKind::NoMutation;
  ConnectedLeftRight() = default;
};

using Dimension = std::pair<Value, int64_t>;
using DimensionIndex = SmallVector<int64_t>;
using DimensionShape = SmallVector<int64_t>;
using ArgumentIndex = SmallVector<int64_t>;

struct DimensionAnalyzerOptions {
  DimensionAnalyzerOptions() = default;

  std::function<bool(Operation *)> isHeadOp;
  std::function<bool(Value)> isTailNode;
  std::function<bool(Value, OpOperand &)> shouldSkipUse;
};

enum class TilingDimensionKind {
  Parallel,
  RankReduced,
  Reduce,
};

class DimensionAnalyzer {
public:
  explicit DimensionAnalyzer(Operation *op,
                             DimensionAnalyzerOptions options = {});
  virtual ~DimensionAnalyzer() = default;

  LogicalResult initialize();

  bool computeTilingDim();

  int64_t getTilingDim(Value v);

  bool isParallelDim(Dimension dim);
  bool isReduceDim(Dimension dim);
  bool isBroadcastAxisDim(Dimension dim);
  std::pair<int64_t, int64_t> getDimPriority(Dimension dim);

  SmallVector<int64_t> getDimShape(Value v);

  Dimension getEarliestDimension(Dimension dim);

  Dimension getDimension(int64_t parentIndex);

  SmallVector<int64_t> getArgumentRef(Value v) const;
  SmallVector<int64_t> getArgumentRefOrCreateDummy(Value v);

  bool areDimensionsEqual(Dimension lhs, Dimension rhs, bool isStrict = false);

protected:
  void initializeStructures();

  void processBFS();

  void unifyGroups();

  void combineInferable();

  void propagateConnection();
  void spreadConnection();

  int64_t allocateArguments(int rank, ArrayRef<int64_t> dimensionRef);

  void processArgument(Value arg);

  void createDummyRefIfNotExist(ArrayRef<Value> values);

  void computeReverseElementMap();

  void mergeValues(ArrayRef<Value> inputs, ArrayRef<Value> outputs,
                   ArrayRef<int64_t> mutatedDims = {},
                   bool mergeMutation = true);

  void updatePreviousType(const Value &val);
  void updatePreviousType(const Value &val, const ShapedType &curType);

  void collapsePropagateOrVerify(Operation *op, const Value &refVal);
  void collapsePropagateOrVerify(const Value &newVal, const Value &arg);
  void initCollapseOrVerify(const Value &val, int64_t refPtr);

  bool processOperation(Operation *op, Value current);

  void processParallelOp(Operation *op, Value current);
  void processValue(Value v, Value current);

  size_t processDecreasingDimensions(ArrayRef<int64_t> inputArgs,
                                     ArrayRef<int64_t> dimensions,
                                     const Value &output);

  size_t processPermutation(ArrayRef<int64_t> inputArgs, ArrayRef<int64_t> perm,
                            const Value &output);

  void processMatmulOp(Operation *op, bool isTransposeA = false,
                       bool isTransposeB = false);
  void processBatchMatmulOp(linalg::BatchMatmulOp batchMatmulOp);

  void processConcatOp(tensor::ConcatOp concatOp);
  void processPadOp(tensor::PadOp padOp);

  template <class T, typename = std::enable_if_t<
                         std::is_same_v<T, tensor::ExtractSliceOp> ||
                         std::is_same_v<T, tensor::InsertSliceOp>>>
  void processSlicingOp(T slicingOp);

  void processExtractSliceOp(tensor::ExtractSliceOp extractSliceOp);
  void processInsertOp(tensor::InsertOp insertOp);
  void processInsertSliceOp(tensor::InsertSliceOp insertSliceOp);
  void processIfOp(scf::IfOp op);
  void processForOp(scf::ForOp op);

  void processSplatOp(triton::SplatOp op);
  void processBroadcastOp(triton::BroadcastOp op);
  void processExpandDimsOp(triton::ExpandDimsOp op);
  void processTransOp(triton::TransOp op);
  void processReduceOp(triton::ReduceOp op);
  void processDotOp(triton::DotOp op);
  void processDotScaledOp(triton::DotScaledOp op);
  void processCatOp(triton::CatOp op);
  void processReshapeOp(triton::ReshapeOp op);
  void processJoinOp(triton::JoinOp op);
  void processSplitOp(triton::SplitOp op);

  template <typename T, typename = std::enable_if_t<
                             std::is_same_v<T, tensor::ExpandShapeOp> ||
                             std::is_same_v<T, tensor::CollapseShapeOp>>>
  void processReshapeOp(T op);

  void joinShape(int a, int b);
  void joinCollapser(int a, int b);
  void disconnect(int a, int b);
  bool isConnected(int a, int b);
  void separateGroup(Value val, BitVector contiguousMask,
                     ArrayRef<int64_t> shape);

  void markDimensions();
  void transferDimMark();

  template <typename IntegerRange>
  void transferDimMarkImpl(Value input, Value output,
                           const IntegerRange &mutated);

  void transferDimMarkImpl(tensor::ExpandShapeOp op);
  void transferDimMarkImpl(tensor::CollapseShapeOp op);
  void transferDimMarkImpl(tensor::ExtractSliceOp op);
  void transferDimMarkImpl(tensor::InsertSliceOp op);
  void transferDimMarkImpl(triton::BroadcastOp op);
  void transferDimMarkImpl(triton::ExpandDimsOp op);
  void transferDimMarkImpl(triton::ReshapeOp op);
  void transferDimMarkImpl(triton::TransOp op);

  template <typename StoreOpTy>
  void computeTilingDimImpl(
      DenseMap<int64_t, DenseMap<int64_t, SmallVector<Dimension>>>
          &parallelDimMap,
      DenseMap<int64_t, int> &numStoreOps);

  bool isAllowedType(Type type);

protected:
  Operation *op_;
  int64_t dimensionAllocation_ = 0;
  int64_t tilingSize = 2;

  SmallVector<Value> argumentList_;
  ArgumentIndex argumentIndex_;
  int64_t argumentTotalLength_ = 0;
  SmallVector<ConnectedLeftRight, 4> isConnected_;
  SmallVector<DimensionIndex> argumentsRef_;
  DenseMap<Value, int> argumentsRefPointer_;
  DenseMap<Value, ShapedType> previousType_;

  std::unique_ptr<ExtendedUnionFind> solverShapeElem_;
  std::unique_ptr<SimpleUnionFind> solverCollapserElem_;
  std::unique_ptr<SimpleUnionFind> solverSegments_;
  std::unique_ptr<SimpleUnionFind> solverGroup_;

  DenseMap<int64_t, Value> reverseShapeElem_;

  DenseMap<Value, int64_t> tilingDim_;

  DenseMap<int64_t, TilingDimensionKind> tilingDimKindMapForCollapser;
  DenseMap<int64_t, TilingDimensionKind> tilingDimKindMapForShape;

  llvm::SmallDenseSet<int> selectedTilingParIdx;

  DenseMap<int64_t, int64_t> transposedDimMap;

  llvm::SmallDenseSet<int64_t> broadcastAxisCaseCandidate;
  llvm::SmallDenseSet<int64_t> broadcastAxisDimRefs;

  DenseSet<Value> headNodeSet;
  DenseSet<Value> tailNodeSet;

  DimensionAnalyzerOptions options;

  bool bindUsingTensorDim = true;

  std::optional<size_t> getShapeRankFromType(Type type);
  std::optional<std::pair<size_t, SmallVector<int64_t>>>
  getValueShapeInfo(Value v);
  SmallVector<int64_t> getShape(Type type);
};

} // namespace triton
} // namespace mlir

#endif // TRITONNPU_UTILS_DIMENSION_ANALYZER_H
