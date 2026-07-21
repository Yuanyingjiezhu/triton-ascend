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

#ifndef TRITONNPU_UTILS_DIMENSION_GRAPH_ANALYZER_H
#define TRITONNPU_UTILS_DIMENSION_GRAPH_ANALYZER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <utility>

namespace mlir {
namespace triton {

struct DimensionGraphAnalyzerOptions {
  DimensionGraphAnalyzerOptions() = default;

  std::function<bool(Operation *)> isHeadOp;
  std::function<bool(Value)> isTailNode;
  std::function<bool(Value, OpOperand &)> shouldSkipUse;
};

class DimensionGraphAnalyzer {
public:
  using Dimension = std::pair<Value, int64_t>;

  struct AxisInfo {
    Value owner;
    int64_t dim = -1;
    int64_t shape = ShapedType::kDynamic;
  };

  struct ValueComponentInfo {
    SmallVector<int64_t> valueIds;
    SmallVector<int64_t> tailValueIds;
    SmallVector<int64_t> axisComponentIds;
    SmallVector<int64_t> selectedAxisComponentIds;
    SmallVector<int64_t> selectedAxisIds;
  };

  struct AxisComponentInfo {
    SmallVector<int64_t> axisIds;
    SmallVector<int64_t> tailAxisIds;
  };

  explicit DimensionGraphAnalyzer(
      Operation *op, DimensionGraphAnalyzerOptions options = {});

  LogicalResult initialize();

  bool computeTilingDim();

  int64_t getTilingDim(Value value) const;

  SmallVector<int64_t> getArgumentRef(Value value) const;
  SmallVector<int64_t> getArgumentRefOrCreateDummy(Value value);

  ArrayRef<Value> getTailValues() const { return tailValues_; }
  ArrayRef<ValueComponentInfo> getValueComponents() const {
    return valueComponents_;
  }
  ArrayRef<AxisComponentInfo> getAxisComponents() const {
    return axisComponents_;
  }

private:
  using AxisSet = DenseSet<int64_t>;
  using AxisSetList = SmallVector<AxisSet, 0>;

  void reset();
  void dumpInitializeSummary() const;
  void dumpComputeSummary() const;
  LogicalResult initializeStructures();
  void processBFS();
  bool processOperation(Operation *operation);

  void processBroadcastOp(BroadcastOp op);
  void processExpandDimsOp(ExpandDimsOp op);
  void processTransOp(TransOp op);
  void processReduceOp(ReduceOp op);
  void processScanOp(ScanOp op);
  void processReshapeOp(ReshapeOp op);
  void processForOp(scf::ForOp op);
  void processWhileOp(scf::WhileOp op);
  void processConditionOp(scf::ConditionOp op);
  void processYieldOp(scf::YieldOp op);
  bool processElementwiseOp(Operation *operation);
  void recordSameRankDependency(Value src, Value dst);

  int64_t getOrCreateValueId(Value value);
  SmallVector<int64_t> createAxesForValue(Value value);
  void createDummyRefIfNotExist(ArrayRef<Value> values);

  void recordAxisDependency(int64_t srcAxis, int64_t dstAxis);
  void removeSkippedDependencyEdges();
  void rebuildComponents();
  void rebuildValueComponents();
  void rebuildAxisComponents();
  void selectAxisComponentsForValueComponents();
  void addTailValue(Value value);
  bool useProducesScalar(OpOperand &use) const;
  bool shouldSkipDependencyValue(Value value) const;
  void selectTilingAxes();
  AxisSetList traverseAxisComponent(int64_t axisComponentId) const;
  void recursiveTraversal(SmallVector<int64_t> frontier, AxisSet visited,
                          AxisSet selectedGroup,
                          AxisSetList &visitedList,
                          int64_t failNum, int64_t level,
                          int64_t axisComponentId) const;
  void dumpAxisSet(StringRef label, const AxisSet &axisSet) const;
  void dumpAxisList(StringRef label, ArrayRef<int64_t> axisIds) const;
  bool standardAction(int64_t axis, SmallVectorImpl<int64_t> &frontier,
                      AxisSet &visited, AxisSet &selectedGroup,
                      int64_t &failNum, int64_t &level,
                      int64_t axisComponentId) const;
  SmallVector<int64_t> getNeighborsInSameValue(int64_t axisId) const;
  SmallVector<int64_t> getInEdgesInAxisComponent(
      int64_t axisId, int64_t axisComponentId) const;
  SmallVector<int64_t> getOutEdgesInAxisComponent(
      int64_t axisId, int64_t axisComponentId) const;
  AxisSet generateSelectedGroup(int64_t axisId,
                                int64_t axisComponentId) const;
  bool isAxisInComponent(int64_t axisId, int64_t axisComponentId) const;
  AxisSet removeUnitShapeAxes(const AxisSet &axisSet) const;
  static int64_t getAxisSetScore(const AxisSet &axisSet);

  SmallVector<int64_t> getNonUnitDims(ArrayRef<int64_t> shape, size_t begin,
                                      size_t end) const;

  Operation *op_ = nullptr;
  DimensionGraphAnalyzerOptions options_;

  int64_t nextValueId_ = 0;
  DenseMap<Value, int64_t> valueIds_;
  SmallVector<Value> valuesById_;
  DenseMap<Value, SmallVector<int64_t>> argumentsRefPointer_;
  SmallVector<AxisInfo> axes_;
  SmallVector<SmallVector<int64_t>> outEdges_;
  SmallVector<SmallVector<int64_t>> inEdges_;
  SmallVector<SmallVector<int64_t>> valueOutEdges_;
  SmallVector<SmallVector<int64_t>> valueInEdges_;
  DenseMap<Value, int64_t> tilingDim_;

  DenseSet<Value> headNodeSet_;
  SmallVector<Value> tailValues_;
  DenseSet<Value> tailValueSet_;
  SmallVector<ValueComponentInfo, 0> valueComponents_;
  SmallVector<int64_t> valueComponentIdByValue_;
  SmallVector<AxisComponentInfo, 0> axisComponents_;
  SmallVector<int64_t> axisComponentIdByAxis_;
};

} // namespace triton
} // namespace mlir

#endif // TRITONNPU_UTILS_DIMENSION_GRAPH_ANALYZER_H
