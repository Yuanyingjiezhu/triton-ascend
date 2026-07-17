//===- BubbleUpPattern.h ----------------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#ifndef TRITON_CV12_SPLIT_BUBBLE_UP_PATTERN_H
#define TRITON_CV12_SPLIT_BUBBLE_UP_PATTERN_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>

namespace mlir::triton::detail {

class BubbleUpStrategy {
public:
  virtual ~BubbleUpStrategy() = default;
  virtual LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                                  PatternRewriter &rewriter) const = 0;
  virtual bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const = 0;
  virtual bool canHandleBlockArgument() const { return false; }
  void setAggressive(bool aggressive) { this->aggressive = aggressive; }
  bool isAggressive() const { return aggressive; }

protected:
  bool aggressive = false;
};

class ElementwiseBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class TritonBroadcastBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class IfBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class TritonDotBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class TritonReduceBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class TritonExpandDimsBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class TritonSplatBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                         PatternRewriter &rewriter) const override;
};

class TritonLoadBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                        PatternRewriter &rewriter) const override;
};

class TritonMakeRangeBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                          PatternRewriter &rewriter) const override;
};

class TritonTransBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                          PatternRewriter &rewriter) const override;
};

class TritonReshapeBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                          PatternRewriter &rewriter) const override;
};

class InsertSliceBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                          PatternRewriter &rewriter) const override;
};

class LoopInBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                          PatternRewriter &rewriter) const override;
  bool canHandleBlockArgument() const override { return true; }
};

class LoopOutBubbleUpStrategy : public BubbleUpStrategy {
public:
  bool isSupportedOperation(tensor::ExtractSliceOp sliceOp) const override;
  LogicalResult execute(tensor::ExtractSliceOp sliceOp,
                          PatternRewriter &rewriter) const override;
  bool canHandleBlockArgument() const override { return true; }
};

class BubbleUpPattern : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  BubbleUpPattern(MLIRContext *context,
                   SmallVector<std::shared_ptr<BubbleUpStrategy>> strategies,
                   bool aggressive = false)
      : OpRewritePattern<tensor::ExtractSliceOp>(context),
        bubbleUpStrategies(std::move(strategies)),
        aggressive(aggressive) {}

  void setAggressive(bool aggressive) {
    this->aggressive = aggressive;
    for (auto &strategy : bubbleUpStrategies)
      strategy->setAggressive(aggressive);
  }

protected:
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                 PatternRewriter &rewriter) const final;

private:
  SmallVector<std::shared_ptr<BubbleUpStrategy>> bubbleUpStrategies;
  bool aggressive = false;
};

} // namespace mlir::triton::detail

#endif // TRITON_CV12_SPLIT_BUBBLE_UP_PATTERN_H
