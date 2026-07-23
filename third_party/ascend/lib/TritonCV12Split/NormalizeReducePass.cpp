//===- NormalizeReducePass.cpp - Normalize tt.reduce ---------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "TritonCV12Split/NormalizeReducePass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_NORMALIZEREDUCE
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-normalize-reduce"

using namespace mlir;

namespace {

static bool isAscend950(ModuleOp moduleOp) {
  Attribute target = moduleOp->getAttr("hacc.target");
  if (!target)
    return false;

  std::string targetString;
  llvm::raw_string_ostream os(targetString);
  target.print(os);
  return llvm::StringRef(os.str()).contains("Ascend950");
}

static bool isScalarI1(Type type) {
  return type.isInteger(1);
}

static bool isI1Tensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getElementType().isInteger(1);
}

static bool hasF16ElementType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tensorType.getElementType().isF16();
  return type.isF16();
}

static bool isSimpleI1AddReduce(triton::ReduceOp op) {
  if (op.getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  if (op.getAxis() != 0 || !isI1Tensor(op.getOperand(0)) ||
      !isScalarI1(op->getResult(0).getType()))
    return false;

  Region &region = op.getCombineOp();
  if (!llvm::hasSingleElement(region))
    return false;

  Block &block = region.front();
  if (block.getNumArguments() != 2)
    return false;

  auto returnOp = dyn_cast<triton::ReduceReturnOp>(block.getTerminator());
  if (!returnOp || returnOp->getNumOperands() != 1)
    return false;

  auto addOp = returnOp->getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!addOp || !addOp.getType().isInteger(1))
    return false;

  Value lhs = addOp->getOperand(0);
  Value rhs = addOp->getOperand(1);
  return lhs == block.getArgument(0) && rhs == block.getArgument(1);
}

/// Matches the canonical TTIR spelling of an f16 reduce-sum:
///   tt.reduce(%src) { ^bb0(%lhs: f16, %rhs: f16):
///     tt.reduce.return arith.addf %lhs, %rhs }
static bool isSimpleF16AddReduce(triton::ReduceOp op) {
  if (op.getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  if (!hasF16ElementType(op.getOperand(0).getType()) ||
      !hasF16ElementType(op->getResult(0).getType()))
    return false;

  Region &region = op.getCombineOp();
  if (!llvm::hasSingleElement(region))
    return false;

  Block &block = region.front();
  if (block.getNumArguments() != 2 ||
      !block.getArgument(0).getType().isF16() ||
      !block.getArgument(1).getType().isF16())
    return false;

  auto returnOp = dyn_cast<triton::ReduceReturnOp>(block.getTerminator());
  if (!returnOp || returnOp->getNumOperands() != 1)
    return false;

  auto addOp = returnOp->getOperand(0).getDefiningOp<arith::AddFOp>();
  if (!addOp || !addOp.getType().isF16())
    return false;

  return addOp.getLhs() == block.getArgument(0) &&
         addOp.getRhs() == block.getArgument(1);
}

struct NormalizeI1ReducePattern : public OpRewritePattern<triton::ReduceOp> {
  using OpRewritePattern<triton::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp || !isAscend950(moduleOp) || !isSimpleI1AddReduce(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getOperand(0);
    auto inputType = cast<RankedTensorType>(input.getType());
    Type i16Type = rewriter.getI16Type();
    auto selectedType = RankedTensorType::get(inputType.getShape(), i16Type,
                                              inputType.getEncoding());

    auto zeroTensorAttr = DenseElementsAttr::get(
        selectedType, rewriter.getIntegerAttr(i16Type, 0));
    auto oneTensorAttr = DenseElementsAttr::get(
        selectedType, rewriter.getIntegerAttr(i16Type, 1));
    Value zeroTensor =
        rewriter.create<arith::ConstantOp>(loc, selectedType, zeroTensorAttr);
    Value oneTensor =
        rewriter.create<arith::ConstantOp>(loc, selectedType, oneTensorAttr);
    Value selected =
        rewriter.create<arith::SelectOp>(loc, input, oneTensor, zeroTensor);

    auto newReduce =
        rewriter.create<triton::ReduceOp>(loc, ValueRange{selected}, op.getAxis());
    newReduce->setAttrs(op->getAttrs());

    Region &newRegion = newReduce.getCombineOp();
    Block *block = rewriter.createBlock(&newRegion);
    block->addArgument(i16Type, loc);
    block->addArgument(i16Type, loc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      Value max = rewriter.create<arith::MaxSIOp>(loc, block->getArgument(0),
                                                  block->getArgument(1));
      rewriter.create<triton::ReduceReturnOp>(loc, max);
    }

    rewriter.setInsertionPointAfter(newReduce);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, i16Type, rewriter.getIntegerAttr(i16Type, 0));
    Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               newReduce->getResult(0), zero);
    rewriter.replaceOp(op, cmp);
    return success();
  }
};

/// Promotes f16 reduce-sum accumulation to f32 while preserving the original
/// f16 result type for downstream TTIR users.
struct NormalizeF16ReduceSumPattern
    : public OpRewritePattern<triton::ReduceOp> {
  using OpRewritePattern<triton::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp || !isAscend950(moduleOp) || !isSimpleF16AddReduce(op))
      return failure();

    Location loc = op.getLoc();
    Value input = op.getOperand(0);
    auto inputType = cast<RankedTensorType>(input.getType());
    auto promotedInputType = RankedTensorType::get(
        inputType.getShape(), rewriter.getF32Type(), inputType.getEncoding());
    Value promotedInput =
        rewriter.create<arith::ExtFOp>(loc, promotedInputType, input);

    auto promotedReduce = rewriter.create<triton::ReduceOp>(
        loc, ValueRange{promotedInput}, op.getAxis());
    promotedReduce->setAttrs(op->getAttrs());

    Block *block = rewriter.createBlock(&promotedReduce.getCombineOp());
    block->addArgument(rewriter.getF32Type(), loc);
    block->addArgument(rewriter.getF32Type(), loc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      Value sum = rewriter.create<arith::AddFOp>(
          loc, block->getArgument(0), block->getArgument(1));
      rewriter.create<triton::ReduceReturnOp>(loc, sum);
    }

    rewriter.setInsertionPointAfter(promotedReduce);
    Value result = rewriter.create<arith::TruncFOp>(
        loc, op->getResult(0).getType(), promotedReduce->getResult(0));
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NormalizeReducePass
    : public triton::impl::NormalizeReduceBase<NormalizeReducePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<NormalizeI1ReducePattern>(&getContext());
    patterns.add<NormalizeF16ReduceSumPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createNormalizeReducePass() {
  return std::make_unique<NormalizeReducePass>();
}
