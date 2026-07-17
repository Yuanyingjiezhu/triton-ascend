/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "TritonCV12Split/NormalizeDotPass.h"

#include "ascend/include/TritonCV12Split/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <optional>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_NORMALIZEDOT
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "normalize-dot"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kStandardized =
    "triton_cv12.normalized_dot";
constexpr llvm::StringLiteral kMayNotExec =
    "triton_cv12.dot_may_not_exec";
constexpr llvm::StringLiteral kAddFromDot =
    "triton_cv12.add_from_dot";
constexpr llvm::StringLiteral kHIVMMatmulLimitedInCube =
    "hivm.matmul_limited_in_cube";

struct SplitInfo {
  bool mayNotExec;
  Value outerInValue;
  Value outerOutValue;
  bool shouldSplit;
  bool shouldAdd = true;
  bool useCounterFallback = false;
};

static Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

static bool isNumericTensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType &&
         isa<FloatType, IntegerType>(tensorType.getElementType());
}

static bool isZeroAttribute(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto elementsAttr = dyn_cast<DenseElementsAttr>(attr)) {
    if (!elementsAttr.isSplat())
      return false;
    Type elementType = elementsAttr.getElementType();
    if (isa<IntegerType>(elementType))
      return elementsAttr.getSplatValue<APInt>().isZero();
    if (isa<FloatType>(elementType))
      return elementsAttr.getSplatValue<APFloat>().isZero();
  }
  return false;
}

static bool isZeroValue(Value value) {
  if (matchPattern(value, m_Zero()) ||
      matchPattern(value, m_AnyZeroFloat()))
    return true;

  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>())
    return isZeroAttribute(constantOp.getValue());
  if (auto splatOp = value.getDefiningOp<triton::SplatOp>())
    return isZeroValue(splatOp.getSrc());
  if (auto broadcastOp = value.getDefiningOp<triton::BroadcastOp>())
    return isZeroValue(broadcastOp.getSrc());
  if (auto expandOp = value.getDefiningOp<triton::ExpandDimsOp>())
    return isZeroValue(expandOp.getSrc());
  if (auto reshapeOp = value.getDefiningOp<triton::ReshapeOp>())
    return isZeroValue(reshapeOp.getSrc());
  return false;
}

static bool scfMayNotExec(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case([&](scf::ForOp forOp) {
        IntegerAttr upperBound;
        IntegerAttr lowerBound;
        if (matchPattern(forOp.getUpperBound(), m_Constant(&upperBound)) &&
            matchPattern(forOp.getLowerBound(), m_Constant(&lowerBound)))
          return upperBound.getValue().sle(lowerBound.getValue());
        return true;
      })
      .Case([&](scf::IfOp ifOp) {
        return !matchPattern(ifOp.getCondition(), m_One());
      })
      .Default([](Operation *) { return false; });
}

static Value searchInAccumulatorChain(Value nextOutput,
                                      bool &accumulatorLimitedToDot,
                                      bool &mayNotExec,
                                      bool &mayNotExecWithIf,
                                      Value &outerInput) {
  Operation *op = nextOutput.getDefiningOp();
  if (!op || outerInput.getDefiningOp())
    return nextOutput;

  Operation *parentOp = op->getParentOp();
  Value nextSearchValue = nextOutput;

  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    auto blockArg = dyn_cast<BlockArgument>(outerInput);
    if (!blockArg || blockArg.getOwner() != forOp.getBody() ||
        blockArg.getArgNumber() < forOp.getNumInductionVars()) {
      accumulatorLimitedToDot = false;
      return nextOutput;
    }

    unsigned argIndex =
        blockArg.getArgNumber() - forOp.getNumInductionVars();
    for (OpOperand &use : blockArg.getUses()) {
      Operation *user = getAncestorInBlock(use.getOwner(), op->getBlock());
      if (user == op)
        continue;
      auto yieldOp = dyn_cast_if_present<scf::YieldOp>(user);
      if (!yieldOp || use.getOperandNumber() != argIndex) {
        accumulatorLimitedToDot = false;
        break;
      }
    }

    if (accumulatorLimitedToDot) {
      for (OpOperand &use : nextOutput.getUses()) {
        Operation *user = getAncestorInBlock(use.getOwner(), op->getBlock());
        auto yieldOp = dyn_cast_if_present<scf::YieldOp>(user);
        if (!yieldOp || yieldOp->getBlock() != forOp.getBody() ||
            use.getOperandNumber() != argIndex) {
          accumulatorLimitedToDot = false;
          break;
        }
      }
    }

    if (accumulatorLimitedToDot) {
      outerInput = forOp.getInitArgs()[argIndex];
      nextSearchValue = forOp.getResult(argIndex);
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
    if (!ifOp.elseBlock()) {
      accumulatorLimitedToDot = false;
      return nextOutput;
    }

    auto *opBlock = op->getBlock();
    auto opYield = cast<scf::YieldOp>(opBlock->getTerminator());
    auto otherYield = cast<scf::YieldOp>(
        opBlock == ifOp.thenBlock() ? ifOp.elseBlock()->getTerminator()
                                    : ifOp.thenBlock()->getTerminator());
    int resultIndex = -1;
    for (unsigned i = 0; i < otherYield->getNumOperands(); ++i) {
      if (otherYield->getOperand(i) == outerInput &&
          opYield.getOperand(i) == nextSearchValue) {
        resultIndex = i;
        break;
      }
    }
    if (resultIndex < 0)
      accumulatorLimitedToDot = false;
    else
      nextSearchValue = ifOp.getResult(resultIndex);
  } else {
    accumulatorLimitedToDot = false;
  }

  if (!accumulatorLimitedToDot)
    return nextOutput;

  bool parentMayNotExec = scfMayNotExec(parentOp);
  mayNotExec |= parentMayNotExec;
  mayNotExecWithIf |= parentMayNotExec && isa<scf::IfOp>(parentOp);
  return searchInAccumulatorChain(nextSearchValue, accumulatorLimitedToDot,
                                  mayNotExec, mayNotExecWithIf, outerInput);
}

static std::optional<Value> getTransparentSource(Operation *op) {
  if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op))
    return reshapeOp.getSrc();
  if (auto expandOp = dyn_cast<triton::ExpandDimsOp>(op))
    return expandOp.getSrc();
  if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op))
    return extractOp.getSource();
  return std::nullopt;
}

static std::optional<Operation *> traceDefiningDot(Value value) {
  if (auto dotOp = value.getDefiningOp<triton::DotOp>())
    return dotOp.getOperation();

  if (Operation *defOp = value.getDefiningOp()) {
    if (auto source = getTransparentSource(defOp))
      return traceDefiningDot(*source);
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (yieldOp && index < yieldOp->getNumOperands())
        return traceDefiningDot(yieldOp.getOperand(index));
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      auto yieldOp =
          dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
      if (yieldOp && index < yieldOp->getNumOperands())
        return traceDefiningDot(yieldOp.getOperand(index));
    }
    return std::nullopt;
  }

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return std::nullopt;
  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp || blockArg.getArgNumber() < forOp.getNumInductionVars())
    return std::nullopt;
  unsigned index =
      blockArg.getArgNumber() - forOp.getNumInductionVars();
  return traceDefiningDot(forOp.getInitArgs()[index]);
}

static bool canReusePreviousLoopAccumulator(triton::DotOp dotOp,
                                            Value outerInput,
                                            bool mayNotExecWithIf) {
  if (mayNotExecWithIf || !outerInput.hasOneUse())
    return false;

  auto currentFor =
      dyn_cast<scf::ForOp>(dotOp->getBlock()->getParentOp());
  auto blockArg = dyn_cast<BlockArgument>(dotOp.getC());
  if (!currentFor || !blockArg ||
      blockArg.getOwner() != currentFor.getBody() ||
      blockArg.getArgNumber() < currentFor.getNumInductionVars())
    return false;

  unsigned index =
      blockArg.getArgNumber() - currentFor.getNumInductionVars();
  if (index >= currentFor.getNumRegionIterArgs() ||
      currentFor.getInitArgs()[index] != outerInput)
    return false;

  if (outerInput.getDefiningOp<triton::DotOp>())
    return true;

  auto previousFor = outerInput.getDefiningOp<scf::ForOp>();
  if (!previousFor || scfMayNotExec(previousFor) ||
      previousFor->getBlock() != currentFor->getBlock() ||
      !previousFor->isBeforeInBlock(currentFor))
    return false;

  return traceDefiningDot(outerInput).has_value();
}

static bool verifyDot(triton::DotOp dotOp) {
  if (dotOp->hasAttr(kStandardized))
    return false;
  if (!isNumericTensor(dotOp.getC().getType()))
    return false;

  auto resultType = cast<RankedTensorType>(dotOp.getType());
  return resultType.getRank() == 2;
}

static std::optional<SplitInfo> handleMayNotExec(Value outerInput,
                                                 Value outerOutput,
                                                 bool useCounterFallback) {
  auto forOp = outerOutput.getDefiningOp<scf::ForOp>();
  if (!forOp)
    return std::nullopt;

  if (!useCounterFallback && !scfMayNotExec(forOp))
    return std::nullopt;

  return SplitInfo{true, outerInput, outerOutput, true,
                   /*shouldAdd=*/true, useCounterFallback};
}

static std::optional<SplitInfo> shouldSplit(triton::DotOp dotOp) {
  if (!verifyDot(dotOp))
    return std::nullopt;

  bool accumulatorLimitedToDot = true;
  bool mayNotExec = false;
  bool mayNotExecWithIf = false;
  Value outerInput = dotOp.getC();
  Value outerOutput = searchInAccumulatorChain(
      dotOp.getResult(), accumulatorLimitedToDot, mayNotExec,
      mayNotExecWithIf, outerInput);

  if (isZeroValue(outerInput)) {
    if (mayNotExec) {
      if (auto splitInfo =
              handleMayNotExec(outerInput, outerOutput, mayNotExecWithIf)) {
        splitInfo->shouldAdd = false;
        return splitInfo;
      }
    }
    return SplitInfo{false, outerInput, outerOutput, false};
  }

  if (canReusePreviousLoopAccumulator(dotOp, outerInput,
                                      mayNotExecWithIf))
    return SplitInfo{false, outerInput, outerOutput, false};

  if (auto definingDot = traceDefiningDot(outerInput)) {
    Operation *dotInCurrentBlock =
        getAncestorInBlock(*definingDot, dotOp->getBlock());
    if (dotInCurrentBlock)
      return SplitInfo{false, outerInput, outerOutput, false};
  }

  if (mayNotExec) {
    if (auto splitInfo =
            handleMayNotExec(outerInput, outerOutput, mayNotExecWithIf))
      return splitInfo;
  }

  return SplitInfo{mayNotExec, outerInput, outerOutput, true};
}

static Value createZeroTensor(OpBuilder &builder, Location loc,
                              RankedTensorType tensorType) {
  Type elementType = tensorType.getElementType();
  Attribute value;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    value = DenseElementsAttr::get(
        tensorType, APFloat::getZero(floatType.getFloatSemantics()));
  } else {
    auto intType = cast<IntegerType>(elementType);
    value =
        DenseElementsAttr::get(tensorType, APInt(intType.getWidth(), 0));
  }
  return builder.create<arith::ConstantOp>(loc, cast<TypedAttr>(value));
}

static Value createCounter(PatternRewriter &rewriter, Location loc,
                           Operation *insertPoint) {
  rewriter.setInsertionPoint(insertPoint);
  auto counterType = MemRefType::get({}, rewriter.getI32Type());
  Value counter = rewriter.create<memref::AllocaOp>(loc, counterType);
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  rewriter.create<memref::StoreOp>(loc, zero, counter, ValueRange{});
  return counter;
}

static void incrementCounter(PatternRewriter &rewriter, Location loc,
                             triton::DotOp dotOp, Value counter) {
  rewriter.setInsertionPointAfter(dotOp);
  Value count = rewriter.create<memref::LoadOp>(loc, counter, ValueRange{});
  Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value next = rewriter.create<arith::AddIOp>(loc, count, one);
  rewriter.create<memref::StoreOp>(loc, next, counter, ValueRange{});
}

static LogicalResult splitDot(triton::DotOp dotOp,
                              PatternRewriter &rewriter,
                              SplitInfo splitInfo) {
  auto outputType = cast<RankedTensorType>(dotOp.getC().getType());
  Type elementType = outputType.getElementType();
  Location loc = dotOp.getLoc();
  Operation *outerDefOp = splitInfo.outerOutValue.getDefiningOp();
  if (!outerDefOp)
    return failure();

  Value counter;
  if (splitInfo.useCounterFallback)
    counter = createCounter(rewriter, loc, outerDefOp);

  rewriter.setInsertionPoint(outerDefOp);
  Value zero = createZeroTensor(rewriter, loc, outputType);
  splitInfo.outerInValue.replaceUsesWithIf(
      zero, [&](OpOperand &operand) {
        return operand.getOwner() == outerDefOp;
      });

  auto forOp =
      dyn_cast_if_present<scf::ForOp>(splitInfo.outerOutValue.getDefiningOp());
  if (!splitInfo.mayNotExec || !forOp) {
    rewriter.setInsertionPoint(dotOp);
    auto newDot = rewriter.create<triton::DotOp>(
        loc, dotOp.getType(), dotOp.getA(), dotOp.getB(), dotOp.getC());
    newDot->setAttrs(dotOp->getAttrs());
    newDot->setAttr(kStandardized, rewriter.getUnitAttr());

    if (splitInfo.outerOutValue == dotOp.getResult())
      splitInfo.outerOutValue = newDot.getResult();
    rewriter.replaceOp(dotOp, newDot.getResult());
  }

  if (splitInfo.useCounterFallback)
    incrementCounter(rewriter, loc, dotOp, counter);

  Value newOutput = splitInfo.outerOutValue;
  rewriter.setInsertionPointAfterValue(splitInfo.outerOutValue);
  Operation *preservedUser = nullptr;

  if (splitInfo.mayNotExec && forOp) {
    Value executed;
    if (splitInfo.useCounterFallback) {
      Value count = rewriter.create<memref::LoadOp>(loc, counter, ValueRange{});
      Value zeroI32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
      executed = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, count, zeroI32);
    } else {
      executed = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, forOp.getUpperBound(),
          forOp.getLowerBound());
    }
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, executed,
        [&](OpBuilder &builder, Location thenLoc) {
          builder.create<scf::YieldOp>(thenLoc, splitInfo.outerOutValue);
        },
        [&](OpBuilder &builder, Location elseLoc) {
          Value elseZero = createZeroTensor(builder, elseLoc, outputType);
          builder.create<scf::YieldOp>(elseLoc, elseZero);
        });
    newOutput = ifOp.getResult(0);
    preservedUser = ifOp;
    forOp->setAttr(kHIVMMatmulLimitedInCube, rewriter.getUnitAttr());
    if (splitInfo.useCounterFallback) {
      if (Operation *parentOp = dotOp->getParentOp())
        parentOp->setAttr(kHIVMMatmulLimitedInCube, rewriter.getUnitAttr());
    }
  }

  if (!splitInfo.shouldAdd) {
    splitInfo.outerOutValue.replaceUsesWithIf(
        newOutput, [&](OpOperand &operand) {
          return !preservedUser || !preservedUser->isAncestor(operand.getOwner());
        });
    return success();
  }

  Operation *addOp;
  if (isa<FloatType>(elementType))
    addOp =
        rewriter.create<arith::AddFOp>(loc, newOutput,
                                      splitInfo.outerInValue);
  else
    addOp =
        rewriter.create<arith::AddIOp>(loc, newOutput,
                                      splitInfo.outerInValue);
  addOp->setAttr(kAddFromDot, rewriter.getUnitAttr());
  if (!preservedUser)
    preservedUser = addOp;

  splitInfo.outerOutValue.replaceUsesWithIf(
      addOp->getResult(0), [&](OpOperand &operand) {
        return !preservedUser->isAncestor(operand.getOwner());
      });
  return success();
}

class NormalizeDotPattern : public OpRewritePattern<triton::DotOp> {
public:
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                PatternRewriter &rewriter) const final {
    auto splitInfo = shouldSplit(dotOp);
    if (!splitInfo)
      return failure();

    dotOp->setAttr(kStandardized, rewriter.getUnitAttr());
    if (splitInfo->shouldSplit)
      return splitDot(dotOp, rewriter, *splitInfo);
    if (splitInfo->mayNotExec)
      dotOp->setAttr(kMayNotExec, rewriter.getUnitAttr());
    return success();
  }
};

static LogicalResult runNormalizeDotImpl(ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<NormalizeDotPattern>(moduleOp.getContext());
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns),
                                          config)))
    return failure();

  bool hasUnhandledMayNotExec = false;
  moduleOp.walk([&](triton::DotOp dotOp) {
    hasUnhandledMayNotExec |= dotOp->hasAttr(kMayNotExec);
  });
  return failure(hasUnhandledMayNotExec);
}

struct NormalizeDotPass
    : public mlir::triton::impl::NormalizeDotBase<NormalizeDotPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    WalkResult walkResult =
        moduleOp.walk([](triton::DotScaledOp) {
          return WalkResult::interrupt();
        });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (failed(runNormalizeDotImpl(moduleOp)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createNormalizeDotPass() {
  return std::make_unique<NormalizeDotPass>();
}
