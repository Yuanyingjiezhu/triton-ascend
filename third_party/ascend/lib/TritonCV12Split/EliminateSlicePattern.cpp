//===- EliminateSlicePattern.cpp ------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#include "TritonCV12Split/EliminateSlicePattern.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::detail {

static bool isMarkedEliminatedSlice(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("to_be_eliminated_slice");
}

static bool isMarkedCVCommunication(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("cv_communication_slice");
}

static bool isMarkedShouldKeptSlice(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("should_kept_slice");
}

static void markCVCommunication(PatternRewriter &rewriter, Operation *op) {
  op->setAttr("cv_communication_slice", UnitAttr::get(rewriter.getContext()));
}

static bool isMarkedCVCommunicationOrShouldKept(Operation *op) {
  return isMarkedCVCommunication(op) || isMarkedShouldKeptSlice(op);
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

struct IfYieldExtractRewrite {
  scf::YieldOp yieldOp;
  unsigned resultIndex;
  tensor::ExtractSliceOp extractOp;
};

struct IfResultRewrite {
  scf::IfOp ifOp;
  unsigned resultIndex;
};

struct IfSourceRewritePlan {
  SmallVector<IfYieldExtractRewrite> yieldExtracts;
  SmallVector<IfResultRewrite> ifResults;
};

static bool collectIfSourceRewrite(Value value,
                                   tensor::InsertSliceOp insertSliceOp,
                                   IfSourceRewritePlan &plan);

static bool collectIfYieldRewrite(scf::YieldOp yieldOp, unsigned resultIndex,
                                  tensor::InsertSliceOp insertSliceOp,
                                  IfSourceRewritePlan &plan) {
  Value value = yieldOp.getOperand(resultIndex);
  if (auto extractOp = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (!isMarkedCVCommunicationOrShouldKept(extractOp) ||
        !sliceParamsMatch(extractOp, insertSliceOp) ||
        extractOp.getSource().getType() != insertSliceOp.getType())
      return false;

    plan.yieldExtracts.push_back({yieldOp, resultIndex, extractOp});
    return true;
  }

  return collectIfSourceRewrite(value, insertSliceOp, plan);
}

static bool collectIfSourceRewrite(Value value,
                                   tensor::InsertSliceOp insertSliceOp,
                                   IfSourceRewritePlan &plan) {
  auto result = dyn_cast<OpResult>(value);
  auto ifOp = result ? dyn_cast<scf::IfOp>(result.getOwner()) : nullptr;
  if (!ifOp || !value.hasOneUse())
    return false;

  unsigned resultIndex = result.getResultNumber();
  scf::YieldOp thenYieldOp = ifOp.thenYield();
  scf::YieldOp elseYieldOp = ifOp.elseYield();
  if (!thenYieldOp || !elseYieldOp ||
      resultIndex >= thenYieldOp.getNumOperands() ||
      resultIndex >= elseYieldOp.getNumOperands())
    return false;

  if (!collectIfYieldRewrite(thenYieldOp, resultIndex, insertSliceOp, plan) ||
      !collectIfYieldRewrite(elseYieldOp, resultIndex, insertSliceOp, plan))
    return false;

  plan.ifResults.push_back({ifOp, resultIndex});
  return true;
}

struct EliminateSliceYieldValCVCommunicationPattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                  PatternRewriter &rewriter) const final {
    if (!isMarkedEliminatedSlice(insertSliceOp))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "missing to_be_eliminated_slice attr");

    if (!insertSliceOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "result has more than one use");

    auto yieldOp =
        dyn_cast<scf::YieldOp>(*insertSliceOp.getResult().getUsers().begin());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                           "result not used by yield");

    auto *parentOp = yieldOp->getParentOp();
    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    auto whileOp = dyn_cast<scf::WhileOp>(parentOp);

    if (!forOp && !whileOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "parent is not for/while loop");

    size_t yieldIndex = 0;
    for (size_t i = 0; i < yieldOp->getNumOperands(); ++i) {
      if (yieldOp->getOperand(i) == insertSliceOp.getResult()) {
        yieldIndex = i;
        break;
      }
    }

    Value loopResult;
    if (forOp) {
      if (yieldIndex >= forOp.getNumResults())
        return rewriter.notifyMatchFailure(insertSliceOp, "invalid yield index");
      loopResult = forOp.getResult(yieldIndex);
    } else if (whileOp) {
      if (yieldIndex >= whileOp.getNumResults())
        return rewriter.notifyMatchFailure(insertSliceOp, "invalid yield index");
      loopResult = whileOp.getResult(yieldIndex);
    }

    tensor::ExtractSliceOp outerExtractOp = nullptr;
    for (Operation *user : loopResult.getUsers()) {
      auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractOp && isMarkedEliminatedSlice(extractOp)) {
        outerExtractOp = extractOp;
        break;
      }
    }

    if (!outerExtractOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "no to_be_eliminated_slice extract_slice on loop result");

    auto sourceExtractOp =
        insertSliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();

    if (sourceExtractOp && isMarkedCVCommunication(sourceExtractOp)) {
      Value sourceValue = sourceExtractOp.getSource();

      rewriter.modifyOpInPlace(yieldOp, [&]() {
        yieldOp->setOperand(yieldIndex, sourceValue);
      });

      outerExtractOp->removeAttr("to_be_eliminated_slice");
      markCVCommunication(rewriter, outerExtractOp);

      rewriter.eraseOp(insertSliceOp);
      rewriter.eraseOp(sourceExtractOp);

      return success();
    }

    IfSourceRewritePlan plan;
    if (collectIfSourceRewrite(insertSliceOp.getSource(), insertSliceOp, plan)) {
      for (IfYieldExtractRewrite &rewrite : plan.yieldExtracts) {
        rewriter.modifyOpInPlace(rewrite.yieldOp, [&]() {
          rewrite.yieldOp->setOperand(rewrite.resultIndex,
                                      rewrite.extractOp.getSource());
        });
      }

      for (IfResultRewrite &rewrite : plan.ifResults) {
        rewriter.modifyOpInPlace(rewrite.ifOp, [&]() {
          rewrite.ifOp.getResult(rewrite.resultIndex)
              .setType(insertSliceOp.getType());
        });
      }

      rewriter.modifyOpInPlace(yieldOp, [&]() {
        yieldOp->setOperand(yieldIndex, insertSliceOp.getSource());
      });

      outerExtractOp->removeAttr("to_be_eliminated_slice");
      markCVCommunication(rewriter, outerExtractOp);

      rewriter.eraseOp(insertSliceOp);
      llvm::SmallPtrSet<Operation *, 8> visitedExtractOps;
      for (IfYieldExtractRewrite &rewrite : plan.yieldExtracts) {
        Operation *extractOp = rewrite.extractOp;
        if (visitedExtractOps.insert(extractOp).second &&
            extractOp->use_empty())
          rewriter.eraseOp(extractOp);
      }

      return success();
    }

    return rewriter.notifyMatchFailure(insertSliceOp,
                                       "no eliminable yield value slice pair");
  }
};

struct EliminateSliceConditionValCVCommunicationPattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                  PatternRewriter &rewriter) const final {
    if (!isMarkedEliminatedSlice(insertSliceOp))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "missing to_be_eliminated_slice attr");

    if (!insertSliceOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "result has more than one use");

    auto conditionOp =
        dyn_cast<scf::ConditionOp>(*insertSliceOp.getResult().getUsers().begin());
    if (!conditionOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                           "result not used by condition");

    auto whileOp = dyn_cast<scf::WhileOp>(conditionOp->getParentOp());
    if (!whileOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "parent is not while loop");

    std::optional<size_t> operandIndex;
    for (size_t i = 1; i < conditionOp->getNumOperands(); ++i) {
      if (conditionOp->getOperand(i) == insertSliceOp.getResult()) {
        operandIndex = i;
        break;
      }
    }
    if (!operandIndex)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "condition operand not found");

    size_t argIndex = *operandIndex - 1;
    if (argIndex >= whileOp.getAfterArguments().size())
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "invalid condition arg index");

    BlockArgument afterArg = whileOp.getAfterArguments()[argIndex];
    tensor::ExtractSliceOp innerExtractOp = nullptr;
    for (Operation *user : afterArg.getUsers()) {
      auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractOp && isMarkedEliminatedSlice(extractOp)) {
        innerExtractOp = extractOp;
        break;
      }
    }

    if (!innerExtractOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "no to_be_eliminated_slice extract_slice on after arg");

    auto sourceExtractOp =
        insertSliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();

    if (sourceExtractOp && isMarkedCVCommunication(sourceExtractOp)) {
      Value sourceValue = sourceExtractOp.getSource();

      rewriter.modifyOpInPlace(conditionOp, [&]() {
        conditionOp->setOperand(*operandIndex, sourceValue);
      });

      innerExtractOp->removeAttr("to_be_eliminated_slice");
      markCVCommunication(rewriter, innerExtractOp);

      rewriter.eraseOp(insertSliceOp);
      rewriter.eraseOp(sourceExtractOp);

      return success();
    }

    return rewriter.notifyMatchFailure(
        insertSliceOp, "no eliminable condition value slice pair");
  }
};

// 处理 insert{eliminated} -> iter arg -> extract{eliminated} 情况
struct EliminateSliceIterArgCVCommunicationPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp innerExtractOp,
                                PatternRewriter &rewriter) const final {
    if (!isMarkedEliminatedSlice(innerExtractOp))
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "missing to_be_eliminated_slice attr");

    auto blockArg = dyn_cast<BlockArgument>(innerExtractOp.getSource());
    if (!blockArg)
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "source is not a block argument");

    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    auto whileOp = dyn_cast<scf::WhileOp>(parentOp);
    if (!forOp && !whileOp)
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "parent is not for/while loop");

    Value initValue;
    OpOperand *initOperand = nullptr;
    if (forOp) {
      auto argNumber = blockArg.getArgNumber();
      auto numInductionVars = forOp.getNumInductionVars();
      if (argNumber < numInductionVars)
        return rewriter.notifyMatchFailure(innerExtractOp,
                                            "block argument is induction variable");

      auto initIdx = argNumber - numInductionVars;
      initValue = forOp.getInits()[initIdx];
      initOperand = &forOp.getInitsMutable()[initIdx];
    } else {
      if (blockArg.getOwner() != whileOp.getBeforeBody())
        return rewriter.notifyMatchFailure(innerExtractOp,
                                            "block argument is not while before argument");
      auto initIdx = blockArg.getArgNumber();
      if (initIdx >= whileOp.getInits().size())
        return rewriter.notifyMatchFailure(innerExtractOp,
                                            "invalid while init index");
      initValue = whileOp.getInits()[initIdx];
      initOperand = &whileOp.getInitsMutable()[initIdx];
    }

    auto outerInsertOp = initValue.getDefiningOp<tensor::InsertSliceOp>();
    if (!outerInsertOp || !isMarkedEliminatedSlice(outerInsertOp))
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "init is not from eliminated insert_slice");

    if (!outerInsertOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "outer insert_slice has more than one use");

    auto outerExtractOp =
        outerInsertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();

    if (outerExtractOp && isMarkedCVCommunication(outerExtractOp)) {
      if (!outerExtractOp.getResult().hasOneUse())
        return rewriter.notifyMatchFailure(innerExtractOp,
                                            "outer extract_slice has more than one use");

      Value fullTensor = outerExtractOp.getSource();

      rewriter.modifyOpInPlace(parentOp, [&]() {
        initOperand->set(fullTensor);
      });

      innerExtractOp->removeAttr("to_be_eliminated_slice");
      markCVCommunication(rewriter, innerExtractOp);

      rewriter.eraseOp(outerInsertOp);
      rewriter.eraseOp(outerExtractOp);

      return success();
    }

    return rewriter.notifyMatchFailure(innerExtractOp,
                                       "no eliminable iter arg slice pair");
  }
};

struct MarkYieldValCVCommunicationFallbackPattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const final {
    if (!isMarkedEliminatedSlice(insertSliceOp))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "missing to_be_eliminated_slice attr");

    if (!insertSliceOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "result has more than one use");

    auto yieldOp =
        dyn_cast<scf::YieldOp>(*insertSliceOp.getResult().getUsers().begin());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "result not used by yield");

    auto *parentOp = yieldOp->getParentOp();
    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    auto whileOp = dyn_cast<scf::WhileOp>(parentOp);
    if (!forOp && !whileOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "parent is not for/while loop");

    size_t yieldIndex = 0;
    for (size_t i = 0; i < yieldOp->getNumOperands(); ++i) {
      if (yieldOp->getOperand(i) == insertSliceOp.getResult()) {
        yieldIndex = i;
        break;
      }
    }

    Value loopResult;
    if (forOp) {
      if (yieldIndex >= forOp.getNumResults())
        return rewriter.notifyMatchFailure(insertSliceOp, "invalid yield index");
      loopResult = forOp.getResult(yieldIndex);
    } else {
      if (yieldIndex >= whileOp.getNumResults())
        return rewriter.notifyMatchFailure(insertSliceOp, "invalid yield index");
      loopResult = whileOp.getResult(yieldIndex);
    }

    tensor::ExtractSliceOp outerExtractOp = nullptr;
    for (Operation *user : loopResult.getUsers()) {
      auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractOp && isMarkedEliminatedSlice(extractOp)) {
        outerExtractOp = extractOp;
        break;
      }
    }
    if (!outerExtractOp)
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "no to_be_eliminated_slice extract_slice on loop result");

    insertSliceOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, insertSliceOp);

    outerExtractOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, outerExtractOp);

    return success();
  }
};

struct MarkConditionValCVCommunicationFallbackPattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const final {
    if (!isMarkedEliminatedSlice(insertSliceOp))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "missing to_be_eliminated_slice attr");

    if (!insertSliceOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "result has more than one use");

    auto conditionOp =
        dyn_cast<scf::ConditionOp>(*insertSliceOp.getResult().getUsers().begin());
    if (!conditionOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "result not used by condition");

    auto whileOp = dyn_cast<scf::WhileOp>(conditionOp->getParentOp());
    if (!whileOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "parent is not while loop");

    std::optional<size_t> operandIndex;
    for (size_t i = 1; i < conditionOp->getNumOperands(); ++i) {
      if (conditionOp->getOperand(i) == insertSliceOp.getResult()) {
        operandIndex = i;
        break;
      }
    }
    if (!operandIndex)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "condition operand not found");

    size_t argIndex = *operandIndex - 1;
    if (argIndex >= whileOp.getAfterArguments().size())
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "invalid condition arg index");

    BlockArgument afterArg = whileOp.getAfterArguments()[argIndex];
    tensor::ExtractSliceOp innerExtractOp = nullptr;
    for (Operation *user : afterArg.getUsers()) {
      auto extractOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractOp && isMarkedEliminatedSlice(extractOp)) {
        innerExtractOp = extractOp;
        break;
      }
    }
    if (!innerExtractOp)
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "no to_be_eliminated_slice extract_slice on after arg");

    insertSliceOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, insertSliceOp);

    innerExtractOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, innerExtractOp);

    return success();
  }
};

struct MarkIterArgCVCommunicationFallbackPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp innerExtractOp,
                                PatternRewriter &rewriter) const final {
    if (!isMarkedEliminatedSlice(innerExtractOp))
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "missing to_be_eliminated_slice attr");

    auto blockArg = dyn_cast<BlockArgument>(innerExtractOp.getSource());
    if (!blockArg)
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "source is not a block argument");

    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    auto whileOp = dyn_cast<scf::WhileOp>(parentOp);
    if (!forOp && !whileOp)
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "parent is not for/while loop");

    Value initValue;
    if (forOp) {
      auto argNumber = blockArg.getArgNumber();
      auto numInductionVars = forOp.getNumInductionVars();
      if (argNumber < numInductionVars)
        return rewriter.notifyMatchFailure(innerExtractOp,
                                            "block argument is induction variable");

      initValue = forOp.getInits()[argNumber - numInductionVars];
    } else {
      if (blockArg.getOwner() != whileOp.getBeforeBody())
        return rewriter.notifyMatchFailure(
            innerExtractOp, "block argument is not while before argument");
      auto initIdx = blockArg.getArgNumber();
      if (initIdx >= whileOp.getInits().size())
        return rewriter.notifyMatchFailure(innerExtractOp,
                                            "invalid while init index");
      initValue = whileOp.getInits()[initIdx];
    }

    auto outerInsertOp = initValue.getDefiningOp<tensor::InsertSliceOp>();
    if (!outerInsertOp || !isMarkedEliminatedSlice(outerInsertOp))
      return rewriter.notifyMatchFailure(
          innerExtractOp, "init is not from eliminated insert_slice");

    if (!outerInsertOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          innerExtractOp, "outer insert_slice has more than one use");

    outerInsertOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, outerInsertOp);

    innerExtractOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, innerExtractOp);

    return success();
  }
};

void populateEliminateSlicePattern(RewritePatternSet &patterns) {
  patterns.add<EliminateSliceYieldValCVCommunicationPattern>(
      patterns.getContext());
  patterns.add<EliminateSliceConditionValCVCommunicationPattern>(
      patterns.getContext());
  patterns.add<EliminateSliceIterArgCVCommunicationPattern>(
      patterns.getContext());
}

void populateEliminateSliceFallbackPattern(RewritePatternSet &patterns) {
  patterns.add<MarkYieldValCVCommunicationFallbackPattern>(
      patterns.getContext());
  patterns.add<MarkConditionValCVCommunicationFallbackPattern>(
      patterns.getContext());
  patterns.add<MarkIterArgCVCommunicationFallbackPattern>(
      patterns.getContext());
}

} // namespace mlir::triton::detail
