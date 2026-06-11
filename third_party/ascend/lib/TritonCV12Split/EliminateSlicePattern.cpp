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

namespace mlir::triton::detail {

static bool isMarkedEliminatedSlice(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("to_be_eliminated_slice");
}

static bool isMarkedCVCommunication(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("cv_communication_slice");
}

static void markCVCommunication(PatternRewriter &rewriter, Operation *op) {
  op->setAttr("cv_communication_slice", UnitAttr::get(rewriter.getContext()));
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

    Operation *yieldUser = *insertSliceOp.getResult().getUsers().begin();
    if (!isa<scf::YieldOp>(yieldUser) && !isa<scf::ConditionOp>(yieldUser))
      return rewriter.notifyMatchFailure(insertSliceOp,
                                           "result not used by yield/condition");

    auto *parentOp = yieldUser->getParentOp();
    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    auto whileOp = dyn_cast<scf::WhileOp>(parentOp);

    if (!forOp && !whileOp)
      return rewriter.notifyMatchFailure(insertSliceOp,
                                          "parent is not for/while loop");

    size_t yieldIndex = 0;
    for (size_t i = 0; i < yieldUser->getNumOperands(); ++i) {
      if (yieldUser->getOperand(i) == insertSliceOp.getResult()) {
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

      rewriter.modifyOpInPlace(yieldUser, [&]() {
        yieldUser->setOperand(yieldIndex, sourceValue);
      });

      outerExtractOp->removeAttr("to_be_eliminated_slice");
      markCVCommunication(rewriter, outerExtractOp);

      rewriter.eraseOp(insertSliceOp);
      rewriter.eraseOp(sourceExtractOp);

      return success();
    }

    insertSliceOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, insertSliceOp);

    outerExtractOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, outerExtractOp);

    return success();
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
    if (!forOp)
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "parent is not for loop");

    auto argNumber = blockArg.getArgNumber();
    auto numInductionVars = forOp.getNumInductionVars();
    if (argNumber < numInductionVars)
      return rewriter.notifyMatchFailure(innerExtractOp,
                                          "block argument is induction variable");

    auto initIdx = argNumber - numInductionVars;
    Value initValue = forOp.getInits()[initIdx];

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

      OpOperand &initOperand = forOp.getInitsMutable()[initIdx];
      rewriter.modifyOpInPlace(forOp, [&]() {
        initOperand.set(fullTensor);
      });

      innerExtractOp->removeAttr("to_be_eliminated_slice");
      markCVCommunication(rewriter, innerExtractOp);

      rewriter.eraseOp(outerInsertOp);
      rewriter.eraseOp(outerExtractOp);

      return success();
    }

    outerInsertOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, outerInsertOp);

    innerExtractOp->removeAttr("to_be_eliminated_slice");
    markCVCommunication(rewriter, innerExtractOp);

    return success();
  }
};

void populateEliminateSlicePattern(RewritePatternSet &patterns) {
  patterns.add<EliminateSliceYieldValCVCommunicationPattern>(patterns.getContext());
  patterns.add<EliminateSliceIterArgCVCommunicationPattern>(patterns.getContext());
}

} // namespace mlir::triton::detail