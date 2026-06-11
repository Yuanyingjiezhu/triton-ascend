//===- NormalizeDotPass.cpp - Normalize tt.dot bias =========================//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//============================================================================//

#include "TritonCV12Split/NormalizeDotPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_NORMALIZEDOT
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

static bool isFromSupportedOp(Value val) {
  if (auto defOp = val.getDefiningOp()) {
    if (isa<triton::LoadOp, arith::ConstantOp, triton::DotOp>(defOp)) {
      return true;
    }
    
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      auto resultIndex = cast<OpResult>(val).getResultNumber();
      Operation *yieldOp = forOp.getBody()->getTerminator();
      if (isa<scf::YieldOp>(yieldOp)) {
        Value yieldValue = yieldOp->getOperand(resultIndex);
        return isFromSupportedOp(yieldValue);
      }
    }
    
    return false;
  }
  return false;
}

struct NormalizeDotPass
    : public mlir::triton::impl::NormalizeDotBase<NormalizeDotPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    moduleOp.walk([&](triton::DotOp dotOp) {
      Value bias = dotOp.getC();
      if (!bias)
        return;

      if (isFromSupportedOp(bias))
        return;

      if (auto iterArg = dyn_cast<BlockArgument>(bias)) {
        auto forOp = dyn_cast<scf::ForOp>(iterArg.getOwner()->getParentOp());
        if (forOp) {
          auto yieldIndex = iterArg.getArgNumber() - forOp.getNumInductionVars();
          if (yieldIndex >= 0 && yieldIndex < static_cast<int>(forOp.getNumResults())) {
            Value initArg = forOp.getInitArgs()[yieldIndex];
            Operation *yieldOp = forOp.getBody()->getTerminator();
            Value yieldValue = yieldOp->getOperand(yieldIndex);
            
            if (isFromSupportedOp(initArg) && isFromSupportedOp(yieldValue))
              return;
          }
        }
      }

      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType)
        return;

      builder.setInsertionPoint(dotOp);
      Location loc = dotOp.getLoc();

      Attribute zeroAttr;
      Type elemType = biasType.getElementType();

      if (isa<FloatType>(elemType)) {
        auto floatType = cast<FloatType>(elemType);
        APFloat zeroVal(floatType.getFloatSemantics(), 0);
        zeroAttr = DenseElementsAttr::get(biasType, zeroVal);
      } else if (isa<IntegerType>(elemType)) {
        auto intType = cast<IntegerType>(elemType);
        APInt zeroVal(intType.getWidth(), 0);
        zeroAttr = DenseElementsAttr::get(biasType, zeroVal);
      } else {
        return;
      }

      auto zeroConst = builder.create<arith::ConstantOp>(loc, cast<TypedAttr>(zeroAttr));

      auto newDot = builder.create<triton::DotOp>(
          loc, dotOp.getType(), dotOp.getA(), dotOp.getB(), zeroConst);

      Value addResult;
      if (isa<FloatType>(elemType)) {
        addResult = builder.create<arith::AddFOp>(loc, newDot, bias);
      } else {
        addResult = builder.create<arith::AddIOp>(loc, newDot, bias);
      }

      dotOp.getResult().replaceAllUsesWith(addResult);
      dotOp.erase();
    });
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createNormalizeDotPass() {
  return std::make_unique<NormalizeDotPass>();
}

} // namespace triton
} // namespace mlir
