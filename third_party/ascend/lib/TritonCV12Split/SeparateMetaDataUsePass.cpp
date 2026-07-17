//===- SeparateMetaDataUsePass.cpp - Split mixed tensor uses -------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TritonCV12Split/Passes.h"
#include "ascend/include/TritonToLinalg/UseAnalysis.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_SEPARATEMETADATAUSE
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-separate-meta-data-use"

using namespace mlir;
using namespace mlir::dataflow;

namespace {

static triton::UseType mergeUseTypes(triton::UseType lhs,
                                     triton::UseType rhs) {
  if (lhs == triton::UseType::Undefined)
    return rhs;
  if (rhs == triton::UseType::Undefined || lhs == rhs)
    return lhs;
  return triton::UseType::MixUse;
}

/// Return how this particular use consumes its operand.
///
/// Keep this consistent with triton::UseAnalysis::visitOperation. Unlike the
/// old cloning code in runUseAnalysis, this classifies OpOperands rather than
/// Operations, so an operation using the same value in data and metadata
/// positions is handled correctly.
static triton::UseType getDirectUseType(OpOperand &use,
                                       const DataFlowSolver &solver) {
  Operation *user = use.getOwner();
  unsigned operandNumber = use.getOperandNumber();
  auto getResultsUseType = [&]() {
    triton::UseType resultUse = triton::UseType::Undefined;
    for (Value result : user->getResults()) {
      const auto *useInfo = solver.lookupState<triton::UseInfo>(result);
      if (useInfo)
        resultUse = mergeUseTypes(resultUse, useInfo->type);
    }
    return resultUse;
  };

  triton::UseType pointerResultUse = triton::UseType::Undefined;
  if (user->getNumResults() == 1) {
    if (auto shapedType =
            dyn_cast<ShapedType>(user->getResult(0).getType())) {
      if (isa<triton::PointerType>(shapedType.getElementType()))
        pointerResultUse = triton::UseType::MetaUse;
    }
  }

  triton::UseType operationUse =
      llvm::TypeSwitch<Operation *, triton::UseType>(user)
          .Case<triton::LoadOp>(
              [](auto) { return triton::UseType::MetaUse; })
          .Case<triton::StoreOp>([operandNumber](auto) {
            return operandNumber == 1 ? triton::UseType::DataUse
                                      : triton::UseType::MetaUse;
          })
          .Case<triton::ascend::IndirectStoreOp>(
              [operandNumber](auto) {
                return operandNumber == 2 ? triton::UseType::DataUse
                                          : triton::UseType::MetaUse;
              })
          .Case<triton::AtomicRMWOp>([operandNumber](auto) {
            if (operandNumber == 0)
              return triton::UseType::MetaUse;
            return operandNumber == 1 ? triton::UseType::DataUse
                                      : triton::UseType::MetaUse;
          })
          .Case<triton::AtomicCASOp>([operandNumber](auto) {
            return operandNumber == 0 ? triton::UseType::MetaUse
                                      : triton::UseType::DataUse;
          })
          .Case<triton::DotOp>([&, operandNumber](auto dot) {
            if (operandNumber < 2)
              return getResultsUseType();
            auto c = dot.getC();
            if (!c || operandNumber != 2)
              return triton::UseType::Undefined;
            auto splat = c.template getDefiningOp<triton::SplatOp>();
            return splat &&
                           splat.getSrc()
                               .template getDefiningOp<arith::ConstantOp>()
                       ? triton::UseType::MetaUse
                       : triton::UseType::DataUse;
          })
          .Case<triton::PrintOp, triton::AssertOp, triton::ReduceOp,
                tensor::ExtractOp>(
              [](auto) { return triton::UseType::DataUse; })
          .Case<bufferization::MaterializeInDestinationOp, hivm::FixpipeOp,
                hivm::CopyOp>([operandNumber](auto) {
            return operandNumber == 0 ? triton::UseType::DataUse
                                      : triton::UseType::Undefined;
          })
          .Case<hivm::CustomOp, hivm::CustomMacroOp>(
              [](auto) { return triton::UseType::MixUse; })
          .Case<LoopLikeOpInterface, scf::IfOp>(
              [](auto) { return triton::UseType::MixUse; })
          .Default([&](Operation *) { return getResultsUseType(); });

  return mergeUseTypes(pointerResultUse, operationUse);
}

static bool isCloneableMixedTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return false;

  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getNumResults() != 1 ||
      producer->getNumRegions() != 0 || !isMemoryEffectFree(producer))
    return false;

  return !isa<triton::LoadOp, LoopLikeOpInterface, scf::IfOp,
              arith::SelectOp>(producer);
}

/// Split one value and return immediately. The caller must discard the solver
/// after a successful rewrite because all lattice states are then stale.
static bool splitOneMixedTensor(triton::FuncOp func,
                                const DataFlowSolver &solver) {
  Value mixedValue;
  SmallVector<OpOperand *> metaUses;

  // Keep the last candidate in block order. Splitting downstream producers
  // first exposes the complete metadata branch before their shared upstream
  // producers are cloned, avoiding unnecessary duplicate clones.
  func.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      const auto *useInfo = solver.lookupState<triton::UseInfo>(result);
      if (!useInfo || useInfo->type != triton::UseType::MixUse ||
          !isCloneableMixedTensor(result))
        continue;

      SmallVector<OpOperand *> candidateMetaUses;
      bool hasMetaUse = false;
      bool hasDataUse = false;
      bool hasBlockingUse = false;
      for (OpOperand &use : result.getUses()) {
        switch (getDirectUseType(use, solver)) {
        case triton::UseType::MetaUse:
          hasMetaUse = true;
          candidateMetaUses.push_back(&use);
          break;
        case triton::UseType::DataUse:
          hasDataUse = true;
          break;
        case triton::UseType::MixUse:
        case triton::UseType::Undefined:
          hasBlockingUse = true;
          break;
        }
      }

      if (!hasMetaUse || !hasDataUse || hasBlockingUse)
        continue;

      mixedValue = result;
      metaUses = std::move(candidateMetaUses);
    }
  });

  if (!mixedValue)
    return false;

  Operation *producer = mixedValue.getDefiningOp();
  OpBuilder builder(producer);
  Operation *metaClone = builder.clone(*producer);

  producer->removeAttr("MetaUse");
  producer->removeAttr("MixUse");
  producer->setAttr("DataUse", UnitAttr::get(func.getContext()));

  metaClone->removeAttr("DataUse");
  metaClone->removeAttr("MixUse");
  metaClone->setAttr("MetaUse", UnitAttr::get(func.getContext()));

  Value metaValue = metaClone->getResult(0);
  for (OpOperand *use : metaUses)
    use->set(metaValue);

  LLVM_DEBUG(llvm::dbgs()
             << "[" DEBUG_TYPE "] split producer: " << *producer << "\n");
  return true;
}

static StringRef getUseTypeAttrName(triton::UseType useType) {
  switch (useType) {
  case triton::UseType::Undefined:
    return "Undefined";
  case triton::UseType::DataUse:
    return "DataUse";
  case triton::UseType::MetaUse:
    return "MetaUse";
  case triton::UseType::MixUse:
    return "MixUse";
  }
  llvm_unreachable("unknown use type");
}

static void annotateUseTypes(triton::FuncOp func,
                             const DataFlowSolver &solver) {
  MLIRContext *context = func.getContext();
  func.walk([&](Operation *op) {
    for (StringRef attr :
         {"Undefined", "DataUse", "MetaUse", "MixUse"})
      op->removeAttr(attr);

    bool hasTensorResult = false;
    triton::UseType mergedUse = triton::UseType::Undefined;
    for (Value result : op->getResults()) {
      if (!isa<TensorType>(result.getType()))
        continue;
      hasTensorResult = true;
      const auto *useInfo = solver.lookupState<triton::UseInfo>(result);
      if (!useInfo || useInfo->type == triton::UseType::Undefined)
        continue;
      if (mergedUse == triton::UseType::Undefined)
        mergedUse = useInfo->type;
      else if (mergedUse != useInfo->type)
        mergedUse = triton::UseType::MixUse;
    }

    if (!hasTensorResult)
      return;
    op->setAttr(getUseTypeAttrName(mergedUse), UnitAttr::get(context));
  });
}

static LogicalResult separateUsesToFixedPoint(triton::FuncOp func) {
  unsigned iteration = 0;
  constexpr unsigned maxIterations = 10000;

  while (iteration++ < maxIterations) {
    SymbolTableCollection symbolTable;
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<triton::UseAnalysis>(symbolTable,
                                     /*atomicRMWPtrAsMetaUse=*/true);
    if (failed(solver.initializeAndRun(func))) {
      func.emitError("failed to run use analysis");
      return failure();
    }

    if (!splitOneMixedTensor(func, solver)) {
      annotateUseTypes(func, solver);
      return success();
    }
  }

  func.emitError("metadata/data use separation did not reach a fixed point");
  return failure();
}

struct SeparateMetaDataUsePass
    : public triton::impl::SeparateMetaDataUseBase<SeparateMetaDataUsePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult result = module.walk([&](triton::FuncOp func) {
      if (failed(separateUsesToFixedPoint(func)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createSeparateMetaDataUsePass() {
  return std::make_unique<SeparateMetaDataUsePass>();
}

} // namespace mlir::triton
