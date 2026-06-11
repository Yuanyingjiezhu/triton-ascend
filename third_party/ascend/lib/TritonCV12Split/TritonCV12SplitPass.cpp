//===- TritonCV12SplitPass.cpp - Try CV1:2 split pipeline -----------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "TritonCV12Split/Passes.h"

#include "TritonCV12Split/BubbleUpExtractSlicePass.h"
#include "TritonCV12Split/NormalizeDotPass.h"
#include "TritonCV12Split/RewriteCommunicationSlicePass.h"
#include "TritonCV12Split/StartSlicePass.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONCV12SPLIT
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-cv12-split"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

static void restoreModule(ModuleOp moduleOp, ModuleOp backupModuleOp) {
  moduleOp->setAttrs(backupModuleOp->getAttrDictionary());
  Region &moduleRegion = moduleOp.getBodyRegion();
  moduleRegion.dropAllReferences();
  moduleRegion.getBlocks().clear();
  moduleRegion.takeBody(backupModuleOp.getBodyRegion());
}

static LogicalResult runSingleModulePass(ModuleOp moduleOp,
                                         std::unique_ptr<Pass> pass) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(std::move(pass));
  return pm.run(moduleOp);
}

static bool hasResidualSliceTags(ModuleOp moduleOp) {
  for (auto &op : *moduleOp.getBody()) {
    if (!isa<tensor::InsertOp, tensor::InsertSliceOp, tensor::ExtractSliceOp>(op))
      continue;
    if (op.hasAttrOfType<UnitAttr>("to_be_bubbled_slice") ||
        op.hasAttrOfType<UnitAttr>("to_be_eliminated_slice") ||
        op.hasAttrOfType<UnitAttr>("cv_communication_slice") ||
        op.hasAttrOfType<UnitAttr>("vv_communication"))
      return true;
  }
  return false;
}

struct TritonCV12SplitPass
    : public mlir::triton::impl::TritonCV12SplitBase<TritonCV12SplitPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    bool hasDot = false;
    moduleOp.walk([&](triton::DotOp) { hasDot = true; });
    if (!hasDot)
      return;

    OwningOpRef<ModuleOp> backupModuleOp(moduleOp.clone());

    if (failed(runSingleModulePass(moduleOp,
                                   triton::createNormalizeDotPass()))) {
      LDBG("TritonNormalizeDot failed, rolling back");
      restoreModule(moduleOp, *backupModuleOp);
      return;
    }

    if (failed(runSingleModulePass(moduleOp, triton::createStartSlicePass()))) {
      LDBG("TritonStartSlice failed, rolling back");
      restoreModule(moduleOp, *backupModuleOp);
      return;
    }

    if (failed(runSingleModulePass(
            moduleOp, triton::createBubbleUpExtractSlicePass()))) {
      LDBG("TritonBubbleUpExtractSlice failed, rolling back");
      restoreModule(moduleOp, *backupModuleOp);
      return;
    }

    if (failed(runSingleModulePass(
            moduleOp, triton::createRewriteCommunicationSlicePass()))) {
      LDBG("TritonRewriteCommunicationSlice failed, rolling back");
      restoreModule(moduleOp, *backupModuleOp);
      return;
    }

    if (hasResidualSliceTags(moduleOp)) {
      LDBG("Residual slice tags found, rolling back");
      restoreModule(moduleOp, *backupModuleOp);
      return;
    }

    moduleOp->setAttr(
        "hivm.disable_auto_tile_and_bind_subblock",
        UnitAttr::get(moduleOp.getContext()));

    llvm::errs() << "[TritonCV12Split] CV1:2 split succeeded for module "
                 << moduleOp.getName().value_or("<unknown>") << "\n";
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonCV12SplitPass() {
  return std::make_unique<TritonCV12SplitPass>();
}

} // namespace triton
} // namespace mlir
