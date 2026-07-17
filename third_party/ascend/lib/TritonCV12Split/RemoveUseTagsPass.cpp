//===- RemoveUseTagsPass.cpp - Remove use analysis attributes ------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "ascend/include/TritonCV12Split/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_REMOVEUSETAGS
#include "ascend/include/TritonCV12Split/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

struct RemoveUseTagsPass
    : public triton::impl::RemoveUseTagsBase<RemoveUseTagsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([](Operation *op) {
      for (StringRef attr :
           {"Undefined", "DataUse", "MetaUse", "MixUse"})
        op->removeAttr(attr);
    });

    PassManager pm(moduleOp.getContext());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(moduleOp)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createRemoveUseTagsPass() {
  return std::make_unique<RemoveUseTagsPass>();
}

} // namespace mlir::triton
