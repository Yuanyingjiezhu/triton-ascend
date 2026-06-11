/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_CV12_SPLIT_PASSES
#define TRITON_CV12_SPLIT_PASSES

include "mlir/Pass/PassBase.td"

def BubbleUpExtractSlice : Pass<"bubble-up-extract-slice", "mlir::ModuleOp"> {
  let summary = "Bubble up tensor.extract_slice operations before their source's defining op";
  let constructor = "triton::createBubbleUpExtractSlicePass()";
  let dependentDialects = ["mlir::tensor::TensorDialect"];
}

def NormalizeDot : Pass<"normalize-dot", "mlir::ModuleOp"> {
  let summary = "Normalize tt.dot by extracting bias operands";
  let constructor = "triton::createNormalizeDotPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "triton::TritonDialect"];
}

def RewriteCommunicationSlice
    : Pass<"rewrite-communication-slice", "mlir::ModuleOp"> {
  let summary = "Rewrite communication tensor slices through workspace memrefs";
  let constructor = "triton::createRewriteCommunicationSlicePass()";
  let dependentDialects = [
    "bishengir::memref_ext::MemRefExtDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def StartSlice : Pass<"start-slice", "mlir::ModuleOp"> {
  let summary = "Slice tt.store, tt.atomic_rmw, and tt.dot operations along axis 0";
  let constructor = "triton::createStartSlicePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::hivm::HIVMDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def TritonCV12Split : Pass<"triton-cv12-split", "mlir::ModuleOp"> {
  let summary = "Try CV1:2 split pipeline and roll back on failure";
  let constructor = "triton::createTritonCV12SplitPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::hivm::HIVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "bishengir::memref_ext::MemRefExtDialect",
    "triton::TritonDialect"
  ];
}

#endif // TRITON_CV12_SPLIT_PASSES
