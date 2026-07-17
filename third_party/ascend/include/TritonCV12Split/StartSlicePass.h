#ifndef TRITON_START_SLICE_PASSES_H
#define TRITON_START_SLICE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createStartSlicePass();

} // namespace triton
} // namespace mlir

#endif // TRITON_START_SLICE_PASSES_H
