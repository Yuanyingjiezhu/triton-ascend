set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonCV12Split)
add_public_tablegen_target(TritonCV12SplitPassIncGen)
