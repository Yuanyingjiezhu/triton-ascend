add_triton_library(MLIRTritonNPUUtils
        Utils.cpp
        InterleaveOptimization.cpp
        DimensionAnalyzer.cpp
        DimensionGraphAnalyzer.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRLoopLikeInterface
  TritonAscendIR
  TritonIR
)