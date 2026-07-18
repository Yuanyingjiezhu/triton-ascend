// NOTE: Snapshot regression coverage for kernels collected under gaoyou/0701.
// NOTE: Regenerate deliberately when --triton-separate-meta-data-use output changes.
// RUN: triton-opt --triton-separate-meta-data-use -split-input-file %s | FileCheck %s --match-full-lines

// -----
// Source: gaoyou/0701/bwd_dqkwg/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_bwd_kernel_dqkwg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: f32, %arg11: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xbf16>
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = tt.get_program_id z : i32
// CHECK-NEXT:     %3 = arith.divsi %2, %c32_i32 : i32
// CHECK-NEXT:     %4 = arith.remsi %2, %c32_i32 : i32
// CHECK-NEXT:     %5 = arith.addi %arg11, %c63_i32 : i32
// CHECK-NEXT:     %6 = arith.divsi %5, %c64_i32 : i32
// CHECK-NEXT:     %7 = arith.muli %3, %6 : i32
// CHECK-NEXT:     %8 = arith.addi %7, %1 : i32
// CHECK-NEXT:     %9 = arith.muli %3, %arg11 : i32
// CHECK-NEXT:     %10 = arith.muli %9, %c32_i32 : i32
// CHECK-NEXT:     %11 = arith.addi %10, %4 : i32
// CHECK-NEXT:     %12 = arith.muli %11, %c128_i32 : i32
// CHECK-NEXT:     %13 = tt.addptr %arg2, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %14 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %15 = arith.muli %8, %c32_i32 : i32
// CHECK-NEXT:     %16 = arith.addi %15, %4 : i32
// CHECK-NEXT:     %17 = arith.extsi %16 : i32 to i64
// CHECK-NEXT:     %18 = arith.muli %17, %c16384_i64 : i64
// CHECK-NEXT:     %19 = tt.addptr %arg4, %18 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %20 = tt.addptr %arg6, %18 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %21 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %22 = tt.addptr %arg1, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %23 = tt.addptr %arg7, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %24 = tt.addptr %arg8, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %25 = arith.muli %0, %arg11 : i32
// CHECK-NEXT:     %26 = arith.muli %25, %c32_i32 : i32
// CHECK-NEXT:     %27 = tt.addptr %arg9, %26 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %28 = arith.muli %1, %c64_i32 : i32
// CHECK-NEXT:     %29 = arith.extsi %arg11 : i32 to i64
// CHECK-NEXT:     %30 = tt.make_tensor_ptr %13, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %31 = tt.make_tensor_ptr %14, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %32 = arith.muli %0, %c128_i32 : i32
// CHECK-NEXT:     %33 = tt.make_tensor_ptr %19, [%c128_i64, %c128_i64], [%c1_i64, %c128_i64], [%c0_i32, %32] {order = array<i32: 0, 1>} : <tensor<128x128xbf16>>
// CHECK-NEXT:     %34 = tt.make_tensor_ptr %20, [%c128_i64, %c128_i64], [%c1_i64, %c128_i64], [%c0_i32, %32] {order = array<i32: 0, 1>} : <tensor<128x128xbf16>>
// CHECK-NEXT:     %35 = tt.load %30 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %36 = tt.load %31 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %37 = tt.load %33 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:     %38 = tt.load %34 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:     %39 = arith.mulf %37, %38 {DataUse} : tensor<128x128xbf16>
// CHECK-NEXT:     %40 = arith.extf %39 {DataUse} : tensor<128x128xbf16> to tensor<128x128xf32>
// CHECK-NEXT:     %41 = tt.reshape %40 allow_reorder {DataUse} : tensor<128x128xf32> -> tensor<16384xf32>
// CHECK-NEXT:     %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %135 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %135 : f32
// CHECK-NEXT:     }) : (tensor<16384xf32>) -> f32
// CHECK-NEXT:     %43 = arith.addf %42, %cst : f32
// CHECK-NEXT:     %44 = tt.trans %35 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:     %45 = tt.dot %36, %44, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %46 = tt.dot %36, %37, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %47 = tt.dot %35, %38, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     gpu.barrier
// CHECK-NEXT:     %48 = tt.make_tensor_ptr %21, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %49 = tt.make_tensor_ptr %22, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %50 = tt.load %48 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %51 = tt.load %49 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %52 = tt.make_tensor_ptr %23, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %53 = tt.make_tensor_ptr %24, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %54 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %55 = tt.splat %28 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %56 = arith.addi %55, %54 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %57 = tt.splat %arg11 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %58 = arith.cmpi slt, %56, %57 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %59 = tt.expand_dims %56 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %60 = tt.expand_dims %56 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %61 = tt.broadcast %59 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %62 = tt.broadcast %60 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %63 = arith.cmpi sge, %61, %62 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %64 = tt.expand_dims %58 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %65 = tt.expand_dims %58 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %66 = tt.broadcast %64 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %67 = tt.broadcast %65 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %68 = arith.andi %66, %67 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %69 = arith.andi %63, %68 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %70 = tt.addptr %arg3, %11 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %71 = tt.addptr %27, %11 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %72 = tt.make_tensor_ptr %70, [%29], [%c32_i64], [%28] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %73 = tt.load %72 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %74 = arith.addi %28, %c64_i32 : i32
// CHECK-NEXT:     %75 = arith.minsi %74, %arg11 : i32
// CHECK-NEXT:     %76 = arith.subi %75, %c1_i32 : i32
// CHECK-NEXT:     %77 = arith.muli %76, %c32_i32 : i32
// CHECK-NEXT:     %78 = tt.addptr %70, %77 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %79 = tt.load %78 : !tt.ptr<bf16>
// CHECK-NEXT:     %80 = arith.extf %79 : bf16 to f32
// CHECK-NEXT:     %81 = math.exp %80 : f32
// CHECK-NEXT:     %82 = arith.mulf %43, %81 : f32
// CHECK-NEXT:     %83 = arith.extf %73 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %84 = math.exp %83 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %85 = tt.expand_dims %84 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %86 = tt.broadcast %85 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %87 = arith.mulf %46, %86 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %88 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %89 = arith.mulf %87, %88 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %90 = arith.subf %cst_1, %73 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:     %91 = tt.splat %79 {DataUse} : bf16 -> tensor<64xbf16>
// CHECK-NEXT:     %92 = arith.addf %90, %91 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:     %93 = arith.extf %92 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %94 = math.exp %93 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %95 = arith.select %58, %94, %cst_0 {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:     %96 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %97 = tt.broadcast %96 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %98 = arith.mulf %47, %97 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %99 = arith.extf %51 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
// CHECK-NEXT:     %100 = arith.mulf %98, %99 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %101 = tt.reshape %100 allow_reorder {DataUse} : tensor<64x128xf32> -> tensor<8192xf32>
// CHECK-NEXT:     %102 = "tt.reduce"(%101) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %135 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %135 : f32
// CHECK-NEXT:     }) : (tensor<8192xf32>) -> f32
// CHECK-NEXT:     %103 = arith.addf %82, %102 : f32
// CHECK-NEXT:     %104 = tt.expand_dims %73 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %105 = tt.expand_dims %73 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:     %106 = tt.broadcast %104 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %107 = tt.broadcast %105 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %108 = arith.subf %106, %107 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:     %109 = arith.extf %108 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %110 = math.exp %109 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %111 = arith.mulf %45, %110 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %112 = arith.select %69, %111, %cst_3 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %113 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     %114 = arith.mulf %112, %113 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %115 = arith.truncf %114 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     %116 = tt.dot %115, %51, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %117 = arith.addf %116, %89 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:     %118 = tt.trans %115 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %119 = tt.dot %118, %50, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %120 = arith.addf %119, %98 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:     %121 = arith.extf %50 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
// CHECK-NEXT:     %122 = arith.mulf %117, %121 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %123 = "tt.reduce"(%122) <{axis = 1 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %135 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %135 : f32
// CHECK-NEXT:     }) {DataUse} : (tensor<64x128xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %124 = arith.mulf %120, %99 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %125 = "tt.reduce"(%124) <{axis = 1 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %135 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %135 : f32
// CHECK-NEXT:     }) {DataUse} : (tensor<64x128xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %126 = arith.subf %123, %125 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %127 = tt.make_tensor_ptr %71, [%29], [%c32_i64], [%28] {order = array<i32: 0>} : <tensor<64xf32>>
// CHECK-NEXT:     %128 = tt.splat %76 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %129 = arith.cmpi slt, %56, %128 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %130 = tt.splat %103 {DataUse} : f32 -> tensor<64xf32>
// CHECK-NEXT:     %131 = arith.addf %126, %130 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %132 = arith.select %129, %126, %131 {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:     %133 = arith.truncf %117 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     tt.store %52, %133 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %134 = arith.truncf %120 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     tt.store %53, %134 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     tt.store %127, %132 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xf32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_bwd_kernel_dqkwg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: f32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64xbf16>
    %c32_i64 = arith.constant 32 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c1_i64 = arith.constant 1 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = arith.remsi %2, %c32_i32 : i32
    %5 = arith.addi %arg11, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %3, %6 : i32
    %8 = arith.addi %7, %1 : i32
    %9 = arith.muli %3, %arg11 : i32
    %10 = arith.muli %9, %c32_i32 : i32
    %11 = arith.addi %10, %4 : i32
    %12 = arith.muli %11, %c128_i32 : i32
    %13 = tt.addptr %arg2, %12 : !tt.ptr<bf16>, i32
    %14 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
    %15 = arith.muli %8, %c32_i32 : i32
    %16 = arith.addi %15, %4 : i32
    %17 = arith.extsi %16 : i32 to i64
    %18 = arith.muli %17, %c16384_i64 : i64
    %19 = tt.addptr %arg4, %18 : !tt.ptr<bf16>, i64
    %20 = tt.addptr %arg6, %18 : !tt.ptr<bf16>, i64
    %21 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
    %22 = tt.addptr %arg1, %12 : !tt.ptr<bf16>, i32
    %23 = tt.addptr %arg7, %12 : !tt.ptr<bf16>, i32
    %24 = tt.addptr %arg8, %12 : !tt.ptr<bf16>, i32
    %25 = arith.muli %0, %arg11 : i32
    %26 = arith.muli %25, %c32_i32 : i32
    %27 = tt.addptr %arg9, %26 : !tt.ptr<f32>, i32
    %28 = arith.muli %1, %c64_i32 : i32
    %29 = arith.extsi %arg11 : i32 to i64
    %30 = tt.make_tensor_ptr %13, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %31 = tt.make_tensor_ptr %14, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %32 = arith.muli %0, %c128_i32 : i32
    %33 = tt.make_tensor_ptr %19, [%c128_i64, %c128_i64], [%c1_i64, %c128_i64], [%c0_i32, %32] {order = array<i32: 0, 1>} : <tensor<128x128xbf16>>
    %34 = tt.make_tensor_ptr %20, [%c128_i64, %c128_i64], [%c1_i64, %c128_i64], [%c0_i32, %32] {order = array<i32: 0, 1>} : <tensor<128x128xbf16>>
    %35 = tt.load %30 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %36 = tt.load %31 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %37 = tt.load %33 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
    %38 = tt.load %34 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
    %39 = arith.mulf %37, %38 : tensor<128x128xbf16>
    %40 = arith.extf %39 : tensor<128x128xbf16> to tensor<128x128xf32>
    %41 = tt.reshape %40 allow_reorder : tensor<128x128xf32> -> tensor<16384xf32>
    %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) : (tensor<16384xf32>) -> f32
    %43 = arith.addf %42, %cst : f32
    %44 = tt.trans %35 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
    %45 = tt.dot %36, %44, %cst_3 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
    %46 = tt.dot %36, %37, %cst_2 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
    %47 = tt.dot %35, %38, %cst_2 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
    gpu.barrier
    %48 = tt.make_tensor_ptr %21, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %49 = tt.make_tensor_ptr %22, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %50 = tt.load %48 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %51 = tt.load %49 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %52 = tt.make_tensor_ptr %23, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %53 = tt.make_tensor_ptr %24, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %54 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %55 = tt.splat %28 : i32 -> tensor<64xi32>
    %56 = arith.addi %55, %54 : tensor<64xi32>
    %57 = tt.splat %arg11 : i32 -> tensor<64xi32>
    %58 = arith.cmpi slt, %56, %57 : tensor<64xi32>
    %59 = tt.expand_dims %56 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %60 = tt.expand_dims %56 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %61 = tt.broadcast %59 : tensor<64x1xi32> -> tensor<64x64xi32>
    %62 = tt.broadcast %60 : tensor<1x64xi32> -> tensor<64x64xi32>
    %63 = arith.cmpi sge, %61, %62 : tensor<64x64xi32>
    %64 = tt.expand_dims %58 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %65 = tt.expand_dims %58 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %66 = tt.broadcast %64 : tensor<64x1xi1> -> tensor<64x64xi1>
    %67 = tt.broadcast %65 : tensor<1x64xi1> -> tensor<64x64xi1>
    %68 = arith.andi %66, %67 : tensor<64x64xi1>
    %69 = arith.andi %63, %68 : tensor<64x64xi1>
    %70 = tt.addptr %arg3, %11 : !tt.ptr<bf16>, i32
    %71 = tt.addptr %27, %11 : !tt.ptr<f32>, i32
    %72 = tt.make_tensor_ptr %70, [%29], [%c32_i64], [%28] {order = array<i32: 0>} : <tensor<64xbf16>>
    %73 = tt.load %72 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %74 = arith.addi %28, %c64_i32 : i32
    %75 = arith.minsi %74, %arg11 : i32
    %76 = arith.subi %75, %c1_i32 : i32
    %77 = arith.muli %76, %c32_i32 : i32
    %78 = tt.addptr %70, %77 : !tt.ptr<bf16>, i32
    %79 = tt.load %78 : !tt.ptr<bf16>
    %80 = arith.extf %79 : bf16 to f32
    %81 = math.exp %80 : f32
    %82 = arith.mulf %43, %81 : f32
    %83 = arith.extf %73 : tensor<64xbf16> to tensor<64xf32>
    %84 = math.exp %83 : tensor<64xf32>
    %85 = tt.expand_dims %84 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %86 = tt.broadcast %85 : tensor<64x1xf32> -> tensor<64x128xf32>
    %87 = arith.mulf %46, %86 : tensor<64x128xf32>
    %88 = tt.splat %arg10 : f32 -> tensor<64x128xf32>
    %89 = arith.mulf %87, %88 : tensor<64x128xf32>
    %90 = arith.subf %cst_1, %73 : tensor<64xbf16>
    %91 = tt.splat %79 : bf16 -> tensor<64xbf16>
    %92 = arith.addf %90, %91 : tensor<64xbf16>
    %93 = arith.extf %92 : tensor<64xbf16> to tensor<64xf32>
    %94 = math.exp %93 : tensor<64xf32>
    %95 = arith.select %58, %94, %cst_0 : tensor<64xi1>, tensor<64xf32>
    %96 = tt.expand_dims %95 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %97 = tt.broadcast %96 : tensor<64x1xf32> -> tensor<64x128xf32>
    %98 = arith.mulf %47, %97 : tensor<64x128xf32>
    %99 = arith.extf %51 : tensor<64x128xbf16> to tensor<64x128xf32>
    %100 = arith.mulf %98, %99 : tensor<64x128xf32>
    %101 = tt.reshape %100 allow_reorder : tensor<64x128xf32> -> tensor<8192xf32>
    %102 = "tt.reduce"(%101) <{axis = 0 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) : (tensor<8192xf32>) -> f32
    %103 = arith.addf %82, %102 : f32
    %104 = tt.expand_dims %73 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %105 = tt.expand_dims %73 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
    %106 = tt.broadcast %104 : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %107 = tt.broadcast %105 : tensor<1x64xbf16> -> tensor<64x64xbf16>
    %108 = arith.subf %106, %107 : tensor<64x64xbf16>
    %109 = arith.extf %108 : tensor<64x64xbf16> to tensor<64x64xf32>
    %110 = math.exp %109 : tensor<64x64xf32>
    %111 = arith.mulf %45, %110 : tensor<64x64xf32>
    %112 = arith.select %69, %111, %cst_3 : tensor<64x64xi1>, tensor<64x64xf32>
    %113 = tt.splat %arg10 : f32 -> tensor<64x64xf32>
    %114 = arith.mulf %112, %113 : tensor<64x64xf32>
    %115 = arith.truncf %114 : tensor<64x64xf32> to tensor<64x64xbf16>
    %116 = tt.dot %115, %51, %cst_2 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %117 = arith.addf %116, %89 {triton_cv12.add_from_dot} : tensor<64x128xf32>
    %118 = tt.trans %115 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
    %119 = tt.dot %118, %50, %cst_2 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %120 = arith.addf %119, %98 {triton_cv12.add_from_dot} : tensor<64x128xf32>
    %121 = arith.extf %50 : tensor<64x128xbf16> to tensor<64x128xf32>
    %122 = arith.mulf %117, %121 : tensor<64x128xf32>
    %123 = "tt.reduce"(%122) <{axis = 1 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) : (tensor<64x128xf32>) -> tensor<64xf32>
    %124 = arith.mulf %120, %99 : tensor<64x128xf32>
    %125 = "tt.reduce"(%124) <{axis = 1 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) : (tensor<64x128xf32>) -> tensor<64xf32>
    %126 = arith.subf %123, %125 : tensor<64xf32>
    %127 = tt.make_tensor_ptr %71, [%29], [%c32_i64], [%28] {order = array<i32: 0>} : <tensor<64xf32>>
    %128 = tt.splat %76 : i32 -> tensor<64xi32>
    %129 = arith.cmpi slt, %56, %128 : tensor<64xi32>
    %130 = tt.splat %103 : f32 -> tensor<64xf32>
    %131 = arith.addf %126, %130 : tensor<64xf32>
    %132 = arith.select %129, %126, %131 : tensor<64xi1>, tensor<64xf32>
    %133 = arith.truncf %117 : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %52, %133 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    %134 = arith.truncf %120 : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %53, %134 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    tt.store %127, %132 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xf32>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/chunk_o/fwd_h_norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_fwd_kernel_h(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xbf16>
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<32> : tensor<64xi32>
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = tt.get_program_id z : i32
// CHECK-NEXT:     %3 = arith.divsi %2, %c32_i32 : i32
// CHECK-NEXT:     %4 = arith.remsi %2, %c32_i32 : i32
// CHECK-NEXT:     %5 = arith.muli %3, %arg4 : i32
// CHECK-NEXT:     %6 = arith.addi %arg4, %c63_i32 : i32
// CHECK-NEXT:     %7 = arith.divsi %6, %c64_i32 : i32
// CHECK-NEXT:     %8 = arith.muli %3, %7 : i32
// CHECK-NEXT:     %9 = arith.muli %5, %c32_i32 : i32
// CHECK-NEXT:     %10 = arith.addi %9, %4 : i32
// CHECK-NEXT:     %11 = arith.muli %10, %c128_i32 : i32
// CHECK-NEXT:     %12 = tt.addptr %arg0, %11 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %13 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:     %14 = arith.extsi %arg4 : i32 to i64
// CHECK-NEXT:     %15 = tt.addptr %arg1, %11 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %16 = arith.muli %1, %c64_i32 : i32
// CHECK-NEXT:     %17 = tt.addptr %arg3, %9 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %18 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %19 = tt.splat %17 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:     %20 = tt.splat %4 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %21 = tt.splat %arg4 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %22 = scf.for %arg5 = %c0_i32 to %7 step %c1_i32 iter_args(%arg6 = %cst_0) -> (tensor<64x64xf32>)  : i32 {
// CHECK-NEXT:       %23 = arith.muli %arg5, %c64_i32 : i32
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %12, [%c128_i64, %14], [%c1_i64, %c4096_i64], [%13, %23] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %25 = tt.make_tensor_ptr %15, [%14, %c128_i64], [%c4096_i64, %c1_i64], [%23, %16] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %26 = arith.addi %8, %arg5 : i32
// CHECK-NEXT:       %27 = arith.muli %26, %c32_i32 : i32
// CHECK-NEXT:       %28 = arith.addi %27, %4 : i32
// CHECK-NEXT:       %29 = arith.extsi %28 : i32 to i64
// CHECK-NEXT:       %30 = arith.muli %29, %c16384_i64 : i64
// CHECK-NEXT:       %31 = tt.addptr %arg2, %30 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %32 = tt.make_tensor_ptr %31, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%13, %16] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %33 = arith.truncf %arg6 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       tt.store %32, %33 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %34 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %35 = tt.load %25 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %36 = arith.addi %arg5, %c1_i32 : i32
// CHECK-NEXT:       %37 = arith.muli %36, %c64_i32 : i32
// CHECK-NEXT:       %38 = arith.minsi %37, %arg4 : i32
// CHECK-NEXT:       %39 = arith.subi %38, %c1_i32 : i32
// CHECK-NEXT:       %40 = arith.muli %39, %c32_i32 : i32
// CHECK-NEXT:       %41 = tt.addptr %17, %40 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %42 = tt.addptr %41, %4 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %43 = tt.load %42 : !tt.ptr<bf16>
// CHECK-NEXT:       %44 = tt.splat %23 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %45 = arith.addi %44, %18 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %46 = arith.muli %45, %cst_1 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %47 = tt.addptr %19, %46 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %48 = tt.addptr %47, %20 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %49 = arith.cmpi slt, %45, %21 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %50 = tt.load %48, %49, %cst {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %51 = arith.extf %43 : bf16 to f32
// CHECK-NEXT:       %52 = math.exp %51 : f32
// CHECK-NEXT:       %53 = tt.splat %52 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:       %54 = arith.mulf %arg6, %53 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %55 = tt.splat %43 {DataUse} : bf16 -> tensor<64xbf16>
// CHECK-NEXT:       %56 = arith.subf %55, %50 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:       %57 = arith.extf %56 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:       %58 = math.exp %57 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %59 = tt.expand_dims %58 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %60 = arith.extf %35 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %61 = tt.broadcast %59 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %62 = arith.mulf %60, %61 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %63 = arith.truncf %62 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %64 = tt.dot %34, %63, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %65 = arith.addf %64, %54 {DataUse, triton_cv12.add_from_dot} : tensor<64x64xf32>
// CHECK-NEXT:       scf.yield %65 : tensor<64x64xf32>
// CHECK-NEXT:     } {Undefined}
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_fwd_kernel_h(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64xbf16>
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_1 = arith.constant dense<32> : tensor<64xi32>
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = arith.remsi %2, %c32_i32 : i32
    %5 = arith.muli %3, %arg4 : i32
    %6 = arith.addi %arg4, %c63_i32 : i32
    %7 = arith.divsi %6, %c64_i32 : i32
    %8 = arith.muli %3, %7 : i32
    %9 = arith.muli %5, %c32_i32 : i32
    %10 = arith.addi %9, %4 : i32
    %11 = arith.muli %10, %c128_i32 : i32
    %12 = tt.addptr %arg0, %11 : !tt.ptr<bf16>, i32
    %13 = arith.muli %0, %c64_i32 : i32
    %14 = arith.extsi %arg4 : i32 to i64
    %15 = tt.addptr %arg1, %11 : !tt.ptr<bf16>, i32
    %16 = arith.muli %1, %c64_i32 : i32
    %17 = tt.addptr %arg3, %9 : !tt.ptr<bf16>, i32
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %19 = tt.splat %17 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %20 = tt.splat %4 : i32 -> tensor<64xi32>
    %21 = tt.splat %arg4 : i32 -> tensor<64xi32>
    %22 = scf.for %arg5 = %c0_i32 to %7 step %c1_i32 iter_args(%arg6 = %cst_0) -> (tensor<64x64xf32>)  : i32 {
      %23 = arith.muli %arg5, %c64_i32 : i32
      %24 = tt.make_tensor_ptr %12, [%c128_i64, %14], [%c1_i64, %c4096_i64], [%13, %23] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
      %25 = tt.make_tensor_ptr %15, [%14, %c128_i64], [%c4096_i64, %c1_i64], [%23, %16] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %26 = arith.addi %8, %arg5 : i32
      %27 = arith.muli %26, %c32_i32 : i32
      %28 = arith.addi %27, %4 : i32
      %29 = arith.extsi %28 : i32 to i64
      %30 = arith.muli %29, %c16384_i64 : i64
      %31 = tt.addptr %arg2, %30 : !tt.ptr<bf16>, i64
      %32 = tt.make_tensor_ptr %31, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%13, %16] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %33 = arith.truncf %arg6 : tensor<64x64xf32> to tensor<64x64xbf16>
      tt.store %32, %33 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
      %34 = tt.load %24 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %35 = tt.load %25 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %36 = arith.addi %arg5, %c1_i32 : i32
      %37 = arith.muli %36, %c64_i32 : i32
      %38 = arith.minsi %37, %arg4 : i32
      %39 = arith.subi %38, %c1_i32 : i32
      %40 = arith.muli %39, %c32_i32 : i32
      %41 = tt.addptr %17, %40 : !tt.ptr<bf16>, i32
      %42 = tt.addptr %41, %4 : !tt.ptr<bf16>, i32
      %43 = tt.load %42 : !tt.ptr<bf16>
      %44 = tt.splat %23 : i32 -> tensor<64xi32>
      %45 = arith.addi %44, %18 : tensor<64xi32>
      %46 = arith.muli %45, %cst_1 : tensor<64xi32>
      %47 = tt.addptr %19, %46 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %48 = tt.addptr %47, %20 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %49 = arith.cmpi slt, %45, %21 : tensor<64xi32>
      %50 = tt.load %48, %49, %cst : tensor<64x!tt.ptr<bf16>>
      %51 = arith.extf %43 : bf16 to f32
      %52 = math.exp %51 : f32
      %53 = tt.splat %52 : f32 -> tensor<64x64xf32>
      %54 = arith.mulf %arg6, %53 : tensor<64x64xf32>
      %55 = tt.splat %43 : bf16 -> tensor<64xbf16>
      %56 = arith.subf %55, %50 : tensor<64xbf16>
      %57 = arith.extf %56 : tensor<64xbf16> to tensor<64xf32>
      %58 = math.exp %57 : tensor<64xf32>
      %59 = tt.expand_dims %58 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %60 = arith.extf %35 : tensor<64x64xbf16> to tensor<64x64xf32>
      %61 = tt.broadcast %59 : tensor<64x1xf32> -> tensor<64x64xf32>
      %62 = arith.mulf %60, %61 : tensor<64x64xf32>
      %63 = arith.truncf %62 : tensor<64x64xf32> to tensor<64x64xbf16>
      %64 = tt.dot %34, %63, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %65 = arith.addf %64, %54 {triton_cv12.add_from_dot} : tensor<64x64xf32>
      scf.yield %65 : tensor<64x64xf32>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/chunk_o/fwd_o_norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_fwd_kernel_o(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = tt.get_program_id z : i32
// CHECK-NEXT:     %3 = arith.divsi %2, %c32_i32 : i32
// CHECK-NEXT:     %4 = arith.remsi %2, %c32_i32 : i32
// CHECK-NEXT:     %5 = arith.addi %arg7, %c63_i32 : i32
// CHECK-NEXT:     %6 = arith.divsi %5, %c64_i32 : i32
// CHECK-NEXT:     %7 = arith.muli %3, %6 : i32
// CHECK-NEXT:     %8 = arith.addi %7, %1 : i32
// CHECK-NEXT:     %9 = arith.muli %3, %arg7 : i32
// CHECK-NEXT:     %10 = arith.muli %9, %c32_i32 : i32
// CHECK-NEXT:     %11 = arith.addi %10, %4 : i32
// CHECK-NEXT:     %12 = arith.muli %11, %c128_i32 : i32
// CHECK-NEXT:     %13 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %14 = tt.addptr %arg1, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %15 = tt.addptr %arg2, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %16 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %17 = arith.muli %8, %c32_i32 : i32
// CHECK-NEXT:     %18 = arith.addi %17, %4 : i32
// CHECK-NEXT:     %19 = arith.extsi %18 : i32 to i64
// CHECK-NEXT:     %20 = arith.muli %19, %c16384_i64 : i64
// CHECK-NEXT:     %21 = tt.addptr %arg3, %20 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %22 = arith.muli %1, %c64_i32 : i32
// CHECK-NEXT:     %23 = arith.extsi %arg7 : i32 to i64
// CHECK-NEXT:     %24 = tt.make_tensor_ptr %13, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %25 = tt.make_tensor_ptr %14, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c0_i32, %22] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:     %26 = arith.muli %0, %c128_i32 : i32
// CHECK-NEXT:     %27 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %26] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:     %28 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %29 = tt.load %25 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:     %30 = tt.load %27 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:     %31 = tt.dot %28, %30, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %32 = tt.dot %28, %29, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %33 = tt.addptr %arg4, %11 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %34 = tt.make_tensor_ptr %33, [%23], [%c32_i64], [%22] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %35 = tt.load %34 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %36 = arith.extf %35 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %37 = math.exp %36 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %38 = tt.expand_dims %37 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %39 = tt.broadcast %38 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %40 = arith.mulf %31, %39 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %41 = tt.expand_dims %35 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %42 = tt.expand_dims %35 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:     %43 = tt.broadcast %41 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %44 = tt.broadcast %42 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %45 = arith.subf %43, %44 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:     %46 = arith.extf %45 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %47 = math.exp %46 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %48 = arith.mulf %32, %47 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %49 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %50 = tt.splat %22 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %51 = arith.addi %50, %49 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %52 = tt.splat %arg7 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %53 = arith.cmpi slt, %51, %52 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %54 = tt.expand_dims %51 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %55 = tt.expand_dims %51 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %56 = tt.broadcast %54 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %57 = tt.broadcast %55 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %58 = arith.cmpi sge, %56, %57 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %59 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %60 = tt.expand_dims %53 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %61 = tt.broadcast %59 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %62 = tt.broadcast %60 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %63 = arith.andi %61, %62 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %64 = arith.andi %58, %63 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %65 = arith.select %64, %48, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %66 = tt.make_tensor_ptr %15, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %26] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %67 = tt.make_tensor_ptr %16, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %26] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %68 = tt.load %66 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %69 = tt.splat %arg6 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %70 = arith.mulf %40, %69 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %71 = arith.truncf %65 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     %72 = tt.dot %71, %68, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %73 = arith.mulf %72, %69 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %74 = arith.addf %70, %73 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %75 = arith.truncf %74 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     tt.store %67, %75 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_fwd_kernel_o(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %c1_i64 = arith.constant 1 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c64_i32 = arith.constant 64 : i32
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = arith.remsi %2, %c32_i32 : i32
    %5 = arith.addi %arg7, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %3, %6 : i32
    %8 = arith.addi %7, %1 : i32
    %9 = arith.muli %3, %arg7 : i32
    %10 = arith.muli %9, %c32_i32 : i32
    %11 = arith.addi %10, %4 : i32
    %12 = arith.muli %11, %c128_i32 : i32
    %13 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
    %14 = tt.addptr %arg1, %12 : !tt.ptr<bf16>, i32
    %15 = tt.addptr %arg2, %12 : !tt.ptr<bf16>, i32
    %16 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
    %17 = arith.muli %8, %c32_i32 : i32
    %18 = arith.addi %17, %4 : i32
    %19 = arith.extsi %18 : i32 to i64
    %20 = arith.muli %19, %c16384_i64 : i64
    %21 = tt.addptr %arg3, %20 : !tt.ptr<bf16>, i64
    %22 = arith.muli %1, %c64_i32 : i32
    %23 = arith.extsi %arg7 : i32 to i64
    %24 = tt.make_tensor_ptr %13, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %25 = tt.make_tensor_ptr %14, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c0_i32, %22] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
    %26 = arith.muli %0, %c128_i32 : i32
    %27 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %26] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
    %28 = tt.load %24 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %29 = tt.load %25 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
    %30 = tt.load %27 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
    %31 = tt.dot %28, %30, %cst_0 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
    %32 = tt.dot %28, %29, %cst {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
    %33 = tt.addptr %arg4, %11 : !tt.ptr<bf16>, i32
    %34 = tt.make_tensor_ptr %33, [%23], [%c32_i64], [%22] {order = array<i32: 0>} : <tensor<64xbf16>>
    %35 = tt.load %34 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %36 = arith.extf %35 : tensor<64xbf16> to tensor<64xf32>
    %37 = math.exp %36 : tensor<64xf32>
    %38 = tt.expand_dims %37 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %39 = tt.broadcast %38 : tensor<64x1xf32> -> tensor<64x128xf32>
    %40 = arith.mulf %31, %39 : tensor<64x128xf32>
    %41 = tt.expand_dims %35 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %42 = tt.expand_dims %35 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
    %43 = tt.broadcast %41 : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %44 = tt.broadcast %42 : tensor<1x64xbf16> -> tensor<64x64xbf16>
    %45 = arith.subf %43, %44 : tensor<64x64xbf16>
    %46 = arith.extf %45 : tensor<64x64xbf16> to tensor<64x64xf32>
    %47 = math.exp %46 : tensor<64x64xf32>
    %48 = arith.mulf %32, %47 : tensor<64x64xf32>
    %49 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %50 = tt.splat %22 : i32 -> tensor<64xi32>
    %51 = arith.addi %50, %49 : tensor<64xi32>
    %52 = tt.splat %arg7 : i32 -> tensor<64xi32>
    %53 = arith.cmpi slt, %51, %52 : tensor<64xi32>
    %54 = tt.expand_dims %51 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %55 = tt.expand_dims %51 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %56 = tt.broadcast %54 : tensor<64x1xi32> -> tensor<64x64xi32>
    %57 = tt.broadcast %55 : tensor<1x64xi32> -> tensor<64x64xi32>
    %58 = arith.cmpi sge, %56, %57 : tensor<64x64xi32>
    %59 = tt.expand_dims %53 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %60 = tt.expand_dims %53 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %61 = tt.broadcast %59 : tensor<64x1xi1> -> tensor<64x64xi1>
    %62 = tt.broadcast %60 : tensor<1x64xi1> -> tensor<64x64xi1>
    %63 = arith.andi %61, %62 : tensor<64x64xi1>
    %64 = arith.andi %58, %63 : tensor<64x64xi1>
    %65 = arith.select %64, %48, %cst : tensor<64x64xi1>, tensor<64x64xf32>
    %66 = tt.make_tensor_ptr %15, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %26] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %67 = tt.make_tensor_ptr %16, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %26] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %68 = tt.load %66 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %69 = tt.splat %arg6 : f32 -> tensor<64x128xf32>
    %70 = arith.mulf %40, %69 : tensor<64x128xf32>
    %71 = arith.truncf %65 : tensor<64x64xf32> to tensor<64x64xbf16>
    %72 = tt.dot %71, %68, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %73 = arith.mulf %72, %69 : tensor<64x128xf32>
    %74 = arith.addf %70, %73 : tensor<64x128xf32>
    %75 = arith.truncf %74 : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %67, %75 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/delta_rule_bwd_dhu/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %c524288_i64 = arith.constant 524288 : i64
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x32xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x32xf32>
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = arith.divsi %1, %c32_i32 : i32
// CHECK-NEXT:     %3 = arith.remsi %1, %c32_i32 : i32
// CHECK-NEXT:     %4 = arith.muli %2, %arg9 : i32
// CHECK-NEXT:     %5 = arith.addi %arg9, %c63_i32 : i32
// CHECK-NEXT:     %6 = arith.divsi %5, %c64_i32 : i32
// CHECK-NEXT:     %7 = arith.muli %2, %6 : i32
// CHECK-NEXT:     %8 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:     %9 = arith.addi %8, %3 : i32
// CHECK-NEXT:     %10 = arith.extsi %9 : i32 to i64
// CHECK-NEXT:     %11 = arith.muli %10, %c128_i64 : i64
// CHECK-NEXT:     %12 = tt.addptr %arg0, %11 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %13 = tt.addptr %arg1, %11 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %14 = tt.addptr %arg2, %11 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %15 = tt.addptr %arg4, %11 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %16 = tt.addptr %arg6, %11 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %17 = tt.addptr %arg7, %11 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %18 = arith.muli %7, %c32_i32 : i32
// CHECK-NEXT:     %19 = arith.addi %18, %3 : i32
// CHECK-NEXT:     %20 = arith.extsi %19 : i32 to i64
// CHECK-NEXT:     %21 = arith.muli %20, %c16384_i64 : i64
// CHECK-NEXT:     %22 = arith.subi %6, %c1_i32 : i32
// CHECK-NEXT:     %23 = arith.muli %0, %c32_i32 : i32
// CHECK-NEXT:     %24 = tt.addptr %arg3, %8 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %25 = tt.addptr %24, %3 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %26 = arith.extsi %arg9 : i32 to i64
// CHECK-NEXT:     %27 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %28 = tt.splat %arg9 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %29 = tt.splat %arg8 {DataUse} : f32 -> tensor<128x32xf32>
// CHECK-NEXT:     %30 = scf.for %arg10 = %c-1_i32 to %22 step %c1_i32 iter_args(%arg11 = %cst) -> (tensor<128x32xf32>)  : i32 {
// CHECK-NEXT:       %31 = arith.subi %22, %arg10 : i32
// CHECK-NEXT:       %32 = arith.addi %31, %c-1_i32 : i32
// CHECK-NEXT:       %33 = arith.extsi %32 : i32 to i64
// CHECK-NEXT:       %34 = arith.muli %33, %c524288_i64 : i64
// CHECK-NEXT:       %35 = arith.addi %21, %34 : i64
// CHECK-NEXT:       %36 = tt.addptr %arg5, %35 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %37 = tt.make_tensor_ptr %36, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %23] {order = array<i32: 1, 0>} : <tensor<128x32xbf16>>
// CHECK-NEXT:       %38 = arith.truncf %arg11 {DataUse} : tensor<128x32xf32> to tensor<128x32xbf16>
// CHECK-NEXT:       tt.store %37, %38 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x32xbf16>>
// CHECK-NEXT:       %39 = arith.muli %31, %c64_i32 : i32
// CHECK-NEXT:       %40 = arith.minsi %39, %arg9 : i32
// CHECK-NEXT:       %41 = arith.subi %40, %c1_i32 : i32
// CHECK-NEXT:       %42 = arith.addi %4, %41 : i32
// CHECK-NEXT:       %43 = arith.muli %42, %c32_i32 : i32
// CHECK-NEXT:       %44 = tt.addptr %arg3, %43 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %45 = tt.addptr %44, %3 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %46 = tt.load %45 : !tt.ptr<f32>
// CHECK-NEXT:       %47 = arith.muli %32, %c64_i32 : i32
// CHECK-NEXT:       %48 = tt.make_tensor_ptr %25, [%26], [%c32_i64], [%47] {order = array<i32: 0>} : <tensor<64xf32>>
// CHECK-NEXT:       %49 = tt.load %48 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
// CHECK-NEXT:       %50 = math.exp %46 : f32
// CHECK-NEXT:       %51 = math.exp %49 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %52 = tt.make_tensor_ptr %16, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %53 = tt.make_tensor_ptr %17, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %54 = tt.make_tensor_ptr %15, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %55 = tt.load %54 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %56 = tt.make_tensor_ptr %13, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %57 = tt.load %56 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %58 = tt.dot %57, %38, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %59 = tt.splat %47 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %60 = arith.addi %59, %27 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %61 = arith.cmpi slt, %60, %28 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %62 = tt.splat %46 {DataUse} : f32 -> tensor<64xf32>
// CHECK-NEXT:       %63 = arith.subf %62, %49 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %64 = math.exp %63 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %65 = arith.select %61, %64, %cst_0 {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:       %66 = tt.expand_dims %65 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %67 = tt.broadcast %66 {DataUse} : tensor<64x1xf32> -> tensor<64x32xf32>
// CHECK-NEXT:       %68 = arith.mulf %58, %67 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %69 = tt.load %52 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %70 = arith.extf %69 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
// CHECK-NEXT:       %71 = arith.addf %68, %70 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %72 = arith.truncf %71 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       tt.store %53, %72 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %73 = tt.make_tensor_ptr %14, [%c128_i64, %26], [%c1_i64, %c4096_i64], [%c0_i32, %47] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:       %74 = tt.make_tensor_ptr %12, [%c128_i64, %26], [%c1_i64, %c4096_i64], [%c0_i32, %47] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:       %75 = tt.load %73 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:       %76 = tt.load %74 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:       %77 = tt.splat %50 {DataUse} : f32 -> tensor<128x32xf32>
// CHECK-NEXT:       %78 = arith.mulf %arg11, %77 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       %79 = tt.expand_dims %51 {DataUse, axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
// CHECK-NEXT:       %80 = arith.extf %76 {DataUse} : tensor<128x64xbf16> to tensor<128x64xf32>
// CHECK-NEXT:       %81 = tt.broadcast %79 {DataUse} : tensor<1x64xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %82 = arith.mulf %80, %81 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:       %83 = arith.extf %55 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
// CHECK-NEXT:       %84 = tt.dot %82, %83, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf32> * tensor<64x32xf32> -> tensor<128x32xf32>
// CHECK-NEXT:       %85 = arith.mulf %84, %29 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       %86 = tt.dot %75, %72, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x32xbf16> -> tensor<128x32xf32>
// CHECK-NEXT:       %87 = arith.subf %85, %86 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       %88 = arith.addf %78, %87 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       scf.yield %88 : tensor<128x32xf32>
// CHECK-NEXT:     } {Undefined}
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c524288_i64 = arith.constant 524288 : i64
    %c-1_i32 = arith.constant -1 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
    %c4096_i64 = arith.constant 4096 : i64
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.remsi %1, %c32_i32 : i32
    %4 = arith.muli %2, %arg9 : i32
    %5 = arith.addi %arg9, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %6 : i32
    %8 = arith.muli %4, %c32_i32 : i32
    %9 = arith.addi %8, %3 : i32
    %10 = arith.extsi %9 : i32 to i64
    %11 = arith.muli %10, %c128_i64 : i64
    %12 = tt.addptr %arg0, %11 : !tt.ptr<bf16>, i64
    %13 = tt.addptr %arg1, %11 : !tt.ptr<bf16>, i64
    %14 = tt.addptr %arg2, %11 : !tt.ptr<bf16>, i64
    %15 = tt.addptr %arg4, %11 : !tt.ptr<bf16>, i64
    %16 = tt.addptr %arg6, %11 : !tt.ptr<bf16>, i64
    %17 = tt.addptr %arg7, %11 : !tt.ptr<bf16>, i64
    %18 = arith.muli %7, %c32_i32 : i32
    %19 = arith.addi %18, %3 : i32
    %20 = arith.extsi %19 : i32 to i64
    %21 = arith.muli %20, %c16384_i64 : i64
    %22 = arith.subi %6, %c1_i32 : i32
    %23 = arith.muli %0, %c32_i32 : i32
    %24 = tt.addptr %arg3, %8 : !tt.ptr<f32>, i32
    %25 = tt.addptr %24, %3 : !tt.ptr<f32>, i32
    %26 = arith.extsi %arg9 : i32 to i64
    %27 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %28 = tt.splat %arg9 : i32 -> tensor<64xi32>
    %29 = tt.splat %arg8 : f32 -> tensor<128x32xf32>
    %30 = scf.for %arg10 = %c-1_i32 to %22 step %c1_i32 iter_args(%arg11 = %cst) -> (tensor<128x32xf32>)  : i32 {
      %31 = arith.subi %22, %arg10 : i32
      %32 = arith.addi %31, %c-1_i32 : i32
      %33 = arith.extsi %32 : i32 to i64
      %34 = arith.muli %33, %c524288_i64 : i64
      %35 = arith.addi %21, %34 : i64
      %36 = tt.addptr %arg5, %35 : !tt.ptr<bf16>, i64
      %37 = tt.make_tensor_ptr %36, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %23] {order = array<i32: 1, 0>} : <tensor<128x32xbf16>>
      %38 = arith.truncf %arg11 : tensor<128x32xf32> to tensor<128x32xbf16>
      tt.store %37, %38 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x32xbf16>>
      %39 = arith.muli %31, %c64_i32 : i32
      %40 = arith.minsi %39, %arg9 : i32
      %41 = arith.subi %40, %c1_i32 : i32
      %42 = arith.addi %4, %41 : i32
      %43 = arith.muli %42, %c32_i32 : i32
      %44 = tt.addptr %arg3, %43 : !tt.ptr<f32>, i32
      %45 = tt.addptr %44, %3 : !tt.ptr<f32>, i32
      %46 = tt.load %45 : !tt.ptr<f32>
      %47 = arith.muli %32, %c64_i32 : i32
      %48 = tt.make_tensor_ptr %25, [%26], [%c32_i64], [%47] {order = array<i32: 0>} : <tensor<64xf32>>
      %49 = tt.load %48 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
      %50 = math.exp %46 : f32
      %51 = math.exp %49 : tensor<64xf32>
      %52 = tt.make_tensor_ptr %16, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %53 = tt.make_tensor_ptr %17, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %54 = tt.make_tensor_ptr %15, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %55 = tt.load %54 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
      %56 = tt.make_tensor_ptr %13, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %57 = tt.load %56 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
      %58 = tt.dot %57, %38, %cst_1 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32>
      %59 = tt.splat %47 : i32 -> tensor<64xi32>
      %60 = arith.addi %59, %27 : tensor<64xi32>
      %61 = arith.cmpi slt, %60, %28 : tensor<64xi32>
      %62 = tt.splat %46 : f32 -> tensor<64xf32>
      %63 = arith.subf %62, %49 : tensor<64xf32>
      %64 = math.exp %63 : tensor<64xf32>
      %65 = arith.select %61, %64, %cst_0 : tensor<64xi1>, tensor<64xf32>
      %66 = tt.expand_dims %65 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %67 = tt.broadcast %66 : tensor<64x1xf32> -> tensor<64x32xf32>
      %68 = arith.mulf %58, %67 : tensor<64x32xf32>
      %69 = tt.load %52 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
      %70 = arith.extf %69 : tensor<64x32xbf16> to tensor<64x32xf32>
      %71 = arith.addf %68, %70 : tensor<64x32xf32>
      %72 = arith.truncf %71 : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %53, %72 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %73 = tt.make_tensor_ptr %14, [%c128_i64, %26], [%c1_i64, %c4096_i64], [%c0_i32, %47] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %74 = tt.make_tensor_ptr %12, [%c128_i64, %26], [%c1_i64, %c4096_i64], [%c0_i32, %47] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %75 = tt.load %73 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %76 = tt.load %74 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %77 = tt.splat %50 : f32 -> tensor<128x32xf32>
      %78 = arith.mulf %arg11, %77 : tensor<128x32xf32>
      %79 = tt.expand_dims %51 {axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
      %80 = arith.extf %76 : tensor<128x64xbf16> to tensor<128x64xf32>
      %81 = tt.broadcast %79 : tensor<1x64xf32> -> tensor<128x64xf32>
      %82 = arith.mulf %80, %81 : tensor<128x64xf32>
      %83 = arith.extf %55 : tensor<64x32xbf16> to tensor<64x32xf32>
      %84 = tt.dot %82, %83, %cst {triton_cv12.normalized_dot} : tensor<128x64xf32> * tensor<64x32xf32> -> tensor<128x32xf32>
      %85 = arith.mulf %84, %29 : tensor<128x32xf32>
      %86 = tt.dot %75, %72, %cst {triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x32xbf16> -> tensor<128x32xf32>
      %87 = arith.subf %85, %86 : tensor<128x32xf32>
      %88 = arith.addf %78, %87 : tensor<128x32xf32>
      scf.yield %88 : tensor<128x32xf32>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/delta_rule_fwd_h/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_gated_delta_rule_fwd_kernel_h_blockdim64(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %c524288_i64 = arith.constant 524288 : i64
// CHECK-NEXT:     %c16384_i32 = arith.constant 16384 : i32
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x32xf32>
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = arith.divsi %1, %c32_i32 : i32
// CHECK-NEXT:     %3 = arith.remsi %1, %c32_i32 : i32
// CHECK-NEXT:     %4 = arith.muli %2, %arg7 : i32
// CHECK-NEXT:     %5 = arith.addi %arg7, %c63_i32 : i32
// CHECK-NEXT:     %6 = arith.divsi %5, %c64_i32 : i32
// CHECK-NEXT:     %7 = arith.muli %2, %6 : i32
// CHECK-NEXT:     %8 = arith.muli %7, %c32_i32 : i32
// CHECK-NEXT:     %9 = arith.addi %8, %3 : i32
// CHECK-NEXT:     %10 = arith.extsi %9 : i32 to i64
// CHECK-NEXT:     %11 = arith.muli %10, %c16384_i64 : i64
// CHECK-NEXT:     %12 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:     %13 = arith.addi %12, %3 : i32
// CHECK-NEXT:     %14 = arith.extsi %13 : i32 to i64
// CHECK-NEXT:     %15 = arith.muli %14, %c128_i64 : i64
// CHECK-NEXT:     %16 = tt.addptr %arg1, %15 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %17 = tt.addptr %arg0, %15 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %18 = tt.addptr %arg2, %15 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %19 = tt.addptr %arg3, %15 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %20 = arith.muli %1, %c16384_i32 : i32
// CHECK-NEXT:     %21 = tt.addptr %arg6, %20 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %22 = arith.muli %0, %c32_i32 : i32
// CHECK-NEXT:     %23 = arith.extsi %arg7 : i32 to i64
// CHECK-NEXT:     %24 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %25 = tt.splat %arg7 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %26 = tt.addptr %arg4, %14 : !tt.ptr<f32>, i64
// CHECK-NEXT:     %27:2 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %cst_0, %arg10 = %cst_0) -> (tensor<64x32xf32>, tensor<64x32xf32>)  : i32 {
// CHECK-NEXT:       %31 = arith.extsi %arg8 : i32 to i64
// CHECK-NEXT:       %32 = arith.muli %31, %c524288_i64 : i64
// CHECK-NEXT:       %33 = arith.addi %11, %32 : i64
// CHECK-NEXT:       %34 = tt.addptr %arg5, %33 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %35 = tt.make_tensor_ptr %34, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %36 = arith.truncf %arg9 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       tt.store %35, %36 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %37 = tt.make_tensor_ptr %34, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c64_i32, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %38 = arith.truncf %arg10 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       tt.store %37, %38 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %39 = arith.muli %arg8, %c64_i32 : i32
// CHECK-NEXT:       %40 = tt.make_tensor_ptr %18, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %41 = tt.load %40 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %42 = tt.dot %41, %36, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %43 = tt.make_tensor_ptr %18, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %c64_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %44 = tt.load %43 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %45 = tt.dot %44, %38, %42 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %46 = tt.make_tensor_ptr %16, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %47 = tt.load %46 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %48 = arith.extf %47 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
// CHECK-NEXT:       %49 = arith.subf %48, %45 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %50 = tt.make_tensor_ptr %19, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %51 = arith.truncf %49 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       tt.store %50, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %52 = arith.addi %arg8, %c1_i32 : i32
// CHECK-NEXT:       %53 = arith.muli %52, %c64_i32 : i32
// CHECK-NEXT:       %54 = arith.minsi %53, %arg7 : i32
// CHECK-NEXT:       %55 = arith.subi %54, %c1_i32 : i32
// CHECK-NEXT:       %56 = tt.splat %39 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %57 = arith.addi %56, %24 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %58 = arith.cmpi slt, %57, %25 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %59 = arith.muli %55, %c32_i32 : i32
// CHECK-NEXT:       %60 = arith.addi %12, %59 : i32
// CHECK-NEXT:       %61 = arith.addi %60, %3 : i32
// CHECK-NEXT:       %62 = arith.extsi %61 : i32 to i64
// CHECK-NEXT:       %63 = tt.addptr %arg4, %62 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %64 = tt.load %63 : !tt.ptr<f32>
// CHECK-NEXT:       %65 = tt.make_tensor_ptr %26, [%23], [%c32_i64], [%39] {order = array<i32: 0>} : <tensor<64xf32>>
// CHECK-NEXT:       %66 = tt.load %65 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
// CHECK-NEXT:       %67 = tt.splat %64 {DataUse} : f32 -> tensor<64xf32>
// CHECK-NEXT:       %68 = arith.subf %67, %66 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %69 = math.exp2 %68 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %70 = arith.select %58, %69, %cst {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:       %71 = tt.expand_dims %70 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %72 = tt.broadcast %71 {DataUse} : tensor<64x1xf32> -> tensor<64x32xf32>
// CHECK-NEXT:       %73 = arith.mulf %49, %72 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %74 = math.exp2 %64 : f32
// CHECK-NEXT:       %75 = tt.splat %74 {DataUse} : f32 -> tensor<64x32xf32>
// CHECK-NEXT:       %76 = arith.mulf %arg9, %75 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %77 = arith.mulf %arg10, %75 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %78 = arith.truncf %73 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       %79 = tt.make_tensor_ptr %17, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c0_i32, %39] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %80 = tt.load %79 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %81 = tt.dot %80, %78, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %82 = arith.addf %81, %76 {DataUse, triton_cv12.add_from_dot} : tensor<64x32xf32>
// CHECK-NEXT:       %83 = tt.make_tensor_ptr %17, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c64_i32, %39] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %84 = tt.load %83 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %85 = tt.dot %84, %78, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %86 = arith.addf %85, %77 {DataUse, triton_cv12.add_from_dot} : tensor<64x32xf32>
// CHECK-NEXT:       scf.yield %82, %86 : tensor<64x32xf32>, tensor<64x32xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %28 = arith.muli %0, %c32_i32 : i32
// CHECK-NEXT:     %29 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %28] {order = array<i32: 1, 0>} : <tensor<64x32xf32>>
// CHECK-NEXT:     tt.store %29, %27#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf32>>
// CHECK-NEXT:     %30 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c64_i32, %28] {order = array<i32: 1, 0>} : <tensor<64x32xf32>>
// CHECK-NEXT:     tt.store %30, %27#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_gated_delta_rule_fwd_kernel_h_blockdim64(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c524288_i64 = arith.constant 524288 : i64
    %c16384_i32 = arith.constant 16384 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
    %c4096_i64 = arith.constant 4096 : i64
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i64 = arith.constant 128 : i64
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.remsi %1, %c32_i32 : i32
    %4 = arith.muli %2, %arg7 : i32
    %5 = arith.addi %arg7, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %6 : i32
    %8 = arith.muli %7, %c32_i32 : i32
    %9 = arith.addi %8, %3 : i32
    %10 = arith.extsi %9 : i32 to i64
    %11 = arith.muli %10, %c16384_i64 : i64
    %12 = arith.muli %4, %c32_i32 : i32
    %13 = arith.addi %12, %3 : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.muli %14, %c128_i64 : i64
    %16 = tt.addptr %arg1, %15 : !tt.ptr<bf16>, i64
    %17 = tt.addptr %arg0, %15 : !tt.ptr<bf16>, i64
    %18 = tt.addptr %arg2, %15 : !tt.ptr<bf16>, i64
    %19 = tt.addptr %arg3, %15 : !tt.ptr<bf16>, i64
    %20 = arith.muli %1, %c16384_i32 : i32
    %21 = tt.addptr %arg6, %20 : !tt.ptr<f32>, i32
    %22 = arith.muli %0, %c32_i32 : i32
    %23 = arith.extsi %arg7 : i32 to i64
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %25 = tt.splat %arg7 : i32 -> tensor<64xi32>
    %26 = tt.addptr %arg4, %14 : !tt.ptr<f32>, i64
    %27:2 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %cst_0, %arg10 = %cst_0) -> (tensor<64x32xf32>, tensor<64x32xf32>)  : i32 {
      %31 = arith.extsi %arg8 : i32 to i64
      %32 = arith.muli %31, %c524288_i64 : i64
      %33 = arith.addi %11, %32 : i64
      %34 = tt.addptr %arg5, %33 : !tt.ptr<bf16>, i64
      %35 = tt.make_tensor_ptr %34, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %36 = arith.truncf %arg9 : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %35, %36 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %37 = tt.make_tensor_ptr %34, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c64_i32, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %38 = arith.truncf %arg10 : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %37, %38 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %39 = arith.muli %arg8, %c64_i32 : i32
      %40 = tt.make_tensor_ptr %18, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %41 = tt.load %40 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %42 = tt.dot %41, %36, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %43 = tt.make_tensor_ptr %18, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %c64_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %44 = tt.load %43 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %45 = tt.dot %44, %38, %42 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %46 = tt.make_tensor_ptr %16, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %47 = tt.load %46 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
      %48 = arith.extf %47 : tensor<64x32xbf16> to tensor<64x32xf32>
      %49 = arith.subf %48, %45 : tensor<64x32xf32>
      %50 = tt.make_tensor_ptr %19, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %51 = arith.truncf %49 : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %50, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %52 = arith.addi %arg8, %c1_i32 : i32
      %53 = arith.muli %52, %c64_i32 : i32
      %54 = arith.minsi %53, %arg7 : i32
      %55 = arith.subi %54, %c1_i32 : i32
      %56 = tt.splat %39 : i32 -> tensor<64xi32>
      %57 = arith.addi %56, %24 : tensor<64xi32>
      %58 = arith.cmpi slt, %57, %25 : tensor<64xi32>
      %59 = arith.muli %55, %c32_i32 : i32
      %60 = arith.addi %12, %59 : i32
      %61 = arith.addi %60, %3 : i32
      %62 = arith.extsi %61 : i32 to i64
      %63 = tt.addptr %arg4, %62 : !tt.ptr<f32>, i64
      %64 = tt.load %63 : !tt.ptr<f32>
      %65 = tt.make_tensor_ptr %26, [%23], [%c32_i64], [%39] {order = array<i32: 0>} : <tensor<64xf32>>
      %66 = tt.load %65 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
      %67 = tt.splat %64 : f32 -> tensor<64xf32>
      %68 = arith.subf %67, %66 : tensor<64xf32>
      %69 = math.exp2 %68 : tensor<64xf32>
      %70 = arith.select %58, %69, %cst : tensor<64xi1>, tensor<64xf32>
      %71 = tt.expand_dims %70 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %72 = tt.broadcast %71 : tensor<64x1xf32> -> tensor<64x32xf32>
      %73 = arith.mulf %49, %72 : tensor<64x32xf32>
      %74 = math.exp2 %64 : f32
      %75 = tt.splat %74 : f32 -> tensor<64x32xf32>
      %76 = arith.mulf %arg9, %75 : tensor<64x32xf32>
      %77 = arith.mulf %arg10, %75 : tensor<64x32xf32>
      %78 = arith.truncf %73 : tensor<64x32xf32> to tensor<64x32xbf16>
      %79 = tt.make_tensor_ptr %17, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c0_i32, %39] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
      %80 = tt.load %79 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %81 = tt.dot %80, %78, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %82 = arith.addf %81, %76 {triton_cv12.add_from_dot} : tensor<64x32xf32>
      %83 = tt.make_tensor_ptr %17, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c64_i32, %39] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
      %84 = tt.load %83 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %85 = tt.dot %84, %78, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %86 = arith.addf %85, %77 {triton_cv12.add_from_dot} : tensor<64x32xf32>
      scf.yield %82, %86 : tensor<64x32xf32>, tensor<64x32xf32>
    }
    %28 = arith.muli %0, %c32_i32 : i32
    %29 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %28] {order = array<i32: 1, 0>} : <tensor<64x32xf32>>
    tt.store %29, %27#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf32>>
    %30 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c64_i32, %28] {order = array<i32: 1, 0>} : <tensor<64x32xf32>>
    tt.store %30, %27#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf32>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/dllm/bwd_kv_ul_norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_bwd_kv_ul(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %cst_6 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %3 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %8 = arith.muli %4, %cst_9 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:     %9 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %10 = tt.addptr %9, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %11 = tt.broadcast %10 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %12 = tt.broadcast %6 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
// CHECK-NEXT:     %13 = tt.addptr %11, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %14 = tt.bitcast %13 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %16 = arith.remsi %0, %1 : i32
// CHECK-NEXT:     %17 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %18 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %19 = tt.splat %18 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %20 = tt.expand_dims %17 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %21 = tt.broadcast %20 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %22 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %23 = tt.splat %22 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %24 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %26 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %28 = tt.splat %arg10 {DataUse} : f32 -> tensor<32x32xf32>
// CHECK-NEXT:     %29 = arith.cmpi ne, %15, %cst_4 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %30 = tt.splat %arg10 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     %31:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %32 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %33 = tt.load %32 : !tt.ptr<i32>
// CHECK-NEXT:       %34 = arith.subi %33, %arg15 : i32
// CHECK-NEXT:       %35 = arith.addi %34, %c31_i32 : i32
// CHECK-NEXT:       %36 = arith.divsi %35, %c32_i32 : i32
// CHECK-NEXT:       %37 = arith.addi %arg16, %36 : i32
// CHECK-NEXT:       %38 = arith.remsi %arg16, %1 : i32
// CHECK-NEXT:       %39 = arith.subi %16, %38 : i32
// CHECK-NEXT:       %40 = arith.addi %39, %1 : i32
// CHECK-NEXT:       %41 = arith.remsi %40, %1 : i32
// CHECK-NEXT:       %42 = arith.addi %arg16, %41 : i32
// CHECK-NEXT:       %43 = tt.splat %33 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %44 = tt.splat %33 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %45 = tt.splat %33 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:       %46 = tt.splat %33 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:       scf.for %arg17 = %42 to %37 step %1  : i32 {
// CHECK-NEXT:         %47 = arith.subi %arg17, %arg16 : i32
// CHECK-NEXT:         %48 = arith.muli %47, %c32_i32 : i32
// CHECK-NEXT:         %49 = arith.addi %arg15, %48 : i32
// CHECK-NEXT:         %50 = tt.splat %49 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %51 = arith.addi %50, %2 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %52 = tt.expand_dims %51 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %53 = arith.muli %52, %cst_8 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %54 = tt.addptr %19, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %55 = tt.broadcast %54 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %56 = tt.addptr %55, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %57 = tt.addptr %23, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %58 = tt.broadcast %57 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %59 = tt.addptr %58, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %60 = tt.addptr %25, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %61 = tt.broadcast %60 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %62 = tt.addptr %61, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %63 = tt.addptr %27, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %64 = tt.broadcast %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %65 = tt.addptr %64, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %66 = arith.cmpi slt, %52, %43 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %67 = tt.broadcast %66 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %68 = tt.load %56, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %69 = tt.load %59, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %70 = tt.trans %68 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %71 = tt.trans %69 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %72 = tt.splat %49 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:         %73 = arith.addi %72, %5 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %74 = arith.cmpi slt, %73, %44 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %75 = tt.splat %49 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:         %76 = arith.addi %75, %7 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %77 = arith.cmpi slt, %76, %45 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %78 = tt.broadcast %74 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %79 = tt.broadcast %77 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %80 = arith.andi %78, %79 {DataUse} : tensor<32x32xi1>
// CHECK-NEXT:         %81 = arith.muli %52, %cst {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %82 = arith.cmpi slt, %51, %46 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %83 = arith.uitofp %80 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
// CHECK-NEXT:         %84 = arith.subf %83, %cst_2 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %85 = arith.mulf %84, %cst_3 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %86:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_7, %arg20 = %cst_7) -> (tensor<32x128xf32>, tensor<32x128xf32>)  : i32 {
// CHECK-NEXT:           %89 = arith.muli %arg18, %c128_i32 : i32
// CHECK-NEXT:           %90 = tt.addptr %arg0, %89 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %91 = tt.splat %90 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %92 = tt.addptr %91, %81 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %93 = tt.broadcast %92 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %94 = tt.addptr %93, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %95 = tt.addptr %arg3, %89 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %96 = tt.splat %95 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %97 = tt.addptr %96, %81 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %98 = tt.broadcast %97 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %99 = tt.addptr %98, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %100 = arith.muli %arg18, %arg13 : i32
// CHECK-NEXT:           %101 = tt.addptr %arg4, %100 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %102 = tt.splat %101 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %103 = tt.addptr %102, %51 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:           %104 = tt.addptr %arg5, %100 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %105 = tt.splat %104 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %106 = tt.addptr %105, %51 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:           %107 = tt.load %94, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %108 = tt.load %106, %82, %cst_0 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %109 = tt.dot %107, %70, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %110 = arith.mulf %109, %28 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %111 = arith.addf %110, %85 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %112 = arith.select %29, %111, %cst_5 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:           %113 = tt.load %99, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %114 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %115 = tt.broadcast %114 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:           %116 = arith.subf %112, %115 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %117 = math.exp %116 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %118 = arith.truncf %117 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %119 = tt.trans %118 {DataUse, order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
// CHECK-NEXT:           %120 = tt.dot %119, %113, %arg20 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %121 = tt.load %103, %82, %cst_0 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %122 = tt.dot %113, %71, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %123 = tt.expand_dims %121 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %124 = tt.broadcast %123 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:           %125 = arith.subf %122, %124 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %126 = arith.mulf %117, %125 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %127 = arith.truncf %126 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %128 = tt.trans %127 {DataUse, order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
// CHECK-NEXT:           %129 = tt.dot %128, %107, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %130 = arith.mulf %129, %30 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %131 = arith.addf %arg19, %130 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %131, %120 : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %87 = arith.truncf %86#0 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:         tt.store %62, %87, %67 : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %88 = arith.truncf %86#1 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:         tt.store %65, %88, %67 : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %33, %37 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_bwd_kv_ul(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<640> : tensor<32x1xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
    %cst_3 = arith.constant dense<1.000000e+06> : tensor<32x32xf32>
    %cst_4 = arith.constant dense<0> : tensor<32x32xi8>
    %cst_5 = arith.constant dense<-1.000000e+06> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    %cst_8 = arith.constant dense<128> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %cst_9 = arith.constant dense<64> : tensor<32x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %4 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %5 = arith.muli %3, %cst_9 : tensor<32x1xi32>
    %6 = tt.splat %arg11 : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %7 = tt.addptr %6, %5 : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %8 = tt.broadcast %7 : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %9 = tt.broadcast %4 : tensor<1x32xi32> -> tensor<32x32xi32>
    %10 = tt.addptr %8, %9 : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %11 = tt.bitcast %10 : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %12 = tt.load %11 {was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %13 = arith.remsi %0, %1 : i32
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %15 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
    %16 = tt.splat %15 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %17 = tt.expand_dims %14 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %18 = tt.broadcast %17 : tensor<1x128xi32> -> tensor<32x128xi32>
    %19 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
    %20 = tt.splat %19 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %21 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
    %22 = tt.splat %21 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %23 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
    %24 = tt.splat %23 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %25 = tt.splat %arg10 : f32 -> tensor<32x32xf32>
    %26 = arith.cmpi ne, %12, %cst_4 : tensor<32x32xi8>
    %27 = tt.splat %arg10 : f32 -> tensor<32x128xf32>
    %28:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
      %29 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
      %30 = tt.load %29 : !tt.ptr<i32>
      %31 = arith.subi %30, %arg15 : i32
      %32 = arith.addi %31, %c31_i32 : i32
      %33 = arith.divsi %32, %c32_i32 : i32
      %34 = arith.addi %arg16, %33 : i32
      %35 = arith.remsi %arg16, %1 : i32
      %36 = arith.subi %13, %35 : i32
      %37 = arith.addi %36, %1 : i32
      %38 = arith.remsi %37, %1 : i32
      %39 = arith.addi %arg16, %38 : i32
      %40 = tt.splat %30 : i32 -> tensor<32x1xi32>
      %41 = tt.splat %30 : i32 -> tensor<1x32xi32>
      %42 = tt.splat %30 : i32 -> tensor<32xi32>
      scf.for %arg17 = %39 to %34 step %1  : i32 {
        %43 = arith.subi %arg17, %arg16 : i32
        %44 = arith.muli %43, %c32_i32 : i32
        %45 = arith.addi %arg15, %44 : i32
        %46 = tt.splat %45 : i32 -> tensor<32xi32>
        %47 = arith.addi %46, %2 : tensor<32xi32>
        %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %49 = arith.muli %48, %cst_8 : tensor<32x1xi32>
        %50 = tt.addptr %16, %49 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %51 = tt.broadcast %50 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %52 = tt.addptr %51, %18 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %53 = tt.addptr %20, %49 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %54 = tt.broadcast %53 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %55 = tt.addptr %54, %18 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %56 = tt.addptr %22, %49 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %57 = tt.broadcast %56 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %58 = tt.addptr %57, %18 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %59 = tt.addptr %24, %49 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %60 = tt.broadcast %59 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %61 = tt.addptr %60, %18 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %62 = arith.cmpi slt, %48, %40 : tensor<32x1xi32>
        %63 = tt.broadcast %62 : tensor<32x1xi1> -> tensor<32x128xi1>
        %64 = tt.load %52, %63, %cst_6 : tensor<32x128x!tt.ptr<bf16>>
        %65 = tt.load %55, %63, %cst_6 : tensor<32x128x!tt.ptr<bf16>>
        %66 = tt.trans %64 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %67 = tt.trans %65 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %68 = tt.splat %45 : i32 -> tensor<32x1xi32>
        %69 = arith.addi %68, %3 : tensor<32x1xi32>
        %70 = arith.cmpi slt, %69, %40 : tensor<32x1xi32>
        %71 = tt.splat %45 : i32 -> tensor<1x32xi32>
        %72 = arith.addi %71, %4 : tensor<1x32xi32>
        %73 = arith.cmpi slt, %72, %41 : tensor<1x32xi32>
        %74 = tt.broadcast %70 : tensor<32x1xi1> -> tensor<32x32xi1>
        %75 = tt.broadcast %73 : tensor<1x32xi1> -> tensor<32x32xi1>
        %76 = arith.andi %74, %75 : tensor<32x32xi1>
        %77 = arith.muli %48, %cst : tensor<32x1xi32>
        %78 = arith.cmpi slt, %47, %42 : tensor<32xi32>
        %79 = arith.uitofp %76 : tensor<32x32xi1> to tensor<32x32xf32>
        %80 = arith.subf %79, %cst_2 : tensor<32x32xf32>
        %81 = arith.mulf %80, %cst_3 : tensor<32x32xf32>
        %82:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_7, %arg20 = %cst_7) -> (tensor<32x128xf32>, tensor<32x128xf32>)  : i32 {
          %85 = arith.muli %arg18, %c128_i32 : i32
          %86 = tt.addptr %arg0, %85 : !tt.ptr<bf16>, i32
          %87 = tt.splat %86 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
          %88 = tt.addptr %87, %77 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %89 = tt.broadcast %88 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %90 = tt.addptr %89, %18 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %91 = tt.addptr %arg3, %85 : !tt.ptr<bf16>, i32
          %92 = tt.splat %91 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
          %93 = tt.addptr %92, %77 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %94 = tt.broadcast %93 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %95 = tt.addptr %94, %18 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %96 = arith.muli %arg18, %arg13 : i32
          %97 = tt.addptr %arg4, %96 : !tt.ptr<f32>, i32
          %98 = tt.splat %97 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
          %99 = tt.addptr %98, %47 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
          %100 = tt.addptr %arg5, %96 : !tt.ptr<f32>, i32
          %101 = tt.splat %100 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
          %102 = tt.addptr %101, %47 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
          %103 = tt.load %90, %63, %cst_6 : tensor<32x128x!tt.ptr<bf16>>
          %104 = tt.load %102, %78, %cst_0 : tensor<32x!tt.ptr<f32>>
          %105 = tt.dot %103, %66, %cst_1 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %106 = arith.mulf %105, %25 : tensor<32x32xf32>
          %107 = arith.addf %106, %81 : tensor<32x32xf32>
          %108 = arith.select %26, %107, %cst_5 : tensor<32x32xi1>, tensor<32x32xf32>
          %109 = tt.load %95, %63, %cst_6 : tensor<32x128x!tt.ptr<bf16>>
          %110 = tt.expand_dims %104 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %111 = tt.broadcast %110 : tensor<32x1xf32> -> tensor<32x32xf32>
          %112 = arith.subf %108, %111 : tensor<32x32xf32>
          %113 = math.exp %112 : tensor<32x32xf32>
          %114 = arith.truncf %113 : tensor<32x32xf32> to tensor<32x32xbf16>
          %115 = tt.trans %114 {order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
          %116 = tt.dot %115, %109, %arg20 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %117 = tt.load %99, %78, %cst_0 : tensor<32x!tt.ptr<f32>>
          %118 = tt.dot %109, %67, %cst_1 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %119 = tt.expand_dims %117 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %120 = tt.broadcast %119 : tensor<32x1xf32> -> tensor<32x32xf32>
          %121 = arith.subf %118, %120 : tensor<32x32xf32>
          %122 = arith.mulf %113, %121 : tensor<32x32xf32>
          %123 = arith.truncf %122 : tensor<32x32xf32> to tensor<32x32xbf16>
          %124 = tt.trans %123 {order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
          %125 = tt.dot %124, %103, %cst_7 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %126 = arith.mulf %125, %27 : tensor<32x128xf32>
          %127 = arith.addf %arg19, %126 : tensor<32x128xf32>
          scf.yield %127, %116 : tensor<32x128xf32>, tensor<32x128xf32>
        }
        %83 = arith.truncf %82#0 : tensor<32x128xf32> to tensor<32x128xbf16>
        tt.store %58, %83, %63 : tensor<32x128x!tt.ptr<bf16>>
        %84 = arith.truncf %82#1 : tensor<32x128xf32> to tensor<32x128xbf16>
        tt.store %61, %84, %63 : tensor<32x128x!tt.ptr<bf16>>
      }
      scf.yield %30, %34 : i32, i32
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/dllm/bwd_kv_ur_norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_bwd_kv_ur(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<640> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<640> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<1.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<1.000000e+06> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<0> : tensor<64x64xi8>
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<64x64xf32>
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %c127_i32 = arith.constant 127 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<64> : tensor<64x1xi32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %3 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %8 = arith.muli %4, %cst_13 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %9 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
// CHECK-NEXT:     %10 = tt.addptr %9, %8 {MetaUse} : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
// CHECK-NEXT:     %11 = tt.broadcast %10 {MetaUse} : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
// CHECK-NEXT:     %12 = tt.broadcast %6 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %13 = tt.addptr %11, %12 {MetaUse} : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
// CHECK-NEXT:     %14 = tt.bitcast %13 {MetaUse} : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:     %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:     %16 = arith.remsi %0, %1 : i32
// CHECK-NEXT:     %17 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %18 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %19 = tt.splat %18 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %20 = tt.expand_dims %17 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %21 = tt.broadcast %20 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %22 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %23 = tt.splat %22 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %24 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %26 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %28 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     %29 = arith.cmpi ne, %15, %cst_8 {DataUse} : tensor<64x64xi8>
// CHECK-NEXT:     %30 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %31 = tt.broadcast %20 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %32 = tt.splat %arg10 {DataUse} : f32 -> tensor<128x64xf32>
// CHECK-NEXT:     %33:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %34 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %35 = tt.load %34 : !tt.ptr<i32>
// CHECK-NEXT:       %36 = arith.subi %35, %arg15 : i32
// CHECK-NEXT:       %37 = arith.addi %36, %c63_i32 : i32
// CHECK-NEXT:       %38 = arith.divsi %37, %c64_i32 : i32
// CHECK-NEXT:       %39 = arith.addi %arg16, %38 : i32
// CHECK-NEXT:       %40 = arith.remsi %arg16, %1 : i32
// CHECK-NEXT:       %41 = arith.subi %16, %40 : i32
// CHECK-NEXT:       %42 = arith.addi %41, %1 : i32
// CHECK-NEXT:       %43 = arith.remsi %42, %1 : i32
// CHECK-NEXT:       %44 = arith.addi %arg16, %43 : i32
// CHECK-NEXT:       %45 = arith.addi %arg12, %arg15 : i32
// CHECK-NEXT:       %46 = tt.splat %35 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:       %47 = tt.splat %35 {DataUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:       %48 = tt.splat %35 {DataUse} : i32 -> tensor<1x64xi32>
// CHECK-NEXT:       %49 = tt.splat %35 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %50 = arith.addi %36, %c127_i32 : i32
// CHECK-NEXT:       %51 = arith.divsi %50, %c128_i32 : i32
// CHECK-NEXT:       %52 = tt.splat %35 {MetaUse} : i32 -> tensor<128x1xi32>
// CHECK-NEXT:       %53 = tt.splat %35 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       scf.for %arg17 = %44 to %39 step %1  : i32 {
// CHECK-NEXT:         %54 = arith.subi %arg17, %arg16 : i32
// CHECK-NEXT:         %55 = arith.muli %54, %c64_i32 : i32
// CHECK-NEXT:         %56 = arith.addi %45, %55 : i32
// CHECK-NEXT:         %57 = tt.splat %56 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %58 = arith.addi %57, %2 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %59 = tt.expand_dims %58 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %60 = arith.muli %59, %cst_12 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %61 = tt.addptr %19, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %62 = tt.broadcast %61 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %63 = tt.addptr %62, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %64 = tt.addptr %23, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %65 = tt.broadcast %64 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %66 = tt.addptr %65, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %67 = tt.addptr %25, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %68 = tt.broadcast %67 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %69 = tt.addptr %68, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %70 = tt.addptr %27, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %71 = tt.broadcast %70 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %72 = tt.addptr %71, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %73 = arith.addi %arg15, %55 : i32
// CHECK-NEXT:         %74 = tt.splat %73 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %75 = arith.addi %74, %2 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %76 = tt.expand_dims %75 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %77 = arith.cmpi slt, %76, %46 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %78 = tt.broadcast %77 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:         %79 = tt.load %63, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %80 = tt.load %66, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %81 = tt.trans %79 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %82 = tt.trans %80 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %83 = tt.splat %73 {DataUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:         %84 = arith.addi %83, %5 {DataUse} : tensor<64x1xi32>
// CHECK-NEXT:         %85 = arith.cmpi slt, %84, %47 {DataUse} : tensor<64x1xi32>
// CHECK-NEXT:         %86 = tt.splat %73 {DataUse} : i32 -> tensor<1x64xi32>
// CHECK-NEXT:         %87 = arith.addi %86, %7 {DataUse} : tensor<1x64xi32>
// CHECK-NEXT:         %88 = arith.cmpi slt, %87, %48 {DataUse} : tensor<1x64xi32>
// CHECK-NEXT:         %89 = tt.broadcast %85 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:         %90 = tt.broadcast %88 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:         %91 = arith.andi %89, %90 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:         %92 = arith.muli %76, %cst_3 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %93 = arith.cmpi slt, %75, %49 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %94 = arith.uitofp %91 {DataUse} : tensor<64x64xi1> to tensor<64x64xf32>
// CHECK-NEXT:         %95 = arith.subf %94, %cst_6 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %96 = arith.mulf %95, %cst_7 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %97 = arith.addi %54, %c1_i32 : i32
// CHECK-NEXT:         %98 = arith.divsi %55, %c128_i32 : i32
// CHECK-NEXT:         %99 = arith.addi %98, %c1_i32 : i32
// CHECK-NEXT:         %100 = arith.muli %99, %c128_i32 : i32
// CHECK-NEXT:         %101 = arith.divsi %100, %c64_i32 : i32
// CHECK-NEXT:         %102:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_11, %arg20 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:           %105 = arith.muli %arg18, %c128_i32 : i32
// CHECK-NEXT:           %106 = tt.addptr %arg0, %105 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %107 = tt.splat %106 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %108 = tt.addptr %107, %92 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:           %109 = tt.broadcast %108 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %110 = tt.addptr %109, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:           %111 = tt.addptr %arg3, %105 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %112 = tt.splat %111 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %113 = tt.addptr %112, %92 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:           %114 = tt.broadcast %113 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %115 = tt.addptr %114, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:           %116 = arith.muli %arg18, %arg13 : i32
// CHECK-NEXT:           %117 = tt.addptr %arg4, %116 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %118 = tt.splat %117 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %119 = tt.addptr %118, %75 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:           %120 = tt.addptr %arg5, %116 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %121 = tt.splat %120 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %122 = tt.addptr %121, %75 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:           %123 = tt.load %110, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %124 = tt.load %122, %93, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %125 = tt.dot %123, %81, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:           %126 = arith.mulf %125, %28 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %127 = arith.addf %126, %96 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %128 = arith.select %29, %127, %cst_9 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:           %129 = tt.load %115, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %130 = tt.expand_dims %124 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:           %131 = tt.broadcast %130 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:           %132 = arith.subf %128, %131 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %133 = math.exp %132 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %134 = arith.truncf %133 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:           %135 = tt.trans %134 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:           %136 = tt.dot %135, %129, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %137 = arith.addf %136, %arg20 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:           %138 = tt.load %119, %93, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %139 = tt.dot %129, %82, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:           %140 = tt.expand_dims %138 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:           %141 = tt.broadcast %140 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:           %142 = arith.subf %139, %141 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %143 = arith.mulf %133, %142 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %144 = arith.truncf %143 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:           %145 = tt.trans %144 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:           %146 = tt.dot %145, %123, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %147 = arith.mulf %146, %30 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %148 = arith.addf %arg19, %147 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %149:2 = scf.for %arg21 = %97 to %101 step %c1_i32 iter_args(%arg22 = %148, %arg23 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:             %161 = arith.muli %arg21, %c64_i32 : i32
// CHECK-NEXT:             %162 = arith.addi %arg15, %161 : i32
// CHECK-NEXT:             %163 = tt.splat %162 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:             %164 = arith.addi %163, %2 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:             %165 = tt.expand_dims %164 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:             %166 = arith.muli %165, %cst_3 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:             %167 = tt.addptr %107, %166 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:             %168 = tt.broadcast %167 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %169 = tt.addptr %168, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:             %170 = tt.addptr %112, %166 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:             %171 = tt.broadcast %170 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %172 = tt.addptr %171, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:             %173 = tt.addptr %118, %164 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:             %174 = tt.addptr %121, %164 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:             %175 = arith.cmpi slt, %165, %46 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:             %176 = arith.cmpi slt, %164, %49 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:             %177 = tt.broadcast %175 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:             %178 = tt.load %169, %177, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %179 = tt.load %174, %176, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:             %180 = tt.dot %178, %81, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:             %181 = arith.mulf %180, %28 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %182 = tt.load %172, %177, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %183 = tt.expand_dims %179 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:             %184 = tt.broadcast %183 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:             %185 = arith.subf %181, %184 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %186 = math.exp %185 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %187 = arith.truncf %186 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:             %188 = tt.trans %187 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:             %189 = tt.dot %188, %182, %arg23 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %190 = tt.load %173, %176, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:             %191 = tt.dot %182, %82, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:             %192 = tt.expand_dims %190 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:             %193 = tt.broadcast %192 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:             %194 = arith.subf %191, %193 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %195 = arith.mulf %186, %194 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %196 = arith.truncf %195 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:             %197 = tt.trans %196 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:             %198 = tt.dot %197, %178, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %199 = arith.mulf %198, %30 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             %200 = arith.addf %arg22, %199 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             scf.yield %200, %189 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse, hivm.matmul_limited_in_cube}
// CHECK-NEXT:           %150 = arith.cmpi sgt, %101, %97 : i32
// CHECK-NEXT:           %151 = scf.if %150 -> (tensor<64x128xf32>) {
// CHECK-NEXT:             scf.yield %149#1 : tensor<64x128xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %cst_11 : tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse}
// CHECK-NEXT:           %152 = arith.addf %151, %137 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:           %153 = tt.splat %106 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %154 = tt.splat %111 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %155 = tt.splat %117 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %156 = tt.splat %120 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %157:2 = scf.for %arg21 = %99 to %51 step %c1_i32 iter_args(%arg22 = %149#0, %arg23 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:             %161 = arith.muli %arg21, %c128_i32 : i32
// CHECK-NEXT:             %162 = arith.addi %arg15, %161 : i32
// CHECK-NEXT:             %163 = tt.splat %162 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:             %164 = arith.addi %163, %17 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:             %165 = tt.expand_dims %164 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:             %166 = arith.muli %165, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:             %167 = tt.addptr %153, %166 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:             %168 = tt.broadcast %167 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %169 = tt.addptr %168, %31 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:             %170 = tt.addptr %154, %166 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:             %171 = tt.broadcast %170 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %172 = tt.addptr %171, %31 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:             %173 = tt.addptr %155, %164 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:             %174 = tt.addptr %156, %164 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:             %175 = arith.cmpi slt, %165, %52 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:             %176 = arith.cmpi slt, %164, %53 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:             %177 = tt.broadcast %175 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:             %178 = tt.load %169, %177, %cst_2 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %179 = tt.load %174, %176, %cst_0 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:             %180 = tt.dot %178, %81, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:             %181 = arith.mulf %180, %32 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %182 = tt.load %172, %177, %cst_2 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %183 = tt.expand_dims %179 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:             %184 = tt.broadcast %183 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:             %185 = arith.subf %181, %184 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %186 = math.exp %185 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %187 = arith.truncf %186 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:             %188 = tt.trans %187 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:             %189 = tt.dot %188, %182, %arg23 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %190 = tt.load %173, %176, %cst_0 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:             %191 = tt.dot %182, %82, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:             %192 = tt.expand_dims %190 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:             %193 = tt.broadcast %192 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:             %194 = arith.subf %191, %193 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %195 = arith.mulf %186, %194 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %196 = arith.truncf %195 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:             %197 = tt.trans %196 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:             %198 = tt.dot %197, %178, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %199 = arith.mulf %198, %30 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             %200 = arith.addf %arg22, %199 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             scf.yield %200, %189 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse, hivm.matmul_limited_in_cube}
// CHECK-NEXT:           %158 = arith.cmpi sgt, %51, %99 : i32
// CHECK-NEXT:           %159 = scf.if %158 -> (tensor<64x128xf32>) {
// CHECK-NEXT:             scf.yield %157#1 : tensor<64x128xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %cst_11 : tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse}
// CHECK-NEXT:           %160 = arith.addf %159, %152 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:           scf.yield %157#0, %160 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %103 = arith.truncf %102#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:         tt.store %69, %103, %78 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %104 = arith.truncf %102#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:         tt.store %72, %104, %78 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %35, %39 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_bwd_kv_ur(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
    %cst = arith.constant dense<640> : tensor<128x1xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_3 = arith.constant dense<640> : tensor<64x1xi32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_6 = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
    %cst_7 = arith.constant dense<1.000000e+06> : tensor<64x64xf32>
    %cst_8 = arith.constant dense<0> : tensor<64x64xi8>
    %cst_9 = arith.constant dense<-1.000000e+06> : tensor<64x64xf32>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %cst_12 = arith.constant dense<128> : tensor<64x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %cst_13 = arith.constant dense<64> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %4 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %5 = arith.muli %3, %cst_13 : tensor<64x1xi32>
    %6 = tt.splat %arg11 : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
    %7 = tt.addptr %6, %5 : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
    %8 = tt.broadcast %7 : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
    %9 = tt.broadcast %4 : tensor<1x64xi32> -> tensor<64x64xi32>
    %10 = tt.addptr %8, %9 : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
    %11 = tt.bitcast %10 : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
    %12 = tt.load %11 {was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
    %13 = arith.remsi %0, %1 : i32
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %15 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
    %16 = tt.splat %15 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %17 = tt.expand_dims %14 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %18 = tt.broadcast %17 : tensor<1x128xi32> -> tensor<64x128xi32>
    %19 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
    %20 = tt.splat %19 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %21 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
    %22 = tt.splat %21 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %23 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
    %24 = tt.splat %23 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %25 = tt.splat %arg10 : f32 -> tensor<64x64xf32>
    %26 = arith.cmpi ne, %12, %cst_8 : tensor<64x64xi8>
    %27 = tt.splat %arg10 : f32 -> tensor<64x128xf32>
    %28 = tt.broadcast %17 : tensor<1x128xi32> -> tensor<128x128xi32>
    %29 = tt.splat %arg10 : f32 -> tensor<128x64xf32>
    %30:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
      %31 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
      %32 = tt.load %31 : !tt.ptr<i32>
      %33 = arith.subi %32, %arg15 : i32
      %34 = arith.addi %33, %c63_i32 : i32
      %35 = arith.divsi %34, %c64_i32 : i32
      %36 = arith.addi %arg16, %35 : i32
      %37 = arith.remsi %arg16, %1 : i32
      %38 = arith.subi %13, %37 : i32
      %39 = arith.addi %38, %1 : i32
      %40 = arith.remsi %39, %1 : i32
      %41 = arith.addi %arg16, %40 : i32
      %42 = arith.addi %arg12, %arg15 : i32
      %43 = tt.splat %32 : i32 -> tensor<64x1xi32>
      %44 = tt.splat %32 : i32 -> tensor<1x64xi32>
      %45 = tt.splat %32 : i32 -> tensor<64xi32>
      %46 = arith.addi %33, %c127_i32 : i32
      %47 = arith.divsi %46, %c128_i32 : i32
      %48 = tt.splat %32 : i32 -> tensor<128x1xi32>
      %49 = tt.splat %32 : i32 -> tensor<128xi32>
      scf.for %arg17 = %41 to %36 step %1  : i32 {
        %50 = arith.subi %arg17, %arg16 : i32
        %51 = arith.muli %50, %c64_i32 : i32
        %52 = arith.addi %42, %51 : i32
        %53 = tt.splat %52 : i32 -> tensor<64xi32>
        %54 = arith.addi %53, %2 : tensor<64xi32>
        %55 = tt.expand_dims %54 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %56 = arith.muli %55, %cst_12 : tensor<64x1xi32>
        %57 = tt.addptr %16, %56 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %58 = tt.broadcast %57 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %59 = tt.addptr %58, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %60 = tt.addptr %20, %56 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %61 = tt.broadcast %60 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %62 = tt.addptr %61, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %63 = tt.addptr %22, %56 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %64 = tt.broadcast %63 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %66 = tt.addptr %24, %56 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %67 = tt.broadcast %66 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %68 = tt.addptr %67, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %69 = arith.addi %arg15, %51 : i32
        %70 = tt.splat %69 : i32 -> tensor<64xi32>
        %71 = arith.addi %70, %2 : tensor<64xi32>
        %72 = tt.expand_dims %71 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %73 = arith.cmpi slt, %72, %43 : tensor<64x1xi32>
        %74 = tt.broadcast %73 : tensor<64x1xi1> -> tensor<64x128xi1>
        %75 = tt.load %59, %74, %cst_10 : tensor<64x128x!tt.ptr<bf16>>
        %76 = tt.load %62, %74, %cst_10 : tensor<64x128x!tt.ptr<bf16>>
        %77 = tt.trans %75 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %78 = tt.trans %76 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %79 = tt.splat %69 : i32 -> tensor<64x1xi32>
        %80 = arith.addi %79, %3 : tensor<64x1xi32>
        %81 = arith.cmpi slt, %80, %43 : tensor<64x1xi32>
        %82 = tt.splat %69 : i32 -> tensor<1x64xi32>
        %83 = arith.addi %82, %4 : tensor<1x64xi32>
        %84 = arith.cmpi slt, %83, %44 : tensor<1x64xi32>
        %85 = tt.broadcast %81 : tensor<64x1xi1> -> tensor<64x64xi1>
        %86 = tt.broadcast %84 : tensor<1x64xi1> -> tensor<64x64xi1>
        %87 = arith.andi %85, %86 : tensor<64x64xi1>
        %88 = arith.muli %72, %cst_3 : tensor<64x1xi32>
        %89 = arith.cmpi slt, %71, %45 : tensor<64xi32>
        %90 = arith.uitofp %87 : tensor<64x64xi1> to tensor<64x64xf32>
        %91 = arith.subf %90, %cst_6 : tensor<64x64xf32>
        %92 = arith.mulf %91, %cst_7 : tensor<64x64xf32>
        %93 = arith.addi %50, %c1_i32 : i32
        %94 = arith.divsi %51, %c128_i32 : i32
        %95 = arith.addi %94, %c1_i32 : i32
        %96 = arith.muli %95, %c128_i32 : i32
        %97 = arith.divsi %96, %c64_i32 : i32
        %98:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_11, %arg20 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
          %101 = arith.muli %arg18, %c128_i32 : i32
          %102 = tt.addptr %arg0, %101 : !tt.ptr<bf16>, i32
          %103 = tt.splat %102 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
          %104 = tt.addptr %103, %88 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
          %105 = tt.broadcast %104 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
          %106 = tt.addptr %105, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
          %107 = tt.addptr %arg3, %101 : !tt.ptr<bf16>, i32
          %108 = tt.splat %107 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
          %109 = tt.addptr %108, %88 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
          %110 = tt.broadcast %109 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
          %111 = tt.addptr %110, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
          %112 = arith.muli %arg18, %arg13 : i32
          %113 = tt.addptr %arg4, %112 : !tt.ptr<f32>, i32
          %114 = tt.splat %113 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
          %115 = tt.addptr %114, %71 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
          %116 = tt.addptr %arg5, %112 : !tt.ptr<f32>, i32
          %117 = tt.splat %116 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
          %118 = tt.addptr %117, %71 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
          %119 = tt.load %106, %74, %cst_10 : tensor<64x128x!tt.ptr<bf16>>
          %120 = tt.load %118, %89, %cst_4 : tensor<64x!tt.ptr<f32>>
          %121 = tt.dot %119, %77, %cst_5 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
          %122 = arith.mulf %121, %25 : tensor<64x64xf32>
          %123 = arith.addf %122, %92 : tensor<64x64xf32>
          %124 = arith.select %26, %123, %cst_9 : tensor<64x64xi1>, tensor<64x64xf32>
          %125 = tt.load %111, %74, %cst_10 : tensor<64x128x!tt.ptr<bf16>>
          %126 = tt.expand_dims %120 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
          %127 = tt.broadcast %126 : tensor<64x1xf32> -> tensor<64x64xf32>
          %128 = arith.subf %124, %127 : tensor<64x64xf32>
          %129 = math.exp %128 : tensor<64x64xf32>
          %130 = arith.truncf %129 : tensor<64x64xf32> to tensor<64x64xbf16>
          %131 = tt.trans %130 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
          %132 = tt.dot %131, %125, %cst_11 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
          %133 = arith.addf %132, %arg20 {triton_cv12.add_from_dot} : tensor<64x128xf32>
          %134 = tt.load %115, %89, %cst_4 : tensor<64x!tt.ptr<f32>>
          %135 = tt.dot %125, %78, %cst_5 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
          %136 = tt.expand_dims %134 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
          %137 = tt.broadcast %136 : tensor<64x1xf32> -> tensor<64x64xf32>
          %138 = arith.subf %135, %137 : tensor<64x64xf32>
          %139 = arith.mulf %129, %138 : tensor<64x64xf32>
          %140 = arith.truncf %139 : tensor<64x64xf32> to tensor<64x64xbf16>
          %141 = tt.trans %140 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
          %142 = tt.dot %141, %119, %cst_11 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
          %143 = arith.mulf %142, %27 : tensor<64x128xf32>
          %144 = arith.addf %arg19, %143 : tensor<64x128xf32>
          %145:2 = scf.for %arg21 = %93 to %97 step %c1_i32 iter_args(%arg22 = %144, %arg23 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
            %157 = arith.muli %arg21, %c64_i32 : i32
            %158 = arith.addi %arg15, %157 : i32
            %159 = tt.splat %158 : i32 -> tensor<64xi32>
            %160 = arith.addi %159, %2 : tensor<64xi32>
            %161 = tt.expand_dims %160 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
            %162 = arith.muli %161, %cst_3 : tensor<64x1xi32>
            %163 = tt.addptr %103, %162 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
            %164 = tt.broadcast %163 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
            %165 = tt.addptr %164, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
            %166 = tt.addptr %108, %162 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
            %167 = tt.broadcast %166 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
            %168 = tt.addptr %167, %18 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
            %169 = tt.addptr %114, %160 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            %170 = tt.addptr %117, %160 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            %171 = arith.cmpi slt, %161, %43 : tensor<64x1xi32>
            %172 = arith.cmpi slt, %160, %45 : tensor<64xi32>
            %173 = tt.broadcast %171 : tensor<64x1xi1> -> tensor<64x128xi1>
            %174 = tt.load %165, %173, %cst_10 : tensor<64x128x!tt.ptr<bf16>>
            %175 = tt.load %170, %172, %cst_4 : tensor<64x!tt.ptr<f32>>
            %176 = tt.dot %174, %77, %cst_5 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
            %177 = arith.mulf %176, %25 : tensor<64x64xf32>
            %178 = tt.load %168, %173, %cst_10 : tensor<64x128x!tt.ptr<bf16>>
            %179 = tt.expand_dims %175 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
            %180 = tt.broadcast %179 : tensor<64x1xf32> -> tensor<64x64xf32>
            %181 = arith.subf %177, %180 : tensor<64x64xf32>
            %182 = math.exp %181 : tensor<64x64xf32>
            %183 = arith.truncf %182 : tensor<64x64xf32> to tensor<64x64xbf16>
            %184 = tt.trans %183 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
            %185 = tt.dot %184, %178, %arg23 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
            %186 = tt.load %169, %172, %cst_4 : tensor<64x!tt.ptr<f32>>
            %187 = tt.dot %178, %78, %cst_5 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
            %188 = tt.expand_dims %186 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
            %189 = tt.broadcast %188 : tensor<64x1xf32> -> tensor<64x64xf32>
            %190 = arith.subf %187, %189 : tensor<64x64xf32>
            %191 = arith.mulf %182, %190 : tensor<64x64xf32>
            %192 = arith.truncf %191 : tensor<64x64xf32> to tensor<64x64xbf16>
            %193 = tt.trans %192 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
            %194 = tt.dot %193, %174, %cst_11 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
            %195 = arith.mulf %194, %27 : tensor<64x128xf32>
            %196 = arith.addf %arg22, %195 : tensor<64x128xf32>
            scf.yield %196, %185 : tensor<64x128xf32>, tensor<64x128xf32>
          } {hivm.matmul_limited_in_cube}
          %146 = arith.cmpi sgt, %97, %93 : i32
          %147 = scf.if %146 -> (tensor<64x128xf32>) {
            scf.yield %145#1 : tensor<64x128xf32>
          } else {
            scf.yield %cst_11 : tensor<64x128xf32>
          }
          %148 = arith.addf %147, %133 {triton_cv12.add_from_dot} : tensor<64x128xf32>
          %149 = tt.splat %102 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
          %150 = tt.splat %107 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
          %151 = tt.splat %113 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
          %152 = tt.splat %116 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
          %153:2 = scf.for %arg21 = %95 to %47 step %c1_i32 iter_args(%arg22 = %145#0, %arg23 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
            %157 = arith.muli %arg21, %c128_i32 : i32
            %158 = arith.addi %arg15, %157 : i32
            %159 = tt.splat %158 : i32 -> tensor<128xi32>
            %160 = arith.addi %159, %14 : tensor<128xi32>
            %161 = tt.expand_dims %160 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
            %162 = arith.muli %161, %cst : tensor<128x1xi32>
            %163 = tt.addptr %149, %162 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
            %164 = tt.broadcast %163 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
            %165 = tt.addptr %164, %28 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
            %166 = tt.addptr %150, %162 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
            %167 = tt.broadcast %166 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
            %168 = tt.addptr %167, %28 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
            %169 = tt.addptr %151, %160 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            %170 = tt.addptr %152, %160 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            %171 = arith.cmpi slt, %161, %48 : tensor<128x1xi32>
            %172 = arith.cmpi slt, %160, %49 : tensor<128xi32>
            %173 = tt.broadcast %171 : tensor<128x1xi1> -> tensor<128x128xi1>
            %174 = tt.load %165, %173, %cst_2 : tensor<128x128x!tt.ptr<bf16>>
            %175 = tt.load %170, %172, %cst_0 : tensor<128x!tt.ptr<f32>>
            %176 = tt.dot %174, %77, %cst_1 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
            %177 = arith.mulf %176, %29 : tensor<128x64xf32>
            %178 = tt.load %168, %173, %cst_2 : tensor<128x128x!tt.ptr<bf16>>
            %179 = tt.expand_dims %175 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
            %180 = tt.broadcast %179 : tensor<128x1xf32> -> tensor<128x64xf32>
            %181 = arith.subf %177, %180 : tensor<128x64xf32>
            %182 = math.exp %181 : tensor<128x64xf32>
            %183 = arith.truncf %182 : tensor<128x64xf32> to tensor<128x64xbf16>
            %184 = tt.trans %183 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
            %185 = tt.dot %184, %178, %arg23 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
            %186 = tt.load %169, %172, %cst_0 : tensor<128x!tt.ptr<f32>>
            %187 = tt.dot %178, %78, %cst_1 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
            %188 = tt.expand_dims %186 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
            %189 = tt.broadcast %188 : tensor<128x1xf32> -> tensor<128x64xf32>
            %190 = arith.subf %187, %189 : tensor<128x64xf32>
            %191 = arith.mulf %182, %190 : tensor<128x64xf32>
            %192 = arith.truncf %191 : tensor<128x64xf32> to tensor<128x64xbf16>
            %193 = tt.trans %192 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
            %194 = tt.dot %193, %174, %cst_11 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
            %195 = arith.mulf %194, %27 : tensor<64x128xf32>
            %196 = arith.addf %arg22, %195 : tensor<64x128xf32>
            scf.yield %196, %185 : tensor<64x128xf32>, tensor<64x128xf32>
          } {hivm.matmul_limited_in_cube}
          %154 = arith.cmpi sgt, %47, %95 : i32
          %155 = scf.if %154 -> (tensor<64x128xf32>) {
            scf.yield %153#1 : tensor<64x128xf32>
          } else {
            scf.yield %cst_11 : tensor<64x128xf32>
          }
          %156 = arith.addf %155, %148 {triton_cv12.add_from_dot} : tensor<64x128xf32>
          scf.yield %153#0, %156 : tensor<64x128xf32>, tensor<64x128xf32>
        }
        %99 = arith.truncf %98#0 : tensor<64x128xf32> to tensor<64x128xbf16>
        tt.store %65, %99, %74 : tensor<64x128x!tt.ptr<bf16>>
        %100 = arith.truncf %98#1 : tensor<64x128xf32> to tensor<64x128xbf16>
        tt.store %68, %100, %74 : tensor<64x128x!tt.ptr<bf16>>
      }
      scf.yield %32, %36 : i32, i32
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/dllm/bwd_q_u_norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_bwd_q_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i32>, %arg8: i32, %arg9: f32, %arg10: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:     %cst_7 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %cst_11 = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %3 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %8 = arith.muli %4, %cst_11 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:     %9 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %10 = tt.addptr %9, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %11 = tt.broadcast %10 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %12 = tt.broadcast %6 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
// CHECK-NEXT:     %13 = tt.addptr %11, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %14 = tt.bitcast %13 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %16 = tt.splat %arg10 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %17 = tt.addptr %16, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %18 = tt.broadcast %17 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %19 = tt.addptr %18, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %20 = tt.bitcast %19 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %21 = tt.load %20 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %22 = arith.remsi %0, %1 : i32
// CHECK-NEXT:     %23 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %24 = tt.expand_dims %23 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %25 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %26 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x32xf32>
// CHECK-NEXT:     %27 = arith.cmpi ne, %15, %cst_5 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %28 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     %29 = arith.cmpi ne, %21, %cst_5 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %30 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %31:2 = scf.for %arg14 = %c0_i32 to %arg8 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %32 = tt.addptr %arg7, %arg14 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %33 = tt.load %32 : !tt.ptr<i32>
// CHECK-NEXT:       %34 = arith.subi %33, %arg15 : i32
// CHECK-NEXT:       %35 = arith.addi %34, %c31_i32 : i32
// CHECK-NEXT:       %36 = arith.divsi %35, %c32_i32 : i32
// CHECK-NEXT:       %37 = arith.addi %arg16, %36 : i32
// CHECK-NEXT:       %38 = arith.muli %arg16, %c5_i32 : i32
// CHECK-NEXT:       %39 = arith.remsi %38, %1 : i32
// CHECK-NEXT:       %40 = arith.subi %22, %39 : i32
// CHECK-NEXT:       %41 = arith.addi %40, %1 : i32
// CHECK-NEXT:       %42 = arith.remsi %41, %1 : i32
// CHECK-NEXT:       %43 = arith.addi %38, %42 : i32
// CHECK-NEXT:       %44 = arith.muli %37, %c5_i32 : i32
// CHECK-NEXT:       %45 = tt.splat %33 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %46 = tt.splat %33 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %47 = tt.splat %33 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:       %48 = tt.splat %33 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:       %49 = arith.addi %arg12, %arg15 : i32
// CHECK-NEXT:       %50 = arith.addi %arg12, %33 : i32
// CHECK-NEXT:       %51 = tt.splat %50 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %52 = tt.splat %50 {MetaUse} : i32 -> tensor<128x1xi32>
// CHECK-NEXT:       scf.for %arg17 = %43 to %44 step %1  : i32 {
// CHECK-NEXT:         %53 = arith.divsi %arg17, %c5_i32 : i32
// CHECK-NEXT:         %54 = arith.subi %53, %arg16 : i32
// CHECK-NEXT:         %55 = arith.remsi %arg17, %c5_i32 : i32
// CHECK-NEXT:         %56 = arith.muli %55, %c128_i32 : i32
// CHECK-NEXT:         %57 = tt.addptr %arg0, %56 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %58 = arith.muli %54, %c32_i32 : i32
// CHECK-NEXT:         %59 = arith.addi %arg15, %58 : i32
// CHECK-NEXT:         %60 = tt.splat %59 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %61 = arith.addi %60, %2 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %62 = tt.expand_dims %61 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %63 = arith.muli %62, %cst_10 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %64 = tt.splat %57 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %65 = tt.addptr %64, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %66 = tt.broadcast %65 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %67 = tt.addptr %66, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %68 = tt.addptr %arg3, %56 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %69 = tt.splat %68 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %70 = tt.addptr %69, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %71 = tt.broadcast %70 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %72 = tt.addptr %71, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %73 = tt.addptr %arg6, %56 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %74 = tt.splat %73 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %75 = tt.addptr %74, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %76 = tt.broadcast %75 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %77 = tt.addptr %76, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %78 = arith.muli %55, %arg13 : i32
// CHECK-NEXT:         %79 = tt.addptr %arg4, %78 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %80 = tt.splat %79 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %81 = tt.addptr %80, %61 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %82 = tt.addptr %arg5, %78 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %83 = tt.splat %82 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %84 = tt.addptr %83, %61 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %85 = arith.cmpi slt, %62, %45 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %86 = arith.cmpi slt, %61, %47 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %87 = tt.broadcast %85 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %88 = tt.load %67, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %89 = tt.load %72, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %90 = tt.load %84, %86, %cst_8 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %91 = tt.load %81, %86, %cst_8 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %92 = tt.splat %59 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:         %93 = arith.addi %92, %5 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %94 = arith.cmpi slt, %93, %46 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %95 = tt.splat %59 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:         %96 = arith.addi %95, %7 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %97 = arith.cmpi slt, %96, %48 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %98 = tt.broadcast %94 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %99 = tt.broadcast %97 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %100 = arith.andi %98, %99 {DataUse} : tensor<32x32xi1>
// CHECK-NEXT:         %101 = arith.divsi %55, %c5_i32 : i32
// CHECK-NEXT:         %102 = arith.muli %101, %c128_i32 : i32
// CHECK-NEXT:         %103 = tt.addptr %arg1, %102 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %104 = arith.muli %62, %cst_1 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %105 = tt.splat %103 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %106 = tt.addptr %105, %104 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %107 = tt.broadcast %106 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %108 = tt.addptr %107, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %109 = tt.addptr %arg2, %102 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %110 = tt.splat %109 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %111 = tt.addptr %110, %104 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %112 = tt.broadcast %111 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %113 = tt.addptr %112, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %114 = tt.load %108, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %115 = tt.trans %114 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %116 = tt.dot %88, %115, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %117 = arith.mulf %116, %26 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %118 = arith.uitofp %100 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
// CHECK-NEXT:         %119 = arith.subf %118, %cst_3 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %120 = arith.mulf %119, %cst_4 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %121 = arith.addf %117, %120 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %122 = arith.select %27, %121, %cst_6 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %123 = tt.expand_dims %90 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %124 = tt.broadcast %123 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %125 = arith.subf %122, %124 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %126 = math.exp %125 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %127 = tt.load %113, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %128 = tt.trans %127 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %129 = tt.dot %89, %128, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %130 = tt.expand_dims %91 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %131 = tt.broadcast %130 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %132 = arith.subf %129, %131 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %133 = arith.mulf %126, %132 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %134 = arith.truncf %133 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %135 = tt.dot %134, %114, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %136 = arith.mulf %135, %28 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %137 = arith.addf %136, %cst_9 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %138 = arith.addi %49, %58 : i32
// CHECK-NEXT:         %139 = tt.splat %138 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %140 = arith.addi %139, %2 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %141 = tt.expand_dims %140 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %142 = arith.muli %141, %cst_1 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %143 = tt.addptr %105, %142 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %144 = tt.broadcast %143 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %145 = tt.addptr %144, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %146 = tt.addptr %110, %142 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %147 = tt.broadcast %146 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %148 = tt.addptr %147, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %149 = arith.cmpi slt, %141, %51 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %150 = tt.broadcast %149 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %151 = tt.load %145, %150, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %152 = tt.trans %151 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %153 = tt.dot %88, %152, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %154 = arith.mulf %153, %26 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %155 = arith.addf %154, %120 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %156 = arith.select %29, %155, %cst_6 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %157 = arith.subf %156, %124 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %158 = math.exp %157 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %159 = tt.load %148, %150, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %160 = tt.trans %159 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %161 = tt.dot %89, %160, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %162 = arith.subf %161, %131 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %163 = arith.mulf %158, %162 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %164 = arith.truncf %163 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %165 = tt.dot %164, %151, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %166 = arith.mulf %165, %28 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %167 = arith.addf %137, %166 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %168 = arith.divsi %58, %c128_i32 : i32
// CHECK-NEXT:         %169 = arith.muli %168, %c128_i32 : i32
// CHECK-NEXT:         %170 = arith.divsi %169, %c32_i32 : i32
// CHECK-NEXT:         %171 = scf.for %arg18 = %170 to %54 step %c1_i32 iter_args(%arg19 = %167) -> (tensor<32x128xf32>)  : i32 {
// CHECK-NEXT:           %178 = arith.muli %arg18, %c32_i32 : i32
// CHECK-NEXT:           %179 = arith.addi %49, %178 : i32
// CHECK-NEXT:           %180 = tt.splat %179 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:           %181 = arith.addi %180, %2 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:           %182 = tt.expand_dims %181 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:           %183 = arith.muli %182, %cst_1 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %184 = tt.addptr %105, %183 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %185 = tt.broadcast %184 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %186 = tt.addptr %185, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %187 = tt.addptr %110, %183 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %188 = tt.broadcast %187 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %189 = tt.addptr %188, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %190 = arith.cmpi slt, %182, %51 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %191 = tt.broadcast %190 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:           %192 = tt.load %186, %191, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %193 = tt.trans %192 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:           %194 = tt.dot %88, %193, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %195 = arith.mulf %194, %26 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %196 = arith.subf %195, %124 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %197 = math.exp %196 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %198 = tt.load %189, %191, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %199 = tt.trans %198 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:           %200 = tt.dot %89, %199, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %201 = arith.subf %200, %131 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %202 = arith.mulf %197, %201 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %203 = arith.truncf %202 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %204 = tt.dot %203, %192, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %205 = arith.mulf %204, %28 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %206 = arith.addf %arg19, %205 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %206 : tensor<32x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %172 = tt.splat %103 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %173 = tt.splat %109 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %174 = tt.broadcast %123 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %175 = tt.broadcast %130 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %176 = scf.for %arg18 = %c0_i32 to %168 step %c1_i32 iter_args(%arg19 = %171) -> (tensor<32x128xf32>)  : i32 {
// CHECK-NEXT:           %178 = arith.muli %arg18, %c128_i32 : i32
// CHECK-NEXT:           %179 = arith.addi %49, %178 : i32
// CHECK-NEXT:           %180 = tt.splat %179 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %181 = arith.addi %180, %23 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %182 = tt.expand_dims %181 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %183 = arith.muli %182, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %184 = tt.addptr %172, %183 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %185 = tt.broadcast %184 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %186 = tt.addptr %185, %30 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %187 = tt.addptr %173, %183 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %188 = tt.broadcast %187 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %189 = tt.addptr %188, %30 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %190 = arith.cmpi slt, %182, %52 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %191 = tt.broadcast %190 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %192 = tt.load %186, %191, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %193 = tt.trans %192 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:           %194 = tt.dot %88, %193, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %195 = arith.mulf %194, %28 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %196 = arith.subf %195, %174 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %197 = math.exp %196 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %198 = tt.load %189, %191, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %199 = tt.trans %198 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:           %200 = tt.dot %89, %199, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %201 = arith.subf %200, %175 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %202 = arith.mulf %197, %201 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %203 = arith.truncf %202 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:           %204 = tt.dot %203, %192, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %205 = arith.mulf %204, %28 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %206 = arith.addf %arg19, %205 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %206 : tensor<32x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %177 = arith.truncf %176 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:         tt.store %77, %177, %87 : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %33, %37 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_bwd_q_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i32>, %arg8: i32, %arg9: f32, %arg10: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
    %cst = arith.constant dense<128> : tensor<128x1xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_1 = arith.constant dense<128> : tensor<32x1xi32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
    %cst_4 = arith.constant dense<1.000000e+06> : tensor<32x32xf32>
    %cst_5 = arith.constant dense<0> : tensor<32x32xi8>
    %cst_6 = arith.constant dense<-1.000000e+06> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    %cst_10 = arith.constant dense<640> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_11 = arith.constant dense<64> : tensor<32x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %4 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %5 = arith.muli %3, %cst_11 : tensor<32x1xi32>
    %6 = tt.splat %arg11 : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %7 = tt.addptr %6, %5 : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %8 = tt.broadcast %7 : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %9 = tt.broadcast %4 : tensor<1x32xi32> -> tensor<32x32xi32>
    %10 = tt.addptr %8, %9 : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %11 = tt.bitcast %10 : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %12 = tt.load %11 {was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %13 = tt.splat %arg10 : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %14 = tt.addptr %13, %5 : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %15 = tt.broadcast %14 : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %16 = tt.addptr %15, %9 : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %17 = tt.bitcast %16 : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %18 = tt.load %17 {was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %19 = arith.remsi %0, %1 : i32
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %21 = tt.expand_dims %20 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %22 = tt.broadcast %21 : tensor<1x128xi32> -> tensor<32x128xi32>
    %23 = tt.splat %arg9 : f32 -> tensor<32x32xf32>
    %24 = arith.cmpi ne, %12, %cst_5 : tensor<32x32xi8>
    %25 = tt.splat %arg9 : f32 -> tensor<32x128xf32>
    %26 = arith.cmpi ne, %18, %cst_5 : tensor<32x32xi8>
    %27 = tt.broadcast %21 : tensor<1x128xi32> -> tensor<128x128xi32>
    %28:2 = scf.for %arg14 = %c0_i32 to %arg8 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
      %29 = tt.addptr %arg7, %arg14 : !tt.ptr<i32>, i32
      %30 = tt.load %29 : !tt.ptr<i32>
      %31 = arith.subi %30, %arg15 : i32
      %32 = arith.addi %31, %c31_i32 : i32
      %33 = arith.divsi %32, %c32_i32 : i32
      %34 = arith.addi %arg16, %33 : i32
      %35 = arith.muli %arg16, %c5_i32 : i32
      %36 = arith.remsi %35, %1 : i32
      %37 = arith.subi %19, %36 : i32
      %38 = arith.addi %37, %1 : i32
      %39 = arith.remsi %38, %1 : i32
      %40 = arith.addi %35, %39 : i32
      %41 = arith.muli %34, %c5_i32 : i32
      %42 = tt.splat %30 : i32 -> tensor<32x1xi32>
      %43 = tt.splat %30 : i32 -> tensor<32xi32>
      %44 = tt.splat %30 : i32 -> tensor<1x32xi32>
      %45 = arith.addi %arg12, %arg15 : i32
      %46 = arith.addi %arg12, %30 : i32
      %47 = tt.splat %46 : i32 -> tensor<32x1xi32>
      %48 = tt.splat %46 : i32 -> tensor<128x1xi32>
      scf.for %arg17 = %40 to %41 step %1  : i32 {
        %49 = arith.divsi %arg17, %c5_i32 : i32
        %50 = arith.subi %49, %arg16 : i32
        %51 = arith.remsi %arg17, %c5_i32 : i32
        %52 = arith.muli %51, %c128_i32 : i32
        %53 = tt.addptr %arg0, %52 : !tt.ptr<bf16>, i32
        %54 = arith.muli %50, %c32_i32 : i32
        %55 = arith.addi %arg15, %54 : i32
        %56 = tt.splat %55 : i32 -> tensor<32xi32>
        %57 = arith.addi %56, %2 : tensor<32xi32>
        %58 = tt.expand_dims %57 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %59 = arith.muli %58, %cst_10 : tensor<32x1xi32>
        %60 = tt.splat %53 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %61 = tt.addptr %60, %59 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %62 = tt.broadcast %61 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %63 = tt.addptr %62, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %64 = tt.addptr %arg3, %52 : !tt.ptr<bf16>, i32
        %65 = tt.splat %64 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %66 = tt.addptr %65, %59 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %67 = tt.broadcast %66 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %68 = tt.addptr %67, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %69 = tt.addptr %arg6, %52 : !tt.ptr<bf16>, i32
        %70 = tt.splat %69 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %71 = tt.addptr %70, %59 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %72 = tt.broadcast %71 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %73 = tt.addptr %72, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %74 = arith.muli %51, %arg13 : i32
        %75 = tt.addptr %arg4, %74 : !tt.ptr<f32>, i32
        %76 = tt.splat %75 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
        %77 = tt.addptr %76, %57 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %78 = tt.addptr %arg5, %74 : !tt.ptr<f32>, i32
        %79 = tt.splat %78 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
        %80 = tt.addptr %79, %57 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %81 = arith.cmpi slt, %58, %42 : tensor<32x1xi32>
        %82 = arith.cmpi slt, %57, %43 : tensor<32xi32>
        %83 = tt.broadcast %81 : tensor<32x1xi1> -> tensor<32x128xi1>
        %84 = tt.load %63, %83, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
        %85 = tt.load %68, %83, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
        %86 = tt.load %80, %82, %cst_8 : tensor<32x!tt.ptr<f32>>
        %87 = tt.load %77, %82, %cst_8 : tensor<32x!tt.ptr<f32>>
        %88 = tt.splat %55 : i32 -> tensor<32x1xi32>
        %89 = arith.addi %88, %3 : tensor<32x1xi32>
        %90 = arith.cmpi slt, %89, %42 : tensor<32x1xi32>
        %91 = tt.splat %55 : i32 -> tensor<1x32xi32>
        %92 = arith.addi %91, %4 : tensor<1x32xi32>
        %93 = arith.cmpi slt, %92, %44 : tensor<1x32xi32>
        %94 = tt.broadcast %90 : tensor<32x1xi1> -> tensor<32x32xi1>
        %95 = tt.broadcast %93 : tensor<1x32xi1> -> tensor<32x32xi1>
        %96 = arith.andi %94, %95 : tensor<32x32xi1>
        %97 = arith.divsi %51, %c5_i32 : i32
        %98 = arith.muli %97, %c128_i32 : i32
        %99 = tt.addptr %arg1, %98 : !tt.ptr<bf16>, i32
        %100 = arith.muli %58, %cst_1 : tensor<32x1xi32>
        %101 = tt.splat %99 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %102 = tt.addptr %101, %100 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %103 = tt.broadcast %102 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %104 = tt.addptr %103, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %105 = tt.addptr %arg2, %98 : !tt.ptr<bf16>, i32
        %106 = tt.splat %105 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %107 = tt.addptr %106, %100 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %108 = tt.broadcast %107 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %109 = tt.addptr %108, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %110 = tt.load %104, %83, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
        %111 = tt.trans %110 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %112 = tt.dot %84, %111, %cst_2 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %113 = arith.mulf %112, %23 : tensor<32x32xf32>
        %114 = arith.uitofp %96 : tensor<32x32xi1> to tensor<32x32xf32>
        %115 = arith.subf %114, %cst_3 : tensor<32x32xf32>
        %116 = arith.mulf %115, %cst_4 : tensor<32x32xf32>
        %117 = arith.addf %113, %116 : tensor<32x32xf32>
        %118 = arith.select %24, %117, %cst_6 : tensor<32x32xi1>, tensor<32x32xf32>
        %119 = tt.expand_dims %86 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %120 = tt.broadcast %119 : tensor<32x1xf32> -> tensor<32x32xf32>
        %121 = arith.subf %118, %120 : tensor<32x32xf32>
        %122 = math.exp %121 : tensor<32x32xf32>
        %123 = tt.load %109, %83, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
        %124 = tt.trans %123 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %125 = tt.dot %85, %124, %cst_2 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %126 = tt.expand_dims %87 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %127 = tt.broadcast %126 : tensor<32x1xf32> -> tensor<32x32xf32>
        %128 = arith.subf %125, %127 : tensor<32x32xf32>
        %129 = arith.mulf %122, %128 : tensor<32x32xf32>
        %130 = arith.truncf %129 : tensor<32x32xf32> to tensor<32x32xbf16>
        %131 = tt.dot %130, %110, %cst_9 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %132 = arith.mulf %131, %25 : tensor<32x128xf32>
        %133 = arith.addf %132, %cst_9 : tensor<32x128xf32>
        %134 = arith.addi %45, %54 : i32
        %135 = tt.splat %134 : i32 -> tensor<32xi32>
        %136 = arith.addi %135, %2 : tensor<32xi32>
        %137 = tt.expand_dims %136 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %138 = arith.muli %137, %cst_1 : tensor<32x1xi32>
        %139 = tt.addptr %101, %138 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %140 = tt.broadcast %139 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %141 = tt.addptr %140, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %142 = tt.addptr %106, %138 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %143 = tt.broadcast %142 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %144 = tt.addptr %143, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %145 = arith.cmpi slt, %137, %47 : tensor<32x1xi32>
        %146 = tt.broadcast %145 : tensor<32x1xi1> -> tensor<32x128xi1>
        %147 = tt.load %141, %146, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
        %148 = tt.trans %147 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %149 = tt.dot %84, %148, %cst_2 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %150 = arith.mulf %149, %23 : tensor<32x32xf32>
        %151 = arith.addf %150, %116 : tensor<32x32xf32>
        %152 = arith.select %26, %151, %cst_6 : tensor<32x32xi1>, tensor<32x32xf32>
        %153 = arith.subf %152, %120 : tensor<32x32xf32>
        %154 = math.exp %153 : tensor<32x32xf32>
        %155 = tt.load %144, %146, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
        %156 = tt.trans %155 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %157 = tt.dot %85, %156, %cst_2 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %158 = arith.subf %157, %127 : tensor<32x32xf32>
        %159 = arith.mulf %154, %158 : tensor<32x32xf32>
        %160 = arith.truncf %159 : tensor<32x32xf32> to tensor<32x32xbf16>
        %161 = tt.dot %160, %147, %cst_9 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %162 = arith.mulf %161, %25 : tensor<32x128xf32>
        %163 = arith.addf %133, %162 : tensor<32x128xf32>
        %164 = arith.divsi %54, %c128_i32 : i32
        %165 = arith.muli %164, %c128_i32 : i32
        %166 = arith.divsi %165, %c32_i32 : i32
        %167 = scf.for %arg18 = %166 to %50 step %c1_i32 iter_args(%arg19 = %163) -> (tensor<32x128xf32>)  : i32 {
          %174 = arith.muli %arg18, %c32_i32 : i32
          %175 = arith.addi %45, %174 : i32
          %176 = tt.splat %175 : i32 -> tensor<32xi32>
          %177 = arith.addi %176, %2 : tensor<32xi32>
          %178 = tt.expand_dims %177 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
          %179 = arith.muli %178, %cst_1 : tensor<32x1xi32>
          %180 = tt.addptr %101, %179 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %181 = tt.broadcast %180 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %182 = tt.addptr %181, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %183 = tt.addptr %106, %179 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %184 = tt.broadcast %183 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %185 = tt.addptr %184, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %186 = arith.cmpi slt, %178, %47 : tensor<32x1xi32>
          %187 = tt.broadcast %186 : tensor<32x1xi1> -> tensor<32x128xi1>
          %188 = tt.load %182, %187, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
          %189 = tt.trans %188 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
          %190 = tt.dot %84, %189, %cst_2 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %191 = arith.mulf %190, %23 : tensor<32x32xf32>
          %192 = arith.subf %191, %120 : tensor<32x32xf32>
          %193 = math.exp %192 : tensor<32x32xf32>
          %194 = tt.load %185, %187, %cst_7 : tensor<32x128x!tt.ptr<bf16>>
          %195 = tt.trans %194 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
          %196 = tt.dot %85, %195, %cst_2 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %197 = arith.subf %196, %127 : tensor<32x32xf32>
          %198 = arith.mulf %193, %197 : tensor<32x32xf32>
          %199 = arith.truncf %198 : tensor<32x32xf32> to tensor<32x32xbf16>
          %200 = tt.dot %199, %188, %cst_9 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %201 = arith.mulf %200, %25 : tensor<32x128xf32>
          %202 = arith.addf %arg19, %201 : tensor<32x128xf32>
          scf.yield %202 : tensor<32x128xf32>
        }
        %168 = tt.splat %99 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %169 = tt.splat %105 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %170 = tt.broadcast %119 : tensor<32x1xf32> -> tensor<32x128xf32>
        %171 = tt.broadcast %126 : tensor<32x1xf32> -> tensor<32x128xf32>
        %172 = scf.for %arg18 = %c0_i32 to %164 step %c1_i32 iter_args(%arg19 = %167) -> (tensor<32x128xf32>)  : i32 {
          %174 = arith.muli %arg18, %c128_i32 : i32
          %175 = arith.addi %45, %174 : i32
          %176 = tt.splat %175 : i32 -> tensor<128xi32>
          %177 = arith.addi %176, %20 : tensor<128xi32>
          %178 = tt.expand_dims %177 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %179 = arith.muli %178, %cst : tensor<128x1xi32>
          %180 = tt.addptr %168, %179 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %181 = tt.broadcast %180 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %182 = tt.addptr %181, %27 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %183 = tt.addptr %169, %179 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %184 = tt.broadcast %183 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %185 = tt.addptr %184, %27 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %186 = arith.cmpi slt, %178, %48 : tensor<128x1xi32>
          %187 = tt.broadcast %186 : tensor<128x1xi1> -> tensor<128x128xi1>
          %188 = tt.load %182, %187, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
          %189 = tt.trans %188 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
          %190 = tt.dot %84, %189, %cst_9 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %191 = arith.mulf %190, %25 : tensor<32x128xf32>
          %192 = arith.subf %191, %170 : tensor<32x128xf32>
          %193 = math.exp %192 : tensor<32x128xf32>
          %194 = tt.load %185, %187, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
          %195 = tt.trans %194 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
          %196 = tt.dot %85, %195, %cst_9 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %197 = arith.subf %196, %171 : tensor<32x128xf32>
          %198 = arith.mulf %193, %197 : tensor<32x128xf32>
          %199 = arith.truncf %198 : tensor<32x128xf32> to tensor<32x128xbf16>
          %200 = tt.dot %199, %188, %cst_9 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %201 = arith.mulf %200, %25 : tensor<32x128xf32>
          %202 = arith.addf %arg19, %201 : tensor<32x128xf32>
          scf.yield %202 : tensor<32x128xf32>
        }
        %173 = arith.truncf %172 : tensor<32x128xf32> to tensor<32x128xbf16>
        tt.store %73, %173, %83 : tensor<32x128x!tt.ptr<bf16>>
      }
      scf.yield %30, %34 : i32, i32
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/dllm/fwd_u_norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_fwd_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32>, %arg6: i32, %arg7: f32, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x1xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32xf32>
// CHECK-NEXT:     %cst_10 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %3 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %8 = arith.muli %4, %cst_13 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:     %9 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %10 = tt.addptr %9, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %11 = tt.broadcast %10 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %12 = tt.broadcast %6 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
// CHECK-NEXT:     %13 = tt.addptr %11, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %14 = tt.bitcast %13 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %16 = tt.splat %arg9 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %17 = tt.addptr %16, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %18 = tt.broadcast %17 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %19 = tt.addptr %18, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %20 = tt.bitcast %19 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %21 = tt.load %20 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %22 = arith.remsi %0, %1 : i32
// CHECK-NEXT:     %23 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %24 = tt.expand_dims %23 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %25 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %26 = tt.splat %arg7 {DataUse} : f32 -> tensor<32x32xf32>
// CHECK-NEXT:     %27 = arith.cmpi ne, %15, %cst_6 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %28 = arith.cmpi ne, %21, %cst_6 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %29 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %30 = tt.splat %arg7 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     %31:2 = scf.for %arg12 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %32 = tt.addptr %arg5, %arg12 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %33 = tt.load %32 : !tt.ptr<i32>
// CHECK-NEXT:       %34 = arith.subi %33, %arg13 : i32
// CHECK-NEXT:       %35 = arith.addi %34, %c31_i32 : i32
// CHECK-NEXT:       %36 = arith.divsi %35, %c32_i32 : i32
// CHECK-NEXT:       %37 = arith.addi %arg14, %36 : i32
// CHECK-NEXT:       %38 = arith.muli %arg14, %c5_i32 : i32
// CHECK-NEXT:       %39 = arith.remsi %38, %1 : i32
// CHECK-NEXT:       %40 = arith.subi %22, %39 : i32
// CHECK-NEXT:       %41 = arith.addi %40, %1 : i32
// CHECK-NEXT:       %42 = arith.remsi %41, %1 : i32
// CHECK-NEXT:       %43 = arith.addi %38, %42 : i32
// CHECK-NEXT:       %44 = arith.muli %37, %c5_i32 : i32
// CHECK-NEXT:       %45 = tt.splat %33 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %46 = tt.splat %33 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %47 = tt.splat %33 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:       %48 = tt.splat %33 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:       %49 = arith.addi %arg10, %arg13 : i32
// CHECK-NEXT:       %50 = arith.addi %arg10, %33 : i32
// CHECK-NEXT:       %51 = tt.splat %50 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %52 = tt.splat %50 {MetaUse} : i32 -> tensor<128x1xi32>
// CHECK-NEXT:       scf.for %arg15 = %43 to %44 step %1  : i32 {
// CHECK-NEXT:         %53 = arith.divsi %arg15, %c5_i32 : i32
// CHECK-NEXT:         %54 = arith.subi %53, %arg14 : i32
// CHECK-NEXT:         %55 = arith.remsi %arg15, %c5_i32 : i32
// CHECK-NEXT:         %56 = arith.muli %55, %c128_i32 : i32
// CHECK-NEXT:         %57 = tt.addptr %arg0, %56 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %58 = arith.muli %54, %c32_i32 : i32
// CHECK-NEXT:         %59 = arith.addi %arg13, %58 : i32
// CHECK-NEXT:         %60 = tt.splat %59 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %61 = arith.addi %60, %2 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %62 = tt.expand_dims %61 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %63 = arith.muli %62, %cst_12 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %64 = tt.splat %57 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %65 = tt.addptr %64, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %66 = tt.broadcast %65 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %67 = tt.addptr %66, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %68 = tt.addptr %arg3, %56 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %69 = tt.splat %68 {MetaUse} : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
// CHECK-NEXT:         %70 = tt.addptr %69, %63 {MetaUse} : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
// CHECK-NEXT:         %71 = tt.broadcast %70 {MetaUse} : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
// CHECK-NEXT:         %72 = tt.addptr %71, %25 {MetaUse} : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
// CHECK-NEXT:         %73 = arith.muli %55, %arg11 : i32
// CHECK-NEXT:         %74 = tt.addptr %arg4, %73 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %75 = tt.splat %74 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %76 = tt.addptr %75, %61 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %77 = arith.cmpi slt, %62, %45 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %78 = arith.cmpi slt, %61, %47 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %79 = tt.broadcast %77 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %80 = tt.load %67, %79, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %81 = tt.splat %59 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:         %82 = arith.addi %81, %5 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %83 = arith.cmpi slt, %82, %46 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %84 = tt.splat %59 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:         %85 = arith.addi %84, %7 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %86 = arith.cmpi slt, %85, %48 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %87 = tt.broadcast %83 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %88 = tt.broadcast %86 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %89 = arith.andi %87, %88 {DataUse} : tensor<32x32xi1>
// CHECK-NEXT:         %90 = arith.divsi %55, %c5_i32 : i32
// CHECK-NEXT:         %91 = arith.muli %90, %c128_i32 : i32
// CHECK-NEXT:         %92 = tt.addptr %arg1, %91 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %93 = arith.muli %62, %cst_2 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %94 = tt.splat %92 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %95 = tt.addptr %94, %93 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %96 = tt.broadcast %95 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %97 = tt.addptr %96, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %98 = tt.addptr %arg2, %91 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %99 = tt.splat %98 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %100 = tt.addptr %99, %93 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %101 = tt.broadcast %100 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %102 = tt.addptr %101, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %103 = tt.load %97, %79, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %104 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %105 = tt.dot %80, %104, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %106 = arith.mulf %105, %26 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %107 = tt.load %102, %79, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %108 = arith.uitofp %89 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
// CHECK-NEXT:         %109 = arith.subf %108, %cst_4 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %110 = arith.mulf %109, %cst_5 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %111 = arith.addf %106, %110 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %112 = arith.select %27, %111, %cst_7 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %113 = "tt.reduce"(%112) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %179 = arith.maxnumf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %179 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %114 = arith.maxnumf %113, %cst_9 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %116 = tt.broadcast %115 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %117 = arith.subf %112, %116 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %118 = math.exp %117 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %119 = arith.subf %cst_9, %114 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %120 = math.exp %119 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %121 = arith.mulf %120, %cst_10 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %122 = "tt.reduce"(%118) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %179 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %179 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %123 = arith.addf %121, %122 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %124 = tt.expand_dims %120 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %125 = arith.mulf %124, %cst {DataUse} : tensor<32x1xf32>
// CHECK-NEXT:         %126 = tt.broadcast %125 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %127 = arith.truncf %118 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %128 = tt.dot %127, %107, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %129 = arith.addf %128, %126 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:         %130 = arith.addi %49, %58 : i32
// CHECK-NEXT:         %131 = tt.splat %130 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %132 = arith.addi %131, %2 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %133 = tt.expand_dims %132 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %134 = arith.muli %133, %cst_2 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %135 = tt.addptr %94, %134 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %136 = tt.broadcast %135 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %137 = tt.addptr %136, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %138 = tt.addptr %99, %134 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %139 = tt.broadcast %138 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %140 = tt.addptr %139, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %141 = arith.cmpi slt, %133, %51 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %142 = tt.broadcast %141 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %143 = tt.load %137, %142, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %144 = tt.trans %143 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %145 = tt.dot %80, %144, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %146 = arith.mulf %145, %26 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %147 = tt.load %140, %142, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %148 = arith.addf %146, %110 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %149 = arith.select %28, %148, %cst_7 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %150 = "tt.reduce"(%149) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %179 = arith.maxnumf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %179 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %151 = arith.maxnumf %114, %150 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %152 = tt.expand_dims %151 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %153 = tt.broadcast %152 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %154 = arith.subf %149, %153 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %155 = math.exp %154 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %156 = arith.subf %114, %151 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %157 = math.exp %156 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %158 = arith.mulf %157, %123 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %159 = "tt.reduce"(%155) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %179 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %179 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %160 = arith.addf %158, %159 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %161 = tt.expand_dims %157 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %162 = tt.broadcast %161 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %163 = arith.mulf %162, %129 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %164 = arith.truncf %155 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %165 = tt.dot %164, %147, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %166 = arith.addf %165, %163 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:         %167 = arith.divsi %58, %c128_i32 : i32
// CHECK-NEXT:         %168 = arith.muli %167, %c128_i32 : i32
// CHECK-NEXT:         %169 = arith.divsi %168, %c32_i32 : i32
// CHECK-NEXT:         %170:3 = scf.for %arg16 = %169 to %54 step %c1_i32 iter_args(%arg17 = %166, %arg18 = %151, %arg19 = %160) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
// CHECK-NEXT:           %179 = arith.muli %arg16, %c32_i32 : i32
// CHECK-NEXT:           %180 = arith.addi %49, %179 : i32
// CHECK-NEXT:           %181 = tt.splat %180 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:           %182 = arith.addi %181, %2 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:           %183 = tt.expand_dims %182 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:           %184 = arith.muli %183, %cst_2 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %185 = tt.addptr %94, %184 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %186 = tt.broadcast %185 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %187 = tt.addptr %186, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %188 = tt.addptr %99, %184 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %189 = tt.broadcast %188 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %190 = tt.addptr %189, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %191 = arith.cmpi slt, %183, %51 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %192 = tt.broadcast %191 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:           %193 = tt.load %187, %192, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %194 = tt.trans %193 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:           %195 = tt.dot %80, %194, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %196 = arith.mulf %195, %26 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %197 = tt.load %190, %192, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %198 = "tt.reduce"(%196) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %215 = arith.maxnumf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %215 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %199 = arith.maxnumf %arg18, %198 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %200 = tt.expand_dims %199 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %201 = tt.broadcast %200 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:           %202 = arith.subf %196, %201 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %203 = math.exp %202 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %204 = arith.subf %arg18, %199 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %205 = math.exp %204 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %206 = arith.mulf %205, %arg19 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %207 = "tt.reduce"(%203) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %215 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %215 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %208 = arith.addf %206, %207 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %209 = tt.expand_dims %205 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %210 = tt.broadcast %209 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:           %211 = arith.mulf %210, %arg17 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %212 = arith.truncf %203 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %213 = tt.dot %212, %197, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %214 = arith.addf %213, %211 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %214, %199, %208 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %171 = tt.splat %92 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %172 = tt.splat %98 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %173:3 = scf.for %arg16 = %c0_i32 to %167 step %c1_i32 iter_args(%arg17 = %170#0, %arg18 = %170#1, %arg19 = %170#2) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
// CHECK-NEXT:           %179 = arith.muli %arg16, %c128_i32 : i32
// CHECK-NEXT:           %180 = arith.addi %49, %179 : i32
// CHECK-NEXT:           %181 = tt.splat %180 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %182 = arith.addi %181, %23 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %183 = tt.expand_dims %182 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %184 = arith.muli %183, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %185 = tt.addptr %171, %184 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %186 = tt.broadcast %185 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %187 = tt.addptr %186, %29 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %188 = tt.addptr %172, %184 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %189 = tt.broadcast %188 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %190 = tt.addptr %189, %29 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %191 = arith.cmpi slt, %183, %52 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %192 = tt.broadcast %191 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %193 = tt.load %187, %192, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %194 = tt.trans %193 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:           %195 = tt.dot %80, %194, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %196 = arith.mulf %195, %30 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %197 = tt.load %190, %192, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %198 = "tt.reduce"(%196) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %215 = arith.maxnumf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %215 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x128xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %199 = arith.maxnumf %arg18, %198 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %200 = tt.expand_dims %199 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %201 = tt.broadcast %200 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:           %202 = arith.subf %196, %201 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %203 = math.exp %202 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %204 = arith.subf %arg18, %199 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %205 = math.exp %204 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %206 = arith.mulf %205, %arg19 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %207 = "tt.reduce"(%203) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %215 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %215 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x128xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %208 = arith.addf %206, %207 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %209 = tt.expand_dims %205 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %210 = tt.broadcast %209 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:           %211 = arith.mulf %210, %arg17 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %212 = arith.truncf %203 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:           %213 = tt.dot %212, %197, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %214 = arith.addf %213, %211 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %214, %199, %208 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %174 = tt.expand_dims %173#2 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %175 = tt.broadcast %174 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %176 = arith.divf %173#0, %175 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %177 = math.log %173#2 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %178 = arith.addf %177, %173#1 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         tt.store %72, %176, %79 : tensor<32x128x!tt.ptr<f32>>
// CHECK-NEXT:         tt.store %76, %178, %78 : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %33, %37 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_fwd_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32>, %arg6: i32, %arg7: f32, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x1xf32>
    %cst_0 = arith.constant dense<128> : tensor<128x1xi32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_2 = arith.constant dense<128> : tensor<32x1xi32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
    %cst_5 = arith.constant dense<1.000000e+06> : tensor<32x32xf32>
    %cst_6 = arith.constant dense<0> : tensor<32x32xi8>
    %cst_7 = arith.constant dense<-1.000000e+06> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_9 = arith.constant dense<-1.000000e+06> : tensor<32xf32>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    %cst_12 = arith.constant dense<640> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_13 = arith.constant dense<64> : tensor<32x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %4 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %5 = arith.muli %3, %cst_13 : tensor<32x1xi32>
    %6 = tt.splat %arg8 : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %7 = tt.addptr %6, %5 : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %8 = tt.broadcast %7 : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %9 = tt.broadcast %4 : tensor<1x32xi32> -> tensor<32x32xi32>
    %10 = tt.addptr %8, %9 : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %11 = tt.bitcast %10 : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %12 = tt.load %11 {was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %13 = tt.splat %arg9 : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %14 = tt.addptr %13, %5 : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %15 = tt.broadcast %14 : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %16 = tt.addptr %15, %9 : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %17 = tt.bitcast %16 : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %18 = tt.load %17 {was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %19 = arith.remsi %0, %1 : i32
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %21 = tt.expand_dims %20 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %22 = tt.broadcast %21 : tensor<1x128xi32> -> tensor<32x128xi32>
    %23 = tt.splat %arg7 : f32 -> tensor<32x32xf32>
    %24 = arith.cmpi ne, %12, %cst_6 : tensor<32x32xi8>
    %25 = arith.cmpi ne, %18, %cst_6 : tensor<32x32xi8>
    %26 = tt.broadcast %21 : tensor<1x128xi32> -> tensor<128x128xi32>
    %27 = tt.splat %arg7 : f32 -> tensor<32x128xf32>
    %28:2 = scf.for %arg12 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
      %29 = tt.addptr %arg5, %arg12 : !tt.ptr<i32>, i32
      %30 = tt.load %29 : !tt.ptr<i32>
      %31 = arith.subi %30, %arg13 : i32
      %32 = arith.addi %31, %c31_i32 : i32
      %33 = arith.divsi %32, %c32_i32 : i32
      %34 = arith.addi %arg14, %33 : i32
      %35 = arith.muli %arg14, %c5_i32 : i32
      %36 = arith.remsi %35, %1 : i32
      %37 = arith.subi %19, %36 : i32
      %38 = arith.addi %37, %1 : i32
      %39 = arith.remsi %38, %1 : i32
      %40 = arith.addi %35, %39 : i32
      %41 = arith.muli %34, %c5_i32 : i32
      %42 = tt.splat %30 : i32 -> tensor<32x1xi32>
      %43 = tt.splat %30 : i32 -> tensor<32xi32>
      %44 = tt.splat %30 : i32 -> tensor<1x32xi32>
      %45 = arith.addi %arg10, %arg13 : i32
      %46 = arith.addi %arg10, %30 : i32
      %47 = tt.splat %46 : i32 -> tensor<32x1xi32>
      %48 = tt.splat %46 : i32 -> tensor<128x1xi32>
      scf.for %arg15 = %40 to %41 step %1  : i32 {
        %49 = arith.divsi %arg15, %c5_i32 : i32
        %50 = arith.subi %49, %arg14 : i32
        %51 = arith.remsi %arg15, %c5_i32 : i32
        %52 = arith.muli %51, %c128_i32 : i32
        %53 = tt.addptr %arg0, %52 : !tt.ptr<bf16>, i32
        %54 = arith.muli %50, %c32_i32 : i32
        %55 = arith.addi %arg13, %54 : i32
        %56 = tt.splat %55 : i32 -> tensor<32xi32>
        %57 = arith.addi %56, %2 : tensor<32xi32>
        %58 = tt.expand_dims %57 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %59 = arith.muli %58, %cst_12 : tensor<32x1xi32>
        %60 = tt.splat %53 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %61 = tt.addptr %60, %59 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %62 = tt.broadcast %61 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %63 = tt.addptr %62, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %64 = tt.addptr %arg3, %52 : !tt.ptr<f32>, i32
        %65 = tt.splat %64 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
        %66 = tt.addptr %65, %59 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
        %67 = tt.broadcast %66 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
        %68 = tt.addptr %67, %22 : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
        %69 = arith.muli %51, %arg11 : i32
        %70 = tt.addptr %arg4, %69 : !tt.ptr<f32>, i32
        %71 = tt.splat %70 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
        %72 = tt.addptr %71, %57 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %73 = arith.cmpi slt, %58, %42 : tensor<32x1xi32>
        %74 = arith.cmpi slt, %57, %43 : tensor<32xi32>
        %75 = tt.broadcast %73 : tensor<32x1xi1> -> tensor<32x128xi1>
        %76 = tt.load %63, %75, %cst_8 : tensor<32x128x!tt.ptr<bf16>>
        %77 = tt.splat %55 : i32 -> tensor<32x1xi32>
        %78 = arith.addi %77, %3 : tensor<32x1xi32>
        %79 = arith.cmpi slt, %78, %42 : tensor<32x1xi32>
        %80 = tt.splat %55 : i32 -> tensor<1x32xi32>
        %81 = arith.addi %80, %4 : tensor<1x32xi32>
        %82 = arith.cmpi slt, %81, %44 : tensor<1x32xi32>
        %83 = tt.broadcast %79 : tensor<32x1xi1> -> tensor<32x32xi1>
        %84 = tt.broadcast %82 : tensor<1x32xi1> -> tensor<32x32xi1>
        %85 = arith.andi %83, %84 : tensor<32x32xi1>
        %86 = arith.divsi %51, %c5_i32 : i32
        %87 = arith.muli %86, %c128_i32 : i32
        %88 = tt.addptr %arg1, %87 : !tt.ptr<bf16>, i32
        %89 = arith.muli %58, %cst_2 : tensor<32x1xi32>
        %90 = tt.splat %88 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %91 = tt.addptr %90, %89 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %92 = tt.broadcast %91 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %93 = tt.addptr %92, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %94 = tt.addptr %arg2, %87 : !tt.ptr<bf16>, i32
        %95 = tt.splat %94 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %96 = tt.addptr %95, %89 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %97 = tt.broadcast %96 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %98 = tt.addptr %97, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %99 = tt.load %93, %75, %cst_8 : tensor<32x128x!tt.ptr<bf16>>
        %100 = tt.trans %99 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %101 = tt.dot %76, %100, %cst_3 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %102 = arith.mulf %101, %23 : tensor<32x32xf32>
        %103 = tt.load %98, %75, %cst_8 : tensor<32x128x!tt.ptr<bf16>>
        %104 = arith.uitofp %85 : tensor<32x32xi1> to tensor<32x32xf32>
        %105 = arith.subf %104, %cst_4 : tensor<32x32xf32>
        %106 = arith.mulf %105, %cst_5 : tensor<32x32xf32>
        %107 = arith.addf %102, %106 : tensor<32x32xf32>
        %108 = arith.select %24, %107, %cst_7 : tensor<32x32xi1>, tensor<32x32xf32>
        %109 = "tt.reduce"(%108) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %175 = arith.maxnumf %arg16, %arg17 : f32
          tt.reduce.return %175 : f32
        }) : (tensor<32x32xf32>) -> tensor<32xf32>
        %110 = arith.maxnumf %109, %cst_9 : tensor<32xf32>
        %111 = tt.expand_dims %110 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %112 = tt.broadcast %111 : tensor<32x1xf32> -> tensor<32x32xf32>
        %113 = arith.subf %108, %112 : tensor<32x32xf32>
        %114 = math.exp %113 : tensor<32x32xf32>
        %115 = arith.subf %cst_9, %110 : tensor<32xf32>
        %116 = math.exp %115 : tensor<32xf32>
        %117 = arith.mulf %116, %cst_10 : tensor<32xf32>
        %118 = "tt.reduce"(%114) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %175 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %175 : f32
        }) : (tensor<32x32xf32>) -> tensor<32xf32>
        %119 = arith.addf %117, %118 : tensor<32xf32>
        %120 = tt.expand_dims %116 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %121 = arith.mulf %120, %cst : tensor<32x1xf32>
        %122 = tt.broadcast %121 : tensor<32x1xf32> -> tensor<32x128xf32>
        %123 = arith.truncf %114 : tensor<32x32xf32> to tensor<32x32xbf16>
        %124 = tt.dot %123, %103, %cst_11 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %125 = arith.addf %124, %122 {triton_cv12.add_from_dot} : tensor<32x128xf32>
        %126 = arith.addi %45, %54 : i32
        %127 = tt.splat %126 : i32 -> tensor<32xi32>
        %128 = arith.addi %127, %2 : tensor<32xi32>
        %129 = tt.expand_dims %128 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %130 = arith.muli %129, %cst_2 : tensor<32x1xi32>
        %131 = tt.addptr %90, %130 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %132 = tt.broadcast %131 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %133 = tt.addptr %132, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %134 = tt.addptr %95, %130 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %135 = tt.broadcast %134 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %136 = tt.addptr %135, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %137 = arith.cmpi slt, %129, %47 : tensor<32x1xi32>
        %138 = tt.broadcast %137 : tensor<32x1xi1> -> tensor<32x128xi1>
        %139 = tt.load %133, %138, %cst_8 : tensor<32x128x!tt.ptr<bf16>>
        %140 = tt.trans %139 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %141 = tt.dot %76, %140, %cst_3 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %142 = arith.mulf %141, %23 : tensor<32x32xf32>
        %143 = tt.load %136, %138, %cst_8 : tensor<32x128x!tt.ptr<bf16>>
        %144 = arith.addf %142, %106 : tensor<32x32xf32>
        %145 = arith.select %25, %144, %cst_7 : tensor<32x32xi1>, tensor<32x32xf32>
        %146 = "tt.reduce"(%145) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %175 = arith.maxnumf %arg16, %arg17 : f32
          tt.reduce.return %175 : f32
        }) : (tensor<32x32xf32>) -> tensor<32xf32>
        %147 = arith.maxnumf %110, %146 : tensor<32xf32>
        %148 = tt.expand_dims %147 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %149 = tt.broadcast %148 : tensor<32x1xf32> -> tensor<32x32xf32>
        %150 = arith.subf %145, %149 : tensor<32x32xf32>
        %151 = math.exp %150 : tensor<32x32xf32>
        %152 = arith.subf %110, %147 : tensor<32xf32>
        %153 = math.exp %152 : tensor<32xf32>
        %154 = arith.mulf %153, %119 : tensor<32xf32>
        %155 = "tt.reduce"(%151) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %175 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %175 : f32
        }) : (tensor<32x32xf32>) -> tensor<32xf32>
        %156 = arith.addf %154, %155 : tensor<32xf32>
        %157 = tt.expand_dims %153 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %158 = tt.broadcast %157 : tensor<32x1xf32> -> tensor<32x128xf32>
        %159 = arith.mulf %158, %125 : tensor<32x128xf32>
        %160 = arith.truncf %151 : tensor<32x32xf32> to tensor<32x32xbf16>
        %161 = tt.dot %160, %143, %cst_11 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %162 = arith.addf %161, %159 {triton_cv12.add_from_dot} : tensor<32x128xf32>
        %163 = arith.divsi %54, %c128_i32 : i32
        %164 = arith.muli %163, %c128_i32 : i32
        %165 = arith.divsi %164, %c32_i32 : i32
        %166:3 = scf.for %arg16 = %165 to %50 step %c1_i32 iter_args(%arg17 = %162, %arg18 = %147, %arg19 = %156) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
          %175 = arith.muli %arg16, %c32_i32 : i32
          %176 = arith.addi %45, %175 : i32
          %177 = tt.splat %176 : i32 -> tensor<32xi32>
          %178 = arith.addi %177, %2 : tensor<32xi32>
          %179 = tt.expand_dims %178 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
          %180 = arith.muli %179, %cst_2 : tensor<32x1xi32>
          %181 = tt.addptr %90, %180 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %182 = tt.broadcast %181 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %183 = tt.addptr %182, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %184 = tt.addptr %95, %180 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %185 = tt.broadcast %184 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %186 = tt.addptr %185, %22 : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %187 = arith.cmpi slt, %179, %47 : tensor<32x1xi32>
          %188 = tt.broadcast %187 : tensor<32x1xi1> -> tensor<32x128xi1>
          %189 = tt.load %183, %188, %cst_8 : tensor<32x128x!tt.ptr<bf16>>
          %190 = tt.trans %189 {order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
          %191 = tt.dot %76, %190, %cst_3 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %192 = arith.mulf %191, %23 : tensor<32x32xf32>
          %193 = tt.load %186, %188, %cst_8 : tensor<32x128x!tt.ptr<bf16>>
          %194 = "tt.reduce"(%192) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %211 = arith.maxnumf %arg20, %arg21 : f32
            tt.reduce.return %211 : f32
          }) : (tensor<32x32xf32>) -> tensor<32xf32>
          %195 = arith.maxnumf %arg18, %194 : tensor<32xf32>
          %196 = tt.expand_dims %195 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %197 = tt.broadcast %196 : tensor<32x1xf32> -> tensor<32x32xf32>
          %198 = arith.subf %192, %197 : tensor<32x32xf32>
          %199 = math.exp %198 : tensor<32x32xf32>
          %200 = arith.subf %arg18, %195 : tensor<32xf32>
          %201 = math.exp %200 : tensor<32xf32>
          %202 = arith.mulf %201, %arg19 : tensor<32xf32>
          %203 = "tt.reduce"(%199) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %211 = arith.addf %arg20, %arg21 : f32
            tt.reduce.return %211 : f32
          }) : (tensor<32x32xf32>) -> tensor<32xf32>
          %204 = arith.addf %202, %203 : tensor<32xf32>
          %205 = tt.expand_dims %201 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %206 = tt.broadcast %205 : tensor<32x1xf32> -> tensor<32x128xf32>
          %207 = arith.mulf %206, %arg17 : tensor<32x128xf32>
          %208 = arith.truncf %199 : tensor<32x32xf32> to tensor<32x32xbf16>
          %209 = tt.dot %208, %193, %cst_11 {triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %210 = arith.addf %209, %207 {triton_cv12.add_from_dot} : tensor<32x128xf32>
          scf.yield %210, %195, %204 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
        }
        %167 = tt.splat %88 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %168 = tt.splat %94 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %169:3 = scf.for %arg16 = %c0_i32 to %163 step %c1_i32 iter_args(%arg17 = %166#0, %arg18 = %166#1, %arg19 = %166#2) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
          %175 = arith.muli %arg16, %c128_i32 : i32
          %176 = arith.addi %45, %175 : i32
          %177 = tt.splat %176 : i32 -> tensor<128xi32>
          %178 = arith.addi %177, %20 : tensor<128xi32>
          %179 = tt.expand_dims %178 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %180 = arith.muli %179, %cst_0 : tensor<128x1xi32>
          %181 = tt.addptr %167, %180 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %182 = tt.broadcast %181 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %183 = tt.addptr %182, %26 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %184 = tt.addptr %168, %180 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %185 = tt.broadcast %184 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %186 = tt.addptr %185, %26 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %187 = arith.cmpi slt, %179, %48 : tensor<128x1xi32>
          %188 = tt.broadcast %187 : tensor<128x1xi1> -> tensor<128x128xi1>
          %189 = tt.load %183, %188, %cst_1 : tensor<128x128x!tt.ptr<bf16>>
          %190 = tt.trans %189 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
          %191 = tt.dot %76, %190, %cst_11 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %192 = arith.mulf %191, %27 : tensor<32x128xf32>
          %193 = tt.load %186, %188, %cst_1 : tensor<128x128x!tt.ptr<bf16>>
          %194 = "tt.reduce"(%192) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %211 = arith.maxnumf %arg20, %arg21 : f32
            tt.reduce.return %211 : f32
          }) : (tensor<32x128xf32>) -> tensor<32xf32>
          %195 = arith.maxnumf %arg18, %194 : tensor<32xf32>
          %196 = tt.expand_dims %195 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %197 = tt.broadcast %196 : tensor<32x1xf32> -> tensor<32x128xf32>
          %198 = arith.subf %192, %197 : tensor<32x128xf32>
          %199 = math.exp %198 : tensor<32x128xf32>
          %200 = arith.subf %arg18, %195 : tensor<32xf32>
          %201 = math.exp %200 : tensor<32xf32>
          %202 = arith.mulf %201, %arg19 : tensor<32xf32>
          %203 = "tt.reduce"(%199) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %211 = arith.addf %arg20, %arg21 : f32
            tt.reduce.return %211 : f32
          }) : (tensor<32x128xf32>) -> tensor<32xf32>
          %204 = arith.addf %202, %203 : tensor<32xf32>
          %205 = tt.expand_dims %201 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %206 = tt.broadcast %205 : tensor<32x1xf32> -> tensor<32x128xf32>
          %207 = arith.mulf %206, %arg17 : tensor<32x128xf32>
          %208 = arith.truncf %199 : tensor<32x128xf32> to tensor<32x128xbf16>
          %209 = tt.dot %208, %193, %cst_11 {triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %210 = arith.addf %209, %207 {triton_cv12.add_from_dot} : tensor<32x128xf32>
          scf.yield %210, %195, %204 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
        }
        %170 = tt.expand_dims %169#2 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %171 = tt.broadcast %170 : tensor<32x1xf32> -> tensor<32x128xf32>
        %172 = arith.divf %169#0, %171 : tensor<32x128xf32>
        %173 = math.log %169#2 : tensor<32xf32>
        %174 = arith.addf %173, %169#1 : tensor<32xf32>
        tt.store %68, %172, %75 : tensor<32x128x!tt.ptr<f32>>
        tt.store %72, %174, %74 : tensor<32x!tt.ptr<f32>>
      }
      scf.yield %30, %34 : i32, i32
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/dv_local/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_bwd_kernel_dv_local(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: f32, %arg6: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = arith.muli %1, %arg6 : i32
// CHECK-NEXT:     %3 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:     %4 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %5 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %6 = tt.splat %3 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %7 = tt.splat %3 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %8 = arith.addi %6, %4 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %9 = arith.addi %7, %5 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %10 = tt.splat %arg6 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %11 = tt.splat %arg6 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %12 = arith.cmpi slt, %8, %10 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %13 = arith.cmpi slt, %9, %11 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %14 = arith.muli %2, %c32_i32 : i32
// CHECK-NEXT:     %15 = arith.muli %1, %c32_i32 : i32
// CHECK-NEXT:     %16 = arith.extsi %arg6 : i32 to i64
// CHECK-NEXT:     %17 = tt.splat %arg5 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     %18 = tt.expand_dims %9 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %19 = tt.expand_dims %9 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %20 = tt.broadcast %18 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %21 = tt.broadcast %19 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %22 = arith.cmpi sle, %20, %21 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %23 = tt.expand_dims %13 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %24 = tt.expand_dims %13 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %25 = tt.broadcast %23 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %26 = tt.broadcast %24 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %27 = arith.andi %25, %26 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %28 = arith.andi %22, %27 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
// CHECK-NEXT:       %29 = arith.addi %14, %arg7 : i32
// CHECK-NEXT:       %30 = arith.muli %29, %c128_i32 : i32
// CHECK-NEXT:       %31 = tt.addptr %arg0, %30 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %32 = tt.addptr %arg1, %30 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %33 = tt.addptr %arg3, %30 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %34 = tt.addptr %arg4, %30 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %35 = arith.addi %15, %arg7 : i32
// CHECK-NEXT:       %36 = arith.muli %35, %arg6 : i32
// CHECK-NEXT:       %37 = tt.addptr %arg2, %36 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %38 = tt.splat %37 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %39 = tt.addptr %38, %8 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %40 = tt.load %39, %12, %cst {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %41 = arith.extf %40 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:       %42 = tt.make_tensor_ptr %32, [%16, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %43 = tt.make_tensor_ptr %31, [%c128_i64, %16], [%c1_i64, %c4096_i64], [%c0_i32, %3] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:       %44 = tt.load %42 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %45 = tt.load %43 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:       %46 = tt.dot %44, %45, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %47 = arith.mulf %46, %17 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %48 = arith.addf %47, %cst_1 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %49 = tt.expand_dims %41 {DataUse, axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
// CHECK-NEXT:       %50 = tt.expand_dims %41 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %51 = tt.broadcast %49 {DataUse} : tensor<1x64xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %52 = tt.broadcast %50 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %53 = arith.subf %51, %52 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %54 = math.exp %53 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %55 = arith.mulf %48, %54 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %56 = arith.select %28, %55, %cst_1 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:       %57 = arith.truncf %56 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %58 = tt.make_tensor_ptr %33, [%16, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %59 = tt.make_tensor_ptr %34, [%16, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %60 = tt.load %58 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %61 = tt.dot %57, %60, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:       %62 = arith.truncf %61 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %59, %62 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_bwd_kernel_dv_local(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: f32, %arg6: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c1_i64 = arith.constant 1 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg6 : i32
    %3 = arith.muli %0, %c64_i32 : i32
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %5 = tt.splat %3 : i32 -> tensor<64xi32>
    %6 = arith.addi %5, %4 : tensor<64xi32>
    %7 = tt.splat %arg6 : i32 -> tensor<64xi32>
    %8 = arith.cmpi slt, %6, %7 : tensor<64xi32>
    %9 = arith.muli %2, %c32_i32 : i32
    %10 = arith.muli %1, %c32_i32 : i32
    %11 = arith.extsi %arg6 : i32 to i64
    %12 = tt.splat %arg5 : f32 -> tensor<64x64xf32>
    %13 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %14 = tt.expand_dims %6 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %15 = tt.broadcast %13 : tensor<64x1xi32> -> tensor<64x64xi32>
    %16 = tt.broadcast %14 : tensor<1x64xi32> -> tensor<64x64xi32>
    %17 = arith.cmpi sle, %15, %16 : tensor<64x64xi32>
    %18 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %19 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %20 = tt.broadcast %18 : tensor<64x1xi1> -> tensor<64x64xi1>
    %21 = tt.broadcast %19 : tensor<1x64xi1> -> tensor<64x64xi1>
    %22 = arith.andi %20, %21 : tensor<64x64xi1>
    %23 = arith.andi %17, %22 : tensor<64x64xi1>
    scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %24 = arith.addi %9, %arg7 : i32
      %25 = arith.muli %24, %c128_i32 : i32
      %26 = tt.addptr %arg0, %25 : !tt.ptr<bf16>, i32
      %27 = tt.addptr %arg1, %25 : !tt.ptr<bf16>, i32
      %28 = tt.addptr %arg3, %25 : !tt.ptr<bf16>, i32
      %29 = tt.addptr %arg4, %25 : !tt.ptr<bf16>, i32
      %30 = arith.addi %10, %arg7 : i32
      %31 = arith.muli %30, %arg6 : i32
      %32 = tt.addptr %arg2, %31 : !tt.ptr<bf16>, i32
      %33 = tt.splat %32 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
      %34 = tt.addptr %33, %6 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %35 = tt.load %34, %8, %cst : tensor<64x!tt.ptr<bf16>>
      %36 = arith.extf %35 : tensor<64xbf16> to tensor<64xf32>
      %37 = tt.make_tensor_ptr %27, [%11, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %38 = tt.make_tensor_ptr %26, [%c128_i64, %11], [%c1_i64, %c4096_i64], [%c0_i32, %3] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %39 = tt.load %37 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
      %40 = tt.load %38 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %41 = tt.dot %39, %40, %cst_1 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
      %42 = arith.mulf %41, %12 : tensor<64x64xf32>
      %43 = arith.addf %42, %cst_1 : tensor<64x64xf32>
      %44 = tt.expand_dims %36 {axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
      %45 = tt.expand_dims %36 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %46 = tt.broadcast %44 : tensor<1x64xf32> -> tensor<64x64xf32>
      %47 = tt.broadcast %45 : tensor<64x1xf32> -> tensor<64x64xf32>
      %48 = arith.subf %46, %47 : tensor<64x64xf32>
      %49 = math.exp %48 : tensor<64x64xf32>
      %50 = arith.mulf %43, %49 : tensor<64x64xf32>
      %51 = arith.select %23, %50, %cst_1 : tensor<64x64xi1>, tensor<64x64xf32>
      %52 = arith.truncf %51 : tensor<64x64xf32> to tensor<64x64xbf16>
      %53 = tt.make_tensor_ptr %28, [%11, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %54 = tt.make_tensor_ptr %29, [%11, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %55 = tt.load %53 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
      %56 = tt.dot %52, %55, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
      %57 = arith.truncf %56 : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %54, %57 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_bwd/norm_dot_false.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c65536_i32 = arith.constant 65536 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<true> : tensor<128x64xi1>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<64> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c4194304_i32 = arith.constant 4194304 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %3 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %4 = tt.broadcast %3 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:     %5 = tt.splat %arg9 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:     scf.for %arg10 = %0 to %c65536_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %6 = arith.divsi %arg10, %c64_i32 : i32
// CHECK-NEXT:       %7 = arith.muli %6, %c64_i32 : i32
// CHECK-NEXT:       %8 = arith.subi %arg10, %7 : i32
// CHECK-NEXT:       %9 = arith.muli %6, %c8192_i32 : i32
// CHECK-NEXT:       %10 = arith.extsi %9 : i32 to i64
// CHECK-NEXT:       %11 = arith.remsi %6, %c8_i32 : i32
// CHECK-NEXT:       %12 = arith.muli %11, %c524288_i32 : i32
// CHECK-NEXT:       %13 = arith.divsi %6, %c8_i32 : i32
// CHECK-NEXT:       %14 = arith.muli %13, %c4194304_i32 : i32
// CHECK-NEXT:       %15 = arith.addi %12, %14 : i32
// CHECK-NEXT:       %16 = arith.extsi %15 : i32 to i64
// CHECK-NEXT:       %17 = tt.addptr %arg0, %16 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %18 = tt.addptr %arg1, %16 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %19 = tt.addptr %arg2, %16 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %20 = tt.addptr %arg3, %16 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %21 = tt.addptr %arg4, %16 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %22 = tt.addptr %arg5, %16 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %23 = tt.addptr %arg6, %16 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %24 = tt.addptr %arg7, %10 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %25 = tt.addptr %arg8, %10 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %26 = arith.muli %8, %c128_i32 : i32
// CHECK-NEXT:       %27 = tt.splat %26 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %28 = arith.addi %27, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %29 = tt.expand_dims %28 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %30 = arith.muli %29, %cst_2 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %31 = tt.splat %18 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %32 = tt.addptr %31, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %33 = tt.broadcast %32 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %34 = tt.addptr %33, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %35 = tt.load %34 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %36 = tt.splat %19 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %37 = tt.addptr %36, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %38 = tt.broadcast %37 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %39 = tt.addptr %38, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %40 = tt.load %39 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %41 = tt.splat %17 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %42 = tt.splat %20 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %43 = tt.splat %24 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %44 = tt.splat %25 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %45 = tt.trans %35 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %46 = tt.trans %40 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %47 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %48:2 = scf.for %arg11 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg12 = %cst_0, %arg13 = %cst_0) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
// CHECK-NEXT:         %59 = arith.muli %arg11, %c128_i32 : i32
// CHECK-NEXT:         %60 = tt.splat %59 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %61 = arith.addi %60, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %62 = tt.expand_dims %61 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %63 = arith.muli %62, %cst_2 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %64 = tt.addptr %41, %63 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %65 = tt.broadcast %64 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %66 = tt.addptr %65, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %67 = tt.load %66 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %68 = tt.addptr %42, %63 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %69 = tt.broadcast %68 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %70 = tt.addptr %69, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %71 = tt.load %70 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %72 = tt.addptr %43, %61 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %73 = tt.load %72 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %74 = tt.addptr %44, %61 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %75 = tt.load %74 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %76 = tt.dot %67, %45, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %77 = arith.mulf %76, %5 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %78 = tt.expand_dims %73 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %79 = tt.broadcast %78 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %80 = arith.subf %77, %79 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %81 = math.exp %80 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %82 = arith.truncf %81 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %83 = tt.trans %82 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %84 = tt.dot %83, %71, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %85 = tt.dot %71, %46, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %86 = tt.expand_dims %75 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %87 = tt.broadcast %86 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %88 = arith.subf %85, %87 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %89 = arith.extf %82 {DataUse} : tensor<128x128xf16> to tensor<128x128xf32>
// CHECK-NEXT:         %90 = arith.mulf %89, %88 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %91 = arith.mulf %90, %5 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.truncf %91 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %93 = tt.trans %92 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %94 = tt.dot %93, %67, %arg12 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %95 = tt.dot %92, %35, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %96 = tt.addptr %47, %63 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %97 = tt.broadcast %96 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %98 = tt.addptr %97, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %99 = arith.truncf %95 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:         %100 = tt.atomic_rmw fadd, acq_rel, gpu, %98, %99, %cst_1 {Undefined} : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
// CHECK-NEXT:         scf.yield %94, %84 : tensor<128x64xf32>, tensor<128x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %49 = tt.splat %22 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %50 = tt.addptr %49, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %51 = tt.broadcast %50 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %52 = tt.addptr %51, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %53 = arith.truncf %48#0 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %52, %53 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %54 = tt.splat %23 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %55 = tt.addptr %54, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %56 = tt.broadcast %55 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %57 = tt.addptr %56, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %58 = arith.truncf %48#1 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %57, %58 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cst_1 = arith.constant dense<true> : tensor<128x64xi1>
    %cst_2 = arith.constant dense<64> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c4194304_i32 = arith.constant 4194304 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %4 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<128x64xi32>
    %5 = tt.splat %arg9 : f32 -> tensor<128x128xf32>
    scf.for %arg10 = %0 to %c65536_i32 step %c28_i32  : i32 {
      %6 = arith.divsi %arg10, %c64_i32 : i32
      %7 = arith.muli %6, %c64_i32 : i32
      %8 = arith.subi %arg10, %7 : i32
      %9 = arith.muli %6, %c8192_i32 : i32
      %10 = arith.extsi %9 : i32 to i64
      %11 = arith.remsi %6, %c8_i32 : i32
      %12 = arith.muli %11, %c524288_i32 : i32
      %13 = arith.divsi %6, %c8_i32 : i32
      %14 = arith.muli %13, %c4194304_i32 : i32
      %15 = arith.addi %12, %14 : i32
      %16 = arith.extsi %15 : i32 to i64
      %17 = tt.addptr %arg0, %16 : !tt.ptr<f16>, i64
      %18 = tt.addptr %arg1, %16 : !tt.ptr<f16>, i64
      %19 = tt.addptr %arg2, %16 : !tt.ptr<f16>, i64
      %20 = tt.addptr %arg3, %16 : !tt.ptr<f16>, i64
      %21 = tt.addptr %arg4, %16 : !tt.ptr<f16>, i64
      %22 = tt.addptr %arg5, %16 : !tt.ptr<f16>, i64
      %23 = tt.addptr %arg6, %16 : !tt.ptr<f16>, i64
      %24 = tt.addptr %arg7, %10 : !tt.ptr<f32>, i64
      %25 = tt.addptr %arg8, %10 : !tt.ptr<f32>, i64
      %26 = arith.muli %8, %c128_i32 : i32
      %27 = tt.splat %26 : i32 -> tensor<128xi32>
      %28 = arith.addi %27, %2 : tensor<128xi32>
      %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %30 = arith.muli %29, %cst_2 : tensor<128x1xi32>
      %31 = tt.splat %18 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %32 = tt.addptr %31, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %33 = tt.broadcast %32 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %34 = tt.addptr %33, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %35 = tt.load %34 : tensor<128x64x!tt.ptr<f16>>
      %36 = tt.splat %19 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %37 = tt.addptr %36, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %38 = tt.broadcast %37 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %39 = tt.addptr %38, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %40 = tt.load %39 : tensor<128x64x!tt.ptr<f16>>
      %41 = tt.splat %17 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.splat %20 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %43 = tt.splat %24 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %44 = tt.splat %25 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %45 = tt.trans %35 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %46 = tt.trans %40 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %47 = tt.splat %21 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %48:2 = scf.for %arg11 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg12 = %cst_0, %arg13 = %cst_0) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
        %59 = arith.muli %arg11, %c128_i32 : i32
        %60 = tt.splat %59 : i32 -> tensor<128xi32>
        %61 = arith.addi %60, %2 : tensor<128xi32>
        %62 = tt.expand_dims %61 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %63 = arith.muli %62, %cst_2 : tensor<128x1xi32>
        %64 = tt.addptr %41, %63 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %65 = tt.broadcast %64 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %66 = tt.addptr %65, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %67 = tt.load %66 : tensor<128x64x!tt.ptr<f16>>
        %68 = tt.addptr %42, %63 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %69 = tt.broadcast %68 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %70 = tt.addptr %69, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %71 = tt.load %70 : tensor<128x64x!tt.ptr<f16>>
        %72 = tt.addptr %43, %61 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %73 = tt.load %72 : tensor<128x!tt.ptr<f32>>
        %74 = tt.addptr %44, %61 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %75 = tt.load %74 : tensor<128x!tt.ptr<f32>>
        %76 = tt.dot %67, %45, %cst {triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %77 = arith.mulf %76, %5 : tensor<128x128xf32>
        %78 = tt.expand_dims %73 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %79 = tt.broadcast %78 : tensor<128x1xf32> -> tensor<128x128xf32>
        %80 = arith.subf %77, %79 : tensor<128x128xf32>
        %81 = math.exp %80 : tensor<128x128xf32>
        %82 = arith.truncf %81 : tensor<128x128xf32> to tensor<128x128xf16>
        %83 = tt.trans %82 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %84 = tt.dot %83, %71, %arg13 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %85 = tt.dot %71, %46, %cst {triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %86 = tt.expand_dims %75 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %87 = tt.broadcast %86 : tensor<128x1xf32> -> tensor<128x128xf32>
        %88 = arith.subf %85, %87 : tensor<128x128xf32>
        %89 = arith.extf %82 : tensor<128x128xf16> to tensor<128x128xf32>
        %90 = arith.mulf %89, %88 : tensor<128x128xf32>
        %91 = arith.mulf %90, %5 : tensor<128x128xf32>
        %92 = arith.truncf %91 : tensor<128x128xf32> to tensor<128x128xf16>
        %93 = tt.trans %92 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %94 = tt.dot %93, %67, %arg12 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %95 = tt.dot %92, %35, %cst_0 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %96 = tt.addptr %47, %63 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %97 = tt.broadcast %96 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %98 = tt.addptr %97, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %99 = arith.truncf %95 : tensor<128x64xf32> to tensor<128x64xf16>
        %100 = tt.atomic_rmw fadd, acq_rel, gpu, %98, %99, %cst_1 : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
        scf.yield %94, %84 : tensor<128x64xf32>, tensor<128x64xf32>
      }
      %49 = tt.splat %22 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %50 = tt.addptr %49, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %51 = tt.broadcast %50 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %52 = tt.addptr %51, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %53 = arith.truncf %48#0 : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %52, %53 : tensor<128x64x!tt.ptr<f16>>
      %54 = tt.splat %23 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %55 = tt.addptr %54, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %56 = tt.broadcast %55 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %57 = tt.addptr %56, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %58 = arith.truncf %48#1 : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %57, %58 : tensor<128x64x!tt.ptr<f16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_bwd/norm_dot_true.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %c8064_i32 = arith.constant 8064 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c65536_i32 = arith.constant 65536 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<true> : tensor<128x64xi1>
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<true> : tensor<64x64xi1>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<64> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_6 = arith.constant {MetaUse} dense<64> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c4194304_i32 = arith.constant 4194304 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %2 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %3 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %4 = tt.make_range {DataUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %5 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %6 = tt.broadcast %5 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:     %7 = tt.broadcast %5 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %8 = tt.splat %arg9 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %9 = tt.splat %arg9 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:     scf.for %arg10 = %0 to %c65536_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %10 = arith.divsi %arg10, %c64_i32 : i32
// CHECK-NEXT:       %11 = arith.muli %10, %c64_i32 : i32
// CHECK-NEXT:       %12 = arith.subi %arg10, %11 : i32
// CHECK-NEXT:       %13 = arith.muli %10, %c8192_i32 : i32
// CHECK-NEXT:       %14 = arith.extsi %13 : i32 to i64
// CHECK-NEXT:       %15 = arith.remsi %10, %c8_i32 : i32
// CHECK-NEXT:       %16 = arith.muli %15, %c524288_i32 : i32
// CHECK-NEXT:       %17 = arith.divsi %10, %c8_i32 : i32
// CHECK-NEXT:       %18 = arith.muli %17, %c4194304_i32 : i32
// CHECK-NEXT:       %19 = arith.addi %16, %18 : i32
// CHECK-NEXT:       %20 = arith.extsi %19 : i32 to i64
// CHECK-NEXT:       %21 = tt.addptr %arg0, %20 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %22 = tt.addptr %arg1, %20 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %23 = tt.addptr %arg2, %20 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %24 = tt.addptr %arg3, %20 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %25 = tt.addptr %arg4, %20 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %26 = tt.addptr %arg5, %20 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %27 = tt.addptr %arg6, %20 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %28 = tt.addptr %arg7, %14 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %29 = tt.addptr %arg8, %14 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %30 = arith.muli %12, %c128_i32 : i32
// CHECK-NEXT:       %31 = tt.splat %30 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %32 = tt.splat %30 {DataUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %33 = arith.addi %31, %3 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %34 = arith.addi %32, %4 {DataUse} : tensor<128xi32>
// CHECK-NEXT:       %35 = tt.expand_dims %33 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %36 = arith.muli %35, %cst_6 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %37 = tt.splat %22 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %38 = tt.addptr %37, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %39 = tt.broadcast %38 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %40 = tt.addptr %39, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %41 = tt.load %40 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %42 = tt.splat %23 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %43 = tt.addptr %42, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %44 = tt.broadcast %43 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %45 = tt.addptr %44, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %46 = tt.load %45 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %47 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %48 = tt.splat %24 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %49 = tt.splat %28 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %50 = tt.splat %29 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %51 = tt.trans %41 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %52 = tt.expand_dims %34 {DataUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:       %53 = tt.broadcast %52 {DataUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:       %54 = tt.trans %46 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %55 = tt.splat %25 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %56:2 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %cst_2, %arg13 = %cst_2) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
// CHECK-NEXT:         %78 = arith.muli %arg11, %c64_i32 : i32
// CHECK-NEXT:         %79 = arith.addi %30, %78 : i32
// CHECK-NEXT:         %80 = tt.splat %79 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %81 = tt.splat %79 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %82 = arith.addi %80, %1 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %83 = arith.addi %81, %2 {DataUse} : tensor<64xi32>
// CHECK-NEXT:         %84 = tt.expand_dims %82 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %85 = tt.expand_dims %83 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %86 = arith.muli %84, %cst_5 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %87 = tt.addptr %47, %86 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %88 = tt.broadcast %87 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %89 = tt.addptr %88, %7 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
// CHECK-NEXT:         %90 = tt.load %89 {DataUse} : tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %91 = tt.addptr %48, %86 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %92 = tt.broadcast %91 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %93 = tt.addptr %92, %7 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
// CHECK-NEXT:         %94 = tt.load %93 {DataUse} : tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %95 = tt.addptr %49, %82 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %96 = tt.load %95 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %97 = tt.addptr %50, %82 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %98 = tt.load %97 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %99 = tt.dot %90, %51, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %100 = arith.mulf %99, %8 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %101 = tt.expand_dims %96 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %102 = tt.broadcast %101 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %103 = arith.subf %100, %102 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %104 = math.exp %103 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %105 = tt.broadcast %85 {DataUse} : tensor<64x1xi32> -> tensor<64x128xi32>
// CHECK-NEXT:         %106 = arith.cmpi sge, %105, %53 {DataUse} : tensor<64x128xi32>
// CHECK-NEXT:         %107 = arith.select %106, %104, %cst_0 {DataUse} : tensor<64x128xi1>, tensor<64x128xf32>
// CHECK-NEXT:         %108 = arith.truncf %107 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
// CHECK-NEXT:         %109 = tt.trans %108 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
// CHECK-NEXT:         %110 = tt.dot %109, %94, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %111 = tt.dot %94, %54, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %112 = tt.expand_dims %98 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %113 = tt.broadcast %112 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %114 = arith.subf %111, %113 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %115 = arith.extf %108 {DataUse} : tensor<64x128xf16> to tensor<64x128xf32>
// CHECK-NEXT:         %116 = arith.mulf %115, %114 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %117 = arith.mulf %116, %8 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %118 = arith.truncf %117 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
// CHECK-NEXT:         %119 = tt.trans %118 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
// CHECK-NEXT:         %120 = tt.dot %119, %90, %arg12 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %121 = tt.dot %118, %41, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %122 = tt.addptr %55, %86 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %123 = tt.broadcast %122 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %124 = tt.addptr %123, %7 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
// CHECK-NEXT:         %125 = arith.truncf %121 {DataUse} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT:         %126 = tt.atomic_rmw fadd, acq_rel, gpu, %124, %125, %cst_4 {Undefined} : (tensor<64x64x!tt.ptr<f16>>, tensor<64x64xf16>, tensor<64x64xi1>) -> tensor<64x64xf16>
// CHECK-NEXT:         scf.yield %120, %110 : tensor<128x64xf32>, tensor<128x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %57 = arith.addi %30, %c128_i32 : i32
// CHECK-NEXT:       %58 = arith.subi %c8064_i32, %30 : i32
// CHECK-NEXT:       %59 = arith.divsi %58, %c128_i32 : i32
// CHECK-NEXT:       %60 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %61 = tt.splat %24 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %62 = tt.splat %28 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %63 = tt.splat %29 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %64 = tt.trans %41 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %65 = tt.trans %46 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %66 = tt.splat %25 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %67:2 = scf.for %arg11 = %c0_i32 to %59 step %c1_i32 iter_args(%arg12 = %56#0, %arg13 = %56#1) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
// CHECK-NEXT:         %78 = arith.muli %arg11, %c128_i32 : i32
// CHECK-NEXT:         %79 = arith.addi %57, %78 : i32
// CHECK-NEXT:         %80 = tt.splat %79 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %81 = arith.addi %80, %3 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %82 = tt.expand_dims %81 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %83 = arith.muli %82, %cst_6 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %84 = tt.addptr %60, %83 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %85 = tt.broadcast %84 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %86 = tt.addptr %85, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %87 = tt.load %86 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %88 = tt.addptr %61, %83 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %89 = tt.broadcast %88 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %90 = tt.addptr %89, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %91 = tt.load %90 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %92 = tt.addptr %62, %81 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %93 = tt.load %92 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %94 = tt.addptr %63, %81 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %95 = tt.load %94 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %96 = tt.dot %87, %64, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %97 = arith.mulf %96, %9 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %98 = tt.expand_dims %93 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %99 = tt.broadcast %98 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %100 = arith.subf %97, %99 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %101 = math.exp %100 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %102 = arith.truncf %101 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %103 = tt.trans %102 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %104 = tt.dot %103, %91, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %105 = tt.dot %91, %65, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %106 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %107 = tt.broadcast %106 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %108 = arith.subf %105, %107 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %109 = arith.extf %102 {DataUse} : tensor<128x128xf16> to tensor<128x128xf32>
// CHECK-NEXT:         %110 = arith.mulf %109, %108 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %111 = arith.mulf %110, %9 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %112 = arith.truncf %111 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %113 = tt.trans %112 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %114 = tt.dot %113, %87, %arg12 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %115 = tt.dot %112, %41, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %116 = tt.addptr %66, %83 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %117 = tt.broadcast %116 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %118 = tt.addptr %117, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %119 = arith.truncf %115 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:         %120 = tt.atomic_rmw fadd, acq_rel, gpu, %118, %119, %cst_3 {Undefined} : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
// CHECK-NEXT:         scf.yield %114, %104 : tensor<128x64xf32>, tensor<128x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %68 = tt.splat %26 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %69 = tt.addptr %68, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %70 = tt.broadcast %69 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %71 = tt.addptr %70, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %72 = arith.truncf %67#0 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %71, %72 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %73 = tt.splat %27 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %74 = tt.addptr %73, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %75 = tt.broadcast %74 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %76 = tt.addptr %75, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %77 = arith.truncf %67#1 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %76, %77 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %c8064_i32 = arith.constant 8064 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cst_3 = arith.constant dense<true> : tensor<128x64xi1>
    %cst_4 = arith.constant dense<true> : tensor<64x64xi1>
    %cst_5 = arith.constant dense<64> : tensor<64x1xi32>
    %cst_6 = arith.constant dense<64> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c4194304_i32 = arith.constant 4194304 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %4 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<128x64xi32>
    %5 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<64x64xi32>
    %6 = tt.splat %arg9 : f32 -> tensor<64x128xf32>
    %7 = tt.splat %arg9 : f32 -> tensor<128x128xf32>
    scf.for %arg10 = %0 to %c65536_i32 step %c28_i32  : i32 {
      %8 = arith.divsi %arg10, %c64_i32 : i32
      %9 = arith.muli %8, %c64_i32 : i32
      %10 = arith.subi %arg10, %9 : i32
      %11 = arith.muli %8, %c8192_i32 : i32
      %12 = arith.extsi %11 : i32 to i64
      %13 = arith.remsi %8, %c8_i32 : i32
      %14 = arith.muli %13, %c524288_i32 : i32
      %15 = arith.divsi %8, %c8_i32 : i32
      %16 = arith.muli %15, %c4194304_i32 : i32
      %17 = arith.addi %14, %16 : i32
      %18 = arith.extsi %17 : i32 to i64
      %19 = tt.addptr %arg0, %18 : !tt.ptr<f16>, i64
      %20 = tt.addptr %arg1, %18 : !tt.ptr<f16>, i64
      %21 = tt.addptr %arg2, %18 : !tt.ptr<f16>, i64
      %22 = tt.addptr %arg3, %18 : !tt.ptr<f16>, i64
      %23 = tt.addptr %arg4, %18 : !tt.ptr<f16>, i64
      %24 = tt.addptr %arg5, %18 : !tt.ptr<f16>, i64
      %25 = tt.addptr %arg6, %18 : !tt.ptr<f16>, i64
      %26 = tt.addptr %arg7, %12 : !tt.ptr<f32>, i64
      %27 = tt.addptr %arg8, %12 : !tt.ptr<f32>, i64
      %28 = arith.muli %10, %c128_i32 : i32
      %29 = tt.splat %28 : i32 -> tensor<128xi32>
      %30 = arith.addi %29, %2 : tensor<128xi32>
      %31 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %32 = arith.muli %31, %cst_6 : tensor<128x1xi32>
      %33 = tt.splat %20 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %34 = tt.addptr %33, %32 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %35 = tt.broadcast %34 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %36 = tt.addptr %35, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %37 = tt.load %36 : tensor<128x64x!tt.ptr<f16>>
      %38 = tt.splat %21 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %39 = tt.addptr %38, %32 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %40 = tt.broadcast %39 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %41 = tt.addptr %40, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %42 = tt.load %41 : tensor<128x64x!tt.ptr<f16>>
      %43 = tt.splat %19 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %44 = tt.splat %22 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %45 = tt.splat %26 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %46 = tt.splat %27 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %47 = tt.trans %37 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %48 = tt.expand_dims %30 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %49 = tt.broadcast %48 : tensor<1x128xi32> -> tensor<64x128xi32>
      %50 = tt.trans %42 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %51 = tt.splat %23 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %52:2 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %cst_2, %arg13 = %cst_2) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
        %74 = arith.muli %arg11, %c64_i32 : i32
        %75 = arith.addi %28, %74 : i32
        %76 = tt.splat %75 : i32 -> tensor<64xi32>
        %77 = arith.addi %76, %1 : tensor<64xi32>
        %78 = tt.expand_dims %77 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %79 = arith.muli %78, %cst_5 : tensor<64x1xi32>
        %80 = tt.addptr %43, %79 : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %81 = tt.broadcast %80 : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
        %82 = tt.addptr %81, %5 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        %83 = tt.load %82 : tensor<64x64x!tt.ptr<f16>>
        %84 = tt.addptr %44, %79 : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %85 = tt.broadcast %84 : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
        %86 = tt.addptr %85, %5 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        %87 = tt.load %86 : tensor<64x64x!tt.ptr<f16>>
        %88 = tt.addptr %45, %77 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %89 = tt.load %88 : tensor<64x!tt.ptr<f32>>
        %90 = tt.addptr %46, %77 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %91 = tt.load %90 : tensor<64x!tt.ptr<f32>>
        %92 = tt.dot %83, %47, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
        %93 = arith.mulf %92, %6 : tensor<64x128xf32>
        %94 = tt.expand_dims %89 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %95 = tt.broadcast %94 : tensor<64x1xf32> -> tensor<64x128xf32>
        %96 = arith.subf %93, %95 : tensor<64x128xf32>
        %97 = math.exp %96 : tensor<64x128xf32>
        %98 = tt.broadcast %78 : tensor<64x1xi32> -> tensor<64x128xi32>
        %99 = arith.cmpi sge, %98, %49 : tensor<64x128xi32>
        %100 = arith.select %99, %97, %cst_0 : tensor<64x128xi1>, tensor<64x128xf32>
        %101 = arith.truncf %100 : tensor<64x128xf32> to tensor<64x128xf16>
        %102 = tt.trans %101 {order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
        %103 = tt.dot %102, %87, %arg13 {triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
        %104 = tt.dot %87, %50, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
        %105 = tt.expand_dims %91 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %106 = tt.broadcast %105 : tensor<64x1xf32> -> tensor<64x128xf32>
        %107 = arith.subf %104, %106 : tensor<64x128xf32>
        %108 = arith.extf %101 : tensor<64x128xf16> to tensor<64x128xf32>
        %109 = arith.mulf %108, %107 : tensor<64x128xf32>
        %110 = arith.mulf %109, %6 : tensor<64x128xf32>
        %111 = arith.truncf %110 : tensor<64x128xf32> to tensor<64x128xf16>
        %112 = tt.trans %111 {order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
        %113 = tt.dot %112, %83, %arg12 {triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
        %114 = tt.dot %111, %37, %cst_1 {triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
        %115 = tt.addptr %51, %79 : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %116 = tt.broadcast %115 : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
        %117 = tt.addptr %116, %5 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        %118 = arith.truncf %114 : tensor<64x64xf32> to tensor<64x64xf16>
        %119 = tt.atomic_rmw fadd, acq_rel, gpu, %117, %118, %cst_4 : (tensor<64x64x!tt.ptr<f16>>, tensor<64x64xf16>, tensor<64x64xi1>) -> tensor<64x64xf16>
        scf.yield %113, %103 : tensor<128x64xf32>, tensor<128x64xf32>
      }
      %53 = arith.addi %28, %c128_i32 : i32
      %54 = arith.subi %c8064_i32, %28 : i32
      %55 = arith.divsi %54, %c128_i32 : i32
      %56 = tt.splat %19 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %57 = tt.splat %22 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %58 = tt.splat %26 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %59 = tt.splat %27 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %60 = tt.trans %37 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %61 = tt.trans %42 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %62 = tt.splat %23 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %63:2 = scf.for %arg11 = %c0_i32 to %55 step %c1_i32 iter_args(%arg12 = %52#0, %arg13 = %52#1) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
        %74 = arith.muli %arg11, %c128_i32 : i32
        %75 = arith.addi %53, %74 : i32
        %76 = tt.splat %75 : i32 -> tensor<128xi32>
        %77 = arith.addi %76, %2 : tensor<128xi32>
        %78 = tt.expand_dims %77 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %79 = arith.muli %78, %cst_6 : tensor<128x1xi32>
        %80 = tt.addptr %56, %79 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %81 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %82 = tt.addptr %81, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %83 = tt.load %82 : tensor<128x64x!tt.ptr<f16>>
        %84 = tt.addptr %57, %79 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %85 = tt.broadcast %84 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %86 = tt.addptr %85, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %87 = tt.load %86 : tensor<128x64x!tt.ptr<f16>>
        %88 = tt.addptr %58, %77 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %89 = tt.load %88 : tensor<128x!tt.ptr<f32>>
        %90 = tt.addptr %59, %77 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %91 = tt.load %90 : tensor<128x!tt.ptr<f32>>
        %92 = tt.dot %83, %60, %cst {triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %93 = arith.mulf %92, %7 : tensor<128x128xf32>
        %94 = tt.expand_dims %89 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %95 = tt.broadcast %94 : tensor<128x1xf32> -> tensor<128x128xf32>
        %96 = arith.subf %93, %95 : tensor<128x128xf32>
        %97 = math.exp %96 : tensor<128x128xf32>
        %98 = arith.truncf %97 : tensor<128x128xf32> to tensor<128x128xf16>
        %99 = tt.trans %98 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %100 = tt.dot %99, %87, %arg13 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %101 = tt.dot %87, %61, %cst {triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %102 = tt.expand_dims %91 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %103 = tt.broadcast %102 : tensor<128x1xf32> -> tensor<128x128xf32>
        %104 = arith.subf %101, %103 : tensor<128x128xf32>
        %105 = arith.extf %98 : tensor<128x128xf16> to tensor<128x128xf32>
        %106 = arith.mulf %105, %104 : tensor<128x128xf32>
        %107 = arith.mulf %106, %7 : tensor<128x128xf32>
        %108 = arith.truncf %107 : tensor<128x128xf32> to tensor<128x128xf16>
        %109 = tt.trans %108 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %110 = tt.dot %109, %83, %arg12 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %111 = tt.dot %108, %37, %cst_2 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %112 = tt.addptr %62, %79 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %113 = tt.broadcast %112 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %114 = tt.addptr %113, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %115 = arith.truncf %111 : tensor<128x64xf32> to tensor<128x64xf16>
        %116 = tt.atomic_rmw fadd, acq_rel, gpu, %114, %115, %cst_3 : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
        scf.yield %110, %100 : tensor<128x64xf32>, tensor<128x64xf32>
      }
      %64 = tt.splat %24 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %65 = tt.addptr %64, %32 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %66 = tt.broadcast %65 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %67 = tt.addptr %66, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %68 = arith.truncf %63#0 : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %67, %68 : tensor<128x64x!tt.ptr<f16>>
      %69 = tt.splat %25 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %70 = tt.addptr %69, %32 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %71 = tt.broadcast %70 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %72 = tt.addptr %71, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %73 = arith.truncf %63#1 : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %72, %73 : tensor<128x64x!tt.ptr<f16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_bwd_fp8/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<true> : tensor<32x128xi1>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant 9.99999997E-7 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1048576_i32 = arith.constant 1048576 : i32
// CHECK-NEXT:     %c131072_i32 = arith.constant 131072 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %cst_6 = arith.constant 4.480000e+02 : f32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %2 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %3 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %4 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %5 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %6 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     scf.for %arg13 = %0 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %7 = arith.divsi %arg13, %c8_i32 : i32
// CHECK-NEXT:       %8 = arith.muli %7, %c8_i32 : i32
// CHECK-NEXT:       %9 = arith.subi %arg13, %8 : i32
// CHECK-NEXT:       %10 = arith.muli %7, %c1024_i32 : i32
// CHECK-NEXT:       %11 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %12 = arith.remsi %7, %c8_i32 : i32
// CHECK-NEXT:       %13 = arith.muli %12, %c131072_i32 : i32
// CHECK-NEXT:       %14 = arith.divsi %7, %c8_i32 : i32
// CHECK-NEXT:       %15 = arith.muli %14, %c1048576_i32 : i32
// CHECK-NEXT:       %16 = arith.addi %13, %15 : i32
// CHECK-NEXT:       %17 = arith.extsi %16 : i32 to i64
// CHECK-NEXT:       %18 = tt.addptr %arg0, %17 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %19 = tt.addptr %arg1, %17 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %20 = tt.addptr %arg2, %17 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %21 = tt.addptr %arg3, %17 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %22 = tt.addptr %arg4, %17 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %23 = tt.addptr %arg5, %17 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %24 = tt.addptr %arg6, %17 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %25 = tt.addptr %arg7, %11 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %26 = tt.addptr %arg8, %11 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %27 = arith.divsi %10, %c128_i32 : i32
// CHECK-NEXT:       %28 = arith.extsi %27 : i32 to i64
// CHECK-NEXT:       %29 = tt.addptr %arg10, %28 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %30 = tt.addptr %arg11, %28 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %31 = arith.muli %9, %c128_i32 : i32
// CHECK-NEXT:       %32 = tt.splat %31 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %33 = arith.addi %32, %1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %34 = tt.expand_dims %33 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %35 = arith.muli %34, %cst_5 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %36 = tt.splat %19 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %37 = tt.addptr %36, %35 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
// CHECK-NEXT:       %38 = tt.broadcast %37 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %39 = tt.addptr %38, %3 {MetaUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
// CHECK-NEXT:       %40 = tt.load %39 {DataUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %41 = tt.splat %20 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %42 = tt.addptr %41, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %43 = tt.broadcast %42 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %44 = tt.addptr %43, %3 {MetaUse} : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
// CHECK-NEXT:       %45 = tt.load %44 {DataUse} : tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %46 = arith.divsi %31, %c128_i32 : i32
// CHECK-NEXT:       %47 = tt.addptr %30, %46 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %48 = tt.load %47 : !tt.ptr<f32>
// CHECK-NEXT:       %49 = tt.splat %18 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %50 = tt.trans %40 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:       %51 = tt.splat %48 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:       %52 = tt.splat %25 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:       %53 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
// CHECK-NEXT:       %54 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:       %55 = tt.splat %26 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:       %56 = tt.splat %22 {MetaUse} : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
// CHECK-NEXT:       %57:2 = scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg15 = %cst_0, %arg16 = %cst_0) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %66 = arith.muli %arg14, %c32_i32 : i32
// CHECK-NEXT:         %67 = tt.splat %66 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %68 = arith.addi %67, %4 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %69 = arith.divsi %66, %c128_i32 : i32
// CHECK-NEXT:         %70 = tt.addptr %29, %69 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %71 = tt.load %70 : !tt.ptr<f32>
// CHECK-NEXT:         %72 = tt.expand_dims %68 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %73 = arith.muli %72, %cst_4 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %74 = tt.addptr %49, %73 {MetaUse} : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
// CHECK-NEXT:         %75 = tt.broadcast %74 {MetaUse} : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %76 = tt.addptr %75, %5 {MetaUse} : tensor<32x128x!tt.ptr<f8E4M3FN>>, tensor<32x128xi32>
// CHECK-NEXT:         %77 = tt.load %76 {DataUse} : tensor<32x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %78 = tt.dot %77, %50, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
// CHECK-NEXT:         %79 = arith.cmpf une, %78, %78 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %80 = arith.select %79, %cst_1, %78 {DataUse} : tensor<32x128xi1>, tensor<32x128xf32>
// CHECK-NEXT:         %81 = tt.splat %71 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:         %82 = arith.mulf %80, %81 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %83 = arith.mulf %82, %51 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %84 = tt.addptr %52, %68 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %85 = tt.load %84 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %86 = arith.mulf %83, %6 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %87 = tt.expand_dims %85 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %88 = tt.broadcast %87 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %89 = arith.subf %86, %88 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %90 = math.exp %89 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %91 = tt.addptr %53, %73 {MetaUse} : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
// CHECK-NEXT:         %92 = tt.broadcast %91 {MetaUse} : tensor<32x1x!tt.ptr<f16>> -> tensor<32x128x!tt.ptr<f16>>
// CHECK-NEXT:         %93 = tt.addptr %92, %5 {MetaUse} : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
// CHECK-NEXT:         %94 = tt.load %93 {DataUse} : tensor<32x128x!tt.ptr<f16>>
// CHECK-NEXT:         %95 = arith.truncf %90 {DataUse} : tensor<32x128xf32> to tensor<32x128xf16>
// CHECK-NEXT:         %96 = tt.trans %95 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xf16> -> tensor<128x32xf16>
// CHECK-NEXT:         %97 = tt.dot %96, %94, %arg15 {DataUse, triton_cv12.normalized_dot} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %98 = tt.dot %94, %54, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xf16> * tensor<128x128xf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %99 = tt.addptr %55, %68 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %100 = tt.load %99 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %101 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %102 = tt.broadcast %101 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %103 = arith.subf %98, %102 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %104 = arith.mulf %90, %103 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %105 = arith.mulf %104, %6 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %106 = math.absf %105 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %107 = tt.reshape %106 allow_reorder {DataUse} : tensor<32x128xf32> -> tensor<4096xf32>
// CHECK-NEXT:         %108 = "tt.reduce"(%107) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg17: f32, %arg18: f32):
// CHECK-NEXT:           %128 = arith.maxnumf %arg17, %arg18 : f32
// CHECK-NEXT:           tt.reduce.return %128 : f32
// CHECK-NEXT:         }) : (tensor<4096xf32>) -> f32
// CHECK-NEXT:         %109 = arith.cmpf ogt, %108, %cst_2 : f32
// CHECK-NEXT:         %110 = scf.if %109 -> (f32) {
// CHECK-NEXT:           %128 = arith.divf %cst_6, %108 : f32
// CHECK-NEXT:           scf.yield %128 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %cst_3 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         %111 = tt.splat %110 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:         %112 = arith.mulf %105, %111 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %113 = tt.fp_to_fp %112 {DataUse}, rounding = rtne : tensor<32x128xf32> -> tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %114 = tt.dot %113, %40, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
// CHECK-NEXT:         %115 = arith.divf %48, %110 : f32
// CHECK-NEXT:         %116 = tt.splat %115 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:         %117 = arith.mulf %114, %116 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %118 = tt.trans %113 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN> -> tensor<128x32xf8E4M3FN>
// CHECK-NEXT:         %119 = tt.dot %118, %77, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<128x32xf8E4M3FN> * tensor<32x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %120 = arith.divf %71, %110 : f32
// CHECK-NEXT:         %121 = tt.splat %120 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:         %122 = arith.mulf %119, %121 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %123 = arith.addf %arg16, %122 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %124 = tt.addptr %56, %73 {MetaUse} : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
// CHECK-NEXT:         %125 = tt.broadcast %124 {MetaUse} : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
// CHECK-NEXT:         %126 = tt.addptr %125, %5 {MetaUse} : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
// CHECK-NEXT:         %127 = tt.atomic_rmw fadd, acq_rel, gpu, %126, %117, %cst {Undefined} : (tensor<32x128x!tt.ptr<f32>>, tensor<32x128xf32>, tensor<32x128xi1>) -> tensor<32x128xf32>
// CHECK-NEXT:         scf.yield %97, %123 : tensor<128x128xf32>, tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %58 = tt.splat %23 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %59 = tt.addptr %58, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %60 = tt.broadcast %59 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %61 = tt.addptr %60, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       tt.store %61, %57#1 : tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %62 = tt.splat %24 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %63 = tt.addptr %62, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %65 = tt.addptr %64, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       tt.store %65, %57#0 : tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst = arith.constant dense<true> : tensor<32x128xi1>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant dense<128> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %cst_5 = arith.constant dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_6 = arith.constant 4.480000e+02 : f32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %3 = tt.broadcast %2 : tensor<1x128xi32> -> tensor<128x128xi32>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %5 = tt.broadcast %2 : tensor<1x128xi32> -> tensor<32x128xi32>
    %6 = tt.splat %arg9 : f32 -> tensor<32x128xf32>
    scf.for %arg13 = %0 to %c8192_i32 step %c28_i32  : i32 {
      %7 = arith.divsi %arg13, %c8_i32 : i32
      %8 = arith.muli %7, %c8_i32 : i32
      %9 = arith.subi %arg13, %8 : i32
      %10 = arith.muli %7, %c1024_i32 : i32
      %11 = arith.extsi %10 : i32 to i64
      %12 = arith.remsi %7, %c8_i32 : i32
      %13 = arith.muli %12, %c131072_i32 : i32
      %14 = arith.divsi %7, %c8_i32 : i32
      %15 = arith.muli %14, %c1048576_i32 : i32
      %16 = arith.addi %13, %15 : i32
      %17 = arith.extsi %16 : i32 to i64
      %18 = tt.addptr %arg0, %17 : !tt.ptr<f8E4M3FN>, i64
      %19 = tt.addptr %arg1, %17 : !tt.ptr<f8E4M3FN>, i64
      %20 = tt.addptr %arg2, %17 : !tt.ptr<f16>, i64
      %21 = tt.addptr %arg3, %17 : !tt.ptr<f16>, i64
      %22 = tt.addptr %arg4, %17 : !tt.ptr<f32>, i64
      %23 = tt.addptr %arg5, %17 : !tt.ptr<f32>, i64
      %24 = tt.addptr %arg6, %17 : !tt.ptr<f32>, i64
      %25 = tt.addptr %arg7, %11 : !tt.ptr<f32>, i64
      %26 = tt.addptr %arg8, %11 : !tt.ptr<f32>, i64
      %27 = arith.divsi %10, %c128_i32 : i32
      %28 = arith.extsi %27 : i32 to i64
      %29 = tt.addptr %arg10, %28 : !tt.ptr<f32>, i64
      %30 = tt.addptr %arg11, %28 : !tt.ptr<f32>, i64
      %31 = arith.muli %9, %c128_i32 : i32
      %32 = tt.splat %31 : i32 -> tensor<128xi32>
      %33 = arith.addi %32, %1 : tensor<128xi32>
      %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %35 = arith.muli %34, %cst_5 : tensor<128x1xi32>
      %36 = tt.splat %19 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
      %37 = tt.addptr %36, %35 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
      %38 = tt.broadcast %37 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
      %39 = tt.addptr %38, %3 : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
      %40 = tt.load %39 : tensor<128x128x!tt.ptr<f8E4M3FN>>
      %41 = tt.splat %20 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.addptr %41, %35 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %43 = tt.broadcast %42 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
      %44 = tt.addptr %43, %3 : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
      %45 = tt.load %44 : tensor<128x128x!tt.ptr<f16>>
      %46 = arith.divsi %31, %c128_i32 : i32
      %47 = tt.addptr %30, %46 : !tt.ptr<f32>, i32
      %48 = tt.load %47 : !tt.ptr<f32>
      %49 = tt.splat %18 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
      %50 = tt.trans %40 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
      %51 = tt.splat %48 : f32 -> tensor<32x128xf32>
      %52 = tt.splat %25 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
      %53 = tt.splat %21 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
      %54 = tt.trans %45 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
      %55 = tt.splat %26 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
      %56 = tt.splat %22 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
      %57:2 = scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg15 = %cst_0, %arg16 = %cst_0) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
        %66 = arith.muli %arg14, %c32_i32 : i32
        %67 = tt.splat %66 : i32 -> tensor<32xi32>
        %68 = arith.addi %67, %4 : tensor<32xi32>
        %69 = arith.divsi %66, %c128_i32 : i32
        %70 = tt.addptr %29, %69 : !tt.ptr<f32>, i32
        %71 = tt.load %70 : !tt.ptr<f32>
        %72 = tt.expand_dims %68 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %73 = arith.muli %72, %cst_4 : tensor<32x1xi32>
        %74 = tt.addptr %49, %73 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
        %75 = tt.broadcast %74 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x128x!tt.ptr<f8E4M3FN>>
        %76 = tt.addptr %75, %5 : tensor<32x128x!tt.ptr<f8E4M3FN>>, tensor<32x128xi32>
        %77 = tt.load %76 : tensor<32x128x!tt.ptr<f8E4M3FN>>
        %78 = tt.dot %77, %50, %cst_1 {triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
        %79 = arith.cmpf une, %78, %78 : tensor<32x128xf32>
        %80 = arith.select %79, %cst_1, %78 : tensor<32x128xi1>, tensor<32x128xf32>
        %81 = tt.splat %71 : f32 -> tensor<32x128xf32>
        %82 = arith.mulf %80, %81 : tensor<32x128xf32>
        %83 = arith.mulf %82, %51 : tensor<32x128xf32>
        %84 = tt.addptr %52, %68 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %85 = tt.load %84 : tensor<32x!tt.ptr<f32>>
        %86 = arith.mulf %83, %6 : tensor<32x128xf32>
        %87 = tt.expand_dims %85 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %88 = tt.broadcast %87 : tensor<32x1xf32> -> tensor<32x128xf32>
        %89 = arith.subf %86, %88 : tensor<32x128xf32>
        %90 = math.exp %89 : tensor<32x128xf32>
        %91 = tt.addptr %53, %73 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
        %92 = tt.broadcast %91 : tensor<32x1x!tt.ptr<f16>> -> tensor<32x128x!tt.ptr<f16>>
        %93 = tt.addptr %92, %5 : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
        %94 = tt.load %93 : tensor<32x128x!tt.ptr<f16>>
        %95 = arith.truncf %90 : tensor<32x128xf32> to tensor<32x128xf16>
        %96 = tt.trans %95 {order = array<i32: 1, 0>} : tensor<32x128xf16> -> tensor<128x32xf16>
        %97 = tt.dot %96, %94, %arg15 {triton_cv12.normalized_dot} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
        %98 = tt.dot %94, %54, %cst_1 {triton_cv12.normalized_dot} : tensor<32x128xf16> * tensor<128x128xf16> -> tensor<32x128xf32>
        %99 = tt.addptr %55, %68 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %100 = tt.load %99 : tensor<32x!tt.ptr<f32>>
        %101 = tt.expand_dims %100 {axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %102 = tt.broadcast %101 : tensor<32x1xf32> -> tensor<32x128xf32>
        %103 = arith.subf %98, %102 : tensor<32x128xf32>
        %104 = arith.mulf %90, %103 : tensor<32x128xf32>
        %105 = arith.mulf %104, %6 : tensor<32x128xf32>
        %106 = math.absf %105 : tensor<32x128xf32>
        %107 = tt.reshape %106 allow_reorder : tensor<32x128xf32> -> tensor<4096xf32>
        %108 = "tt.reduce"(%107) <{axis = 0 : i32}> ({
        ^bb0(%arg17: f32, %arg18: f32):
          %128 = arith.maxnumf %arg17, %arg18 : f32
          tt.reduce.return %128 : f32
        }) : (tensor<4096xf32>) -> f32
        %109 = arith.cmpf ogt, %108, %cst_2 : f32
        %110 = scf.if %109 -> (f32) {
          %128 = arith.divf %cst_6, %108 : f32
          scf.yield %128 : f32
        } else {
          scf.yield %cst_3 : f32
        }
        %111 = tt.splat %110 : f32 -> tensor<32x128xf32>
        %112 = arith.mulf %105, %111 : tensor<32x128xf32>
        %113 = tt.fp_to_fp %112, rounding = rtne : tensor<32x128xf32> -> tensor<32x128xf8E4M3FN>
        %114 = tt.dot %113, %40, %cst_1 {triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
        %115 = arith.divf %48, %110 : f32
        %116 = tt.splat %115 : f32 -> tensor<32x128xf32>
        %117 = arith.mulf %114, %116 : tensor<32x128xf32>
        %118 = tt.trans %113 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN> -> tensor<128x32xf8E4M3FN>
        %119 = tt.dot %118, %77, %cst_0 {triton_cv12.normalized_dot} : tensor<128x32xf8E4M3FN> * tensor<32x128xf8E4M3FN> -> tensor<128x128xf32>
        %120 = arith.divf %71, %110 : f32
        %121 = tt.splat %120 : f32 -> tensor<128x128xf32>
        %122 = arith.mulf %119, %121 : tensor<128x128xf32>
        %123 = arith.addf %arg16, %122 : tensor<128x128xf32>
        %124 = tt.addptr %56, %73 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
        %125 = tt.broadcast %124 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
        %126 = tt.addptr %125, %5 : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
        %127 = tt.atomic_rmw fadd, acq_rel, gpu, %126, %117, %cst : (tensor<32x128x!tt.ptr<f32>>, tensor<32x128xf32>, tensor<32x128xi1>) -> tensor<32x128xf32>
        scf.yield %97, %123 : tensor<128x128xf32>, tensor<128x128xf32>
      }
      %58 = tt.splat %23 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %59 = tt.addptr %58, %35 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %60 = tt.broadcast %59 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %61 = tt.addptr %60, %3 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %61, %57#1 : tensor<128x128x!tt.ptr<f32>>
      %62 = tt.splat %24 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %63 = tt.addptr %62, %35 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %64 = tt.broadcast %63 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %65 = tt.addptr %64, %3 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %65, %57#0 : tensor<128x128x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_bwd_fp8_Q2/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<true> : tensor<64x128xi1>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant 9.99999997E-7 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1048576_i32 = arith.constant 1048576 : i32
// CHECK-NEXT:     %c131072_i32 = arith.constant 131072 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %cst_6 = arith.constant 4.480000e+02 : f32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %2 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %3 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %4 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %5 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %6 = tt.splat %arg9 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     scf.for %arg13 = %0 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %7 = arith.divsi %arg13, %c8_i32 : i32
// CHECK-NEXT:       %8 = arith.muli %7, %c8_i32 : i32
// CHECK-NEXT:       %9 = arith.subi %arg13, %8 : i32
// CHECK-NEXT:       %10 = arith.muli %7, %c1024_i32 : i32
// CHECK-NEXT:       %11 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %12 = arith.remsi %7, %c8_i32 : i32
// CHECK-NEXT:       %13 = arith.muli %12, %c131072_i32 : i32
// CHECK-NEXT:       %14 = arith.divsi %7, %c8_i32 : i32
// CHECK-NEXT:       %15 = arith.muli %14, %c1048576_i32 : i32
// CHECK-NEXT:       %16 = arith.addi %13, %15 : i32
// CHECK-NEXT:       %17 = arith.extsi %16 : i32 to i64
// CHECK-NEXT:       %18 = tt.addptr %arg0, %17 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %19 = tt.addptr %arg1, %17 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %20 = tt.addptr %arg2, %17 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %21 = tt.addptr %arg3, %17 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %22 = tt.addptr %arg4, %17 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %23 = tt.addptr %arg5, %17 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %24 = tt.addptr %arg6, %17 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %25 = tt.addptr %arg7, %11 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %26 = tt.addptr %arg8, %11 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %27 = arith.divsi %10, %c128_i32 : i32
// CHECK-NEXT:       %28 = arith.extsi %27 : i32 to i64
// CHECK-NEXT:       %29 = tt.addptr %arg10, %28 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %30 = tt.addptr %arg11, %28 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %31 = arith.muli %9, %c128_i32 : i32
// CHECK-NEXT:       %32 = tt.splat %31 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %33 = arith.addi %32, %1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %34 = tt.expand_dims %33 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %35 = arith.muli %34, %cst_5 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %36 = tt.splat %19 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %37 = tt.addptr %36, %35 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
// CHECK-NEXT:       %38 = tt.broadcast %37 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %39 = tt.addptr %38, %3 {MetaUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
// CHECK-NEXT:       %40 = tt.load %39 {DataUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %41 = tt.splat %20 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %42 = tt.addptr %41, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %43 = tt.broadcast %42 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %44 = tt.addptr %43, %3 {MetaUse} : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
// CHECK-NEXT:       %45 = tt.load %44 {DataUse} : tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %46 = arith.divsi %31, %c128_i32 : i32
// CHECK-NEXT:       %47 = tt.addptr %30, %46 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %48 = tt.load %47 : !tt.ptr<f32>
// CHECK-NEXT:       %49 = tt.splat %18 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %50 = tt.trans %40 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:       %51 = tt.splat %48 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:       %52 = tt.splat %25 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %53 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %54 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:       %55 = tt.splat %26 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %56 = tt.splat %22 {MetaUse} : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
// CHECK-NEXT:       %57:2 = scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg15 = %cst_0, %arg16 = %cst_0) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %66 = arith.muli %arg14, %c64_i32 : i32
// CHECK-NEXT:         %67 = tt.splat %66 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %68 = arith.addi %67, %4 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %69 = arith.divsi %66, %c128_i32 : i32
// CHECK-NEXT:         %70 = tt.addptr %29, %69 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %71 = tt.load %70 : !tt.ptr<f32>
// CHECK-NEXT:         %72 = tt.expand_dims %68 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %73 = arith.muli %72, %cst_4 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %74 = tt.addptr %49, %73 {MetaUse} : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
// CHECK-NEXT:         %75 = tt.broadcast %74 {MetaUse} : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %76 = tt.addptr %75, %5 {MetaUse} : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
// CHECK-NEXT:         %77 = tt.load %76 {DataUse} : tensor<64x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %78 = tt.dot %77, %50, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
// CHECK-NEXT:         %79 = arith.cmpf une, %78, %78 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %80 = arith.select %79, %cst_1, %78 {DataUse} : tensor<64x128xi1>, tensor<64x128xf32>
// CHECK-NEXT:         %81 = tt.splat %71 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:         %82 = arith.mulf %80, %81 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %83 = arith.mulf %82, %51 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %84 = tt.addptr %52, %68 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %85 = tt.load %84 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %86 = arith.mulf %83, %6 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %87 = tt.expand_dims %85 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %88 = tt.broadcast %87 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %89 = arith.subf %86, %88 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %90 = math.exp %89 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %91 = tt.addptr %53, %73 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %92 = tt.broadcast %91 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
// CHECK-NEXT:         %93 = tt.addptr %92, %5 {MetaUse} : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
// CHECK-NEXT:         %94 = tt.load %93 {DataUse} : tensor<64x128x!tt.ptr<f16>>
// CHECK-NEXT:         %95 = arith.truncf %90 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
// CHECK-NEXT:         %96 = tt.trans %95 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
// CHECK-NEXT:         %97 = tt.dot %96, %94, %arg15 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %98 = tt.dot %94, %54, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x128xf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %99 = tt.addptr %55, %68 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %100 = tt.load %99 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %101 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %102 = tt.broadcast %101 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %103 = arith.subf %98, %102 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %104 = arith.mulf %90, %103 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %105 = arith.mulf %104, %6 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %106 = math.absf %105 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %107 = tt.reshape %106 allow_reorder {DataUse} : tensor<64x128xf32> -> tensor<8192xf32>
// CHECK-NEXT:         %108 = "tt.reduce"(%107) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg17: f32, %arg18: f32):
// CHECK-NEXT:           %128 = arith.maxnumf %arg17, %arg18 : f32
// CHECK-NEXT:           tt.reduce.return %128 : f32
// CHECK-NEXT:         }) : (tensor<8192xf32>) -> f32
// CHECK-NEXT:         %109 = arith.cmpf ogt, %108, %cst_2 : f32
// CHECK-NEXT:         %110 = scf.if %109 -> (f32) {
// CHECK-NEXT:           %128 = arith.divf %cst_6, %108 : f32
// CHECK-NEXT:           scf.yield %128 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %cst_3 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         %111 = tt.splat %110 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:         %112 = arith.mulf %105, %111 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %113 = tt.fp_to_fp %112 {DataUse}, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %114 = tt.dot %113, %40, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
// CHECK-NEXT:         %115 = arith.divf %48, %110 : f32
// CHECK-NEXT:         %116 = tt.splat %115 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:         %117 = arith.mulf %114, %116 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %118 = tt.trans %113 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
// CHECK-NEXT:         %119 = tt.dot %118, %77, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf8E4M3FN> * tensor<64x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %120 = arith.divf %71, %110 : f32
// CHECK-NEXT:         %121 = tt.splat %120 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:         %122 = arith.mulf %119, %121 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %123 = arith.addf %arg16, %122 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %124 = tt.addptr %56, %73 {MetaUse} : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
// CHECK-NEXT:         %125 = tt.broadcast %124 {MetaUse} : tensor<64x1x!tt.ptr<f32>> -> tensor<64x128x!tt.ptr<f32>>
// CHECK-NEXT:         %126 = tt.addptr %125, %5 {MetaUse} : tensor<64x128x!tt.ptr<f32>>, tensor<64x128xi32>
// CHECK-NEXT:         %127 = tt.atomic_rmw fadd, acq_rel, gpu, %126, %117, %cst {Undefined} : (tensor<64x128x!tt.ptr<f32>>, tensor<64x128xf32>, tensor<64x128xi1>) -> tensor<64x128xf32>
// CHECK-NEXT:         scf.yield %97, %123 : tensor<128x128xf32>, tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %58 = tt.splat %23 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %59 = tt.addptr %58, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %60 = tt.broadcast %59 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %61 = tt.addptr %60, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       tt.store %61, %57#1 : tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %62 = tt.splat %24 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %63 = tt.addptr %62, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %65 = tt.addptr %64, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       tt.store %65, %57#0 : tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst = arith.constant dense<true> : tensor<64x128xi1>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant dense<128> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_5 = arith.constant dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_6 = arith.constant 4.480000e+02 : f32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %3 = tt.broadcast %2 : tensor<1x128xi32> -> tensor<128x128xi32>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %5 = tt.broadcast %2 : tensor<1x128xi32> -> tensor<64x128xi32>
    %6 = tt.splat %arg9 : f32 -> tensor<64x128xf32>
    scf.for %arg13 = %0 to %c8192_i32 step %c28_i32  : i32 {
      %7 = arith.divsi %arg13, %c8_i32 : i32
      %8 = arith.muli %7, %c8_i32 : i32
      %9 = arith.subi %arg13, %8 : i32
      %10 = arith.muli %7, %c1024_i32 : i32
      %11 = arith.extsi %10 : i32 to i64
      %12 = arith.remsi %7, %c8_i32 : i32
      %13 = arith.muli %12, %c131072_i32 : i32
      %14 = arith.divsi %7, %c8_i32 : i32
      %15 = arith.muli %14, %c1048576_i32 : i32
      %16 = arith.addi %13, %15 : i32
      %17 = arith.extsi %16 : i32 to i64
      %18 = tt.addptr %arg0, %17 : !tt.ptr<f8E4M3FN>, i64
      %19 = tt.addptr %arg1, %17 : !tt.ptr<f8E4M3FN>, i64
      %20 = tt.addptr %arg2, %17 : !tt.ptr<f16>, i64
      %21 = tt.addptr %arg3, %17 : !tt.ptr<f16>, i64
      %22 = tt.addptr %arg4, %17 : !tt.ptr<f32>, i64
      %23 = tt.addptr %arg5, %17 : !tt.ptr<f32>, i64
      %24 = tt.addptr %arg6, %17 : !tt.ptr<f32>, i64
      %25 = tt.addptr %arg7, %11 : !tt.ptr<f32>, i64
      %26 = tt.addptr %arg8, %11 : !tt.ptr<f32>, i64
      %27 = arith.divsi %10, %c128_i32 : i32
      %28 = arith.extsi %27 : i32 to i64
      %29 = tt.addptr %arg10, %28 : !tt.ptr<f32>, i64
      %30 = tt.addptr %arg11, %28 : !tt.ptr<f32>, i64
      %31 = arith.muli %9, %c128_i32 : i32
      %32 = tt.splat %31 : i32 -> tensor<128xi32>
      %33 = arith.addi %32, %1 : tensor<128xi32>
      %34 = tt.expand_dims %33 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %35 = arith.muli %34, %cst_5 : tensor<128x1xi32>
      %36 = tt.splat %19 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
      %37 = tt.addptr %36, %35 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
      %38 = tt.broadcast %37 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
      %39 = tt.addptr %38, %3 : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
      %40 = tt.load %39 : tensor<128x128x!tt.ptr<f8E4M3FN>>
      %41 = tt.splat %20 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.addptr %41, %35 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %43 = tt.broadcast %42 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
      %44 = tt.addptr %43, %3 : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
      %45 = tt.load %44 : tensor<128x128x!tt.ptr<f16>>
      %46 = arith.divsi %31, %c128_i32 : i32
      %47 = tt.addptr %30, %46 : !tt.ptr<f32>, i32
      %48 = tt.load %47 : !tt.ptr<f32>
      %49 = tt.splat %18 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
      %50 = tt.trans %40 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
      %51 = tt.splat %48 : f32 -> tensor<64x128xf32>
      %52 = tt.splat %25 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %53 = tt.splat %21 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %54 = tt.trans %45 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
      %55 = tt.splat %26 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %56 = tt.splat %22 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
      %57:2 = scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg15 = %cst_0, %arg16 = %cst_0) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
        %66 = arith.muli %arg14, %c64_i32 : i32
        %67 = tt.splat %66 : i32 -> tensor<64xi32>
        %68 = arith.addi %67, %4 : tensor<64xi32>
        %69 = arith.divsi %66, %c128_i32 : i32
        %70 = tt.addptr %29, %69 : !tt.ptr<f32>, i32
        %71 = tt.load %70 : !tt.ptr<f32>
        %72 = tt.expand_dims %68 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %73 = arith.muli %72, %cst_4 : tensor<64x1xi32>
        %74 = tt.addptr %49, %73 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
        %75 = tt.broadcast %74 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
        %76 = tt.addptr %75, %5 : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
        %77 = tt.load %76 : tensor<64x128x!tt.ptr<f8E4M3FN>>
        %78 = tt.dot %77, %50, %cst_1 {triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
        %79 = arith.cmpf une, %78, %78 : tensor<64x128xf32>
        %80 = arith.select %79, %cst_1, %78 : tensor<64x128xi1>, tensor<64x128xf32>
        %81 = tt.splat %71 : f32 -> tensor<64x128xf32>
        %82 = arith.mulf %80, %81 : tensor<64x128xf32>
        %83 = arith.mulf %82, %51 : tensor<64x128xf32>
        %84 = tt.addptr %52, %68 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %85 = tt.load %84 : tensor<64x!tt.ptr<f32>>
        %86 = arith.mulf %83, %6 : tensor<64x128xf32>
        %87 = tt.expand_dims %85 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %88 = tt.broadcast %87 : tensor<64x1xf32> -> tensor<64x128xf32>
        %89 = arith.subf %86, %88 : tensor<64x128xf32>
        %90 = math.exp %89 : tensor<64x128xf32>
        %91 = tt.addptr %53, %73 : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %92 = tt.broadcast %91 : tensor<64x1x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
        %93 = tt.addptr %92, %5 : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
        %94 = tt.load %93 : tensor<64x128x!tt.ptr<f16>>
        %95 = arith.truncf %90 : tensor<64x128xf32> to tensor<64x128xf16>
        %96 = tt.trans %95 {order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
        %97 = tt.dot %96, %94, %arg15 {triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %98 = tt.dot %94, %54, %cst_1 {triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x128xf16> -> tensor<64x128xf32>
        %99 = tt.addptr %55, %68 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %100 = tt.load %99 : tensor<64x!tt.ptr<f32>>
        %101 = tt.expand_dims %100 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %102 = tt.broadcast %101 : tensor<64x1xf32> -> tensor<64x128xf32>
        %103 = arith.subf %98, %102 : tensor<64x128xf32>
        %104 = arith.mulf %90, %103 : tensor<64x128xf32>
        %105 = arith.mulf %104, %6 : tensor<64x128xf32>
        %106 = math.absf %105 : tensor<64x128xf32>
        %107 = tt.reshape %106 allow_reorder : tensor<64x128xf32> -> tensor<8192xf32>
        %108 = "tt.reduce"(%107) <{axis = 0 : i32}> ({
        ^bb0(%arg17: f32, %arg18: f32):
          %128 = arith.maxnumf %arg17, %arg18 : f32
          tt.reduce.return %128 : f32
        }) : (tensor<8192xf32>) -> f32
        %109 = arith.cmpf ogt, %108, %cst_2 : f32
        %110 = scf.if %109 -> (f32) {
          %128 = arith.divf %cst_6, %108 : f32
          scf.yield %128 : f32
        } else {
          scf.yield %cst_3 : f32
        }
        %111 = tt.splat %110 : f32 -> tensor<64x128xf32>
        %112 = arith.mulf %105, %111 : tensor<64x128xf32>
        %113 = tt.fp_to_fp %112, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
        %114 = tt.dot %113, %40, %cst_1 {triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
        %115 = arith.divf %48, %110 : f32
        %116 = tt.splat %115 : f32 -> tensor<64x128xf32>
        %117 = arith.mulf %114, %116 : tensor<64x128xf32>
        %118 = tt.trans %113 {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
        %119 = tt.dot %118, %77, %cst_0 {triton_cv12.normalized_dot} : tensor<128x64xf8E4M3FN> * tensor<64x128xf8E4M3FN> -> tensor<128x128xf32>
        %120 = arith.divf %71, %110 : f32
        %121 = tt.splat %120 : f32 -> tensor<128x128xf32>
        %122 = arith.mulf %119, %121 : tensor<128x128xf32>
        %123 = arith.addf %arg16, %122 : tensor<128x128xf32>
        %124 = tt.addptr %56, %73 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
        %125 = tt.broadcast %124 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x128x!tt.ptr<f32>>
        %126 = tt.addptr %125, %5 : tensor<64x128x!tt.ptr<f32>>, tensor<64x128xi32>
        %127 = tt.atomic_rmw fadd, acq_rel, gpu, %126, %117, %cst : (tensor<64x128x!tt.ptr<f32>>, tensor<64x128xf32>, tensor<64x128xi1>) -> tensor<64x128xf32>
        scf.yield %97, %123 : tensor<128x128xf32>, tensor<128x128xf32>
      }
      %58 = tt.splat %23 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %59 = tt.addptr %58, %35 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %60 = tt.broadcast %59 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %61 = tt.addptr %60, %3 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %61, %57#1 : tensor<128x128x!tt.ptr<f32>>
      %62 = tt.splat %24 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %63 = tt.addptr %62, %35 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %64 = tt.broadcast %63 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %65 = tt.addptr %64, %3 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %65, %57#0 : tensor<128x128x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_fwd/norm_dot_false.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<5.000000e-01> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c65536_i32 = arith.constant 65536 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c8192_i64 = arith.constant 8192 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c8388608_i64 = arith.constant 8388608 : i64
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg8 = %0 to %c65536_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %2 = arith.divsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %3 = arith.remsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %4 = arith.divsi %2, %c8_i32 : i32
// CHECK-NEXT:       %5 = arith.remsi %2, %c8_i32 : i32
// CHECK-NEXT:       %6 = arith.muli %5, %arg7 : i32
// CHECK-NEXT:       %7 = arith.divsi %6, %c8_i32 : i32
// CHECK-NEXT:       %8 = arith.extsi %4 : i32 to i64
// CHECK-NEXT:       %9 = arith.muli %8, %c8388608_i64 : i64
// CHECK-NEXT:       %10 = arith.extsi %5 : i32 to i64
// CHECK-NEXT:       %11 = arith.muli %10, %c1048576_i64 : i64
// CHECK-NEXT:       %12 = arith.addi %9, %11 : i64
// CHECK-NEXT:       %13 = arith.extsi %7 : i32 to i64
// CHECK-NEXT:       %14 = arith.muli %13, %c1048576_i64 : i64
// CHECK-NEXT:       %15 = arith.addi %9, %14 : i64
// CHECK-NEXT:       %16 = tt.addptr %arg0, %12 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %17 = arith.muli %3, %c128_i32 : i32
// CHECK-NEXT:       %18 = tt.make_tensor_ptr %16, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
// CHECK-NEXT:       %19 = tt.addptr %arg2, %15 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %20 = tt.make_tensor_ptr %19, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
// CHECK-NEXT:       %21 = tt.addptr %arg1, %15 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %22 = tt.make_tensor_ptr %21, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
// CHECK-NEXT:       %23 = tt.addptr %arg5, %12 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %23, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
// CHECK-NEXT:       %25 = tt.splat %17 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %26 = arith.addi %25, %1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %27 = tt.load %18 {DataUse} : !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:       %28:5 = scf.for %arg9 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %cst_2, %arg12 = %cst_0, %arg13 = %20, %arg14 = %22) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>)  : i32 {
// CHECK-NEXT:         %39 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:         %40 = tt.trans %39 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %41 = tt.dot %27, %40, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %42 = arith.mulf %41, %cst_1 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %43 = "tt.reduce"(%42) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %63 = arith.maximumf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %63 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %44 = arith.maximumf %arg12, %43 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %45 = tt.expand_dims %44 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %46 = tt.broadcast %45 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %47 = arith.subf %42, %46 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %48 = math.exp %47 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %49 = arith.truncf %48 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %50 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:         %51 = "tt.reduce"(%48) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %63 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %63 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %52 = arith.subf %arg12, %44 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %53 = math.exp %52 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %54 = arith.mulf %arg10, %53 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %55 = arith.addf %54, %51 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %56 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %57 = tt.broadcast %56 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %58 = arith.mulf %arg11, %57 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %59 = tt.dot %49, %50, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %60 = arith.addf %59, %58 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         %61 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
// CHECK-NEXT:         %62 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
// CHECK-NEXT:         scf.yield %55, %60, %44, %61, %62 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %29 = math.log %28#0 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %30 = arith.addf %28#2, %29 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %31 = tt.expand_dims %28#0 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %32 = tt.broadcast %31 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %33 = arith.divf %28#1, %32 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %34 = arith.muli %2, %c8192_i32 : i32
// CHECK-NEXT:       %35 = tt.addptr %arg4, %34 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %36 = tt.splat %35 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %37 = tt.addptr %36, %26 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       tt.store %37, %30 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %38 = arith.truncf %33 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:       tt.store %24, %38 : !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant dense<5.000000e-01> : tensor<128x128xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c28_i32 = arith.constant 28 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    scf.for %arg8 = %0 to %c65536_i32 step %c28_i32  : i32 {
      %2 = arith.divsi %arg8, %c64_i32 : i32
      %3 = arith.remsi %arg8, %c64_i32 : i32
      %4 = arith.divsi %2, %c8_i32 : i32
      %5 = arith.remsi %2, %c8_i32 : i32
      %6 = arith.muli %5, %arg7 : i32
      %7 = arith.divsi %6, %c8_i32 : i32
      %8 = arith.extsi %4 : i32 to i64
      %9 = arith.muli %8, %c8388608_i64 : i64
      %10 = arith.extsi %5 : i32 to i64
      %11 = arith.muli %10, %c1048576_i64 : i64
      %12 = arith.addi %9, %11 : i64
      %13 = arith.extsi %7 : i32 to i64
      %14 = arith.muli %13, %c1048576_i64 : i64
      %15 = arith.addi %9, %14 : i64
      %16 = tt.addptr %arg0, %12 : !tt.ptr<f16>, i64
      %17 = arith.muli %3, %c128_i32 : i32
      %18 = tt.make_tensor_ptr %16, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
      %19 = tt.addptr %arg2, %15 : !tt.ptr<f16>, i64
      %20 = tt.make_tensor_ptr %19, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
      %21 = tt.addptr %arg1, %15 : !tt.ptr<f16>, i64
      %22 = tt.make_tensor_ptr %21, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
      %23 = tt.addptr %arg5, %12 : !tt.ptr<f16>, i64
      %24 = tt.make_tensor_ptr %23, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
      %25 = tt.splat %17 : i32 -> tensor<128xi32>
      %26 = arith.addi %25, %1 : tensor<128xi32>
      %27 = tt.load %18 : !tt.ptr<tensor<128x128xf16>>
      %28:5 = scf.for %arg9 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %cst_2, %arg12 = %cst_0, %arg13 = %20, %arg14 = %22) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>)  : i32 {
        %39 = tt.load %arg14 : !tt.ptr<tensor<128x128xf16>>
        %40 = tt.trans %39 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %41 = tt.dot %27, %40, %cst_2 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
        %42 = arith.mulf %41, %cst_1 : tensor<128x128xf32>
        %43 = "tt.reduce"(%42) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %63 = arith.maximumf %arg15, %arg16 : f32
          tt.reduce.return %63 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %44 = arith.maximumf %arg12, %43 : tensor<128xf32>
        %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %46 = tt.broadcast %45 : tensor<128x1xf32> -> tensor<128x128xf32>
        %47 = arith.subf %42, %46 : tensor<128x128xf32>
        %48 = math.exp %47 : tensor<128x128xf32>
        %49 = arith.truncf %48 : tensor<128x128xf32> to tensor<128x128xf16>
        %50 = tt.load %arg13 : !tt.ptr<tensor<128x128xf16>>
        %51 = "tt.reduce"(%48) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %63 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %63 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %52 = arith.subf %arg12, %44 : tensor<128xf32>
        %53 = math.exp %52 : tensor<128xf32>
        %54 = arith.mulf %arg10, %53 : tensor<128xf32>
        %55 = arith.addf %54, %51 : tensor<128xf32>
        %56 = tt.expand_dims %53 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %57 = tt.broadcast %56 : tensor<128x1xf32> -> tensor<128x128xf32>
        %58 = arith.mulf %arg11, %57 : tensor<128x128xf32>
        %59 = tt.dot %49, %50, %cst_2 {triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
        %60 = arith.addf %59, %58 {triton_cv12.add_from_dot} : tensor<128x128xf32>
        %61 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
        %62 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
        scf.yield %55, %60, %44, %61, %62 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %29 = math.log %28#0 : tensor<128xf32>
      %30 = arith.addf %28#2, %29 : tensor<128xf32>
      %31 = tt.expand_dims %28#0 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %32 = tt.broadcast %31 : tensor<128x1xf32> -> tensor<128x128xf32>
      %33 = arith.divf %28#1, %32 : tensor<128x128xf32>
      %34 = arith.muli %2, %c8192_i32 : i32
      %35 = tt.addptr %arg4, %34 : !tt.ptr<f32>, i32
      %36 = tt.splat %35 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %37 = tt.addptr %36, %26 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %37, %30 : tensor<128x!tt.ptr<f32>>
      %38 = arith.truncf %33 : tensor<128x128xf32> to tensor<128x128xf16>
      tt.store %24, %38 : !tt.ptr<tensor<128x128xf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_fwd/norm_dot_true.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<5.000000e-01> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c8192_i64 = arith.constant 8192 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c8388608_i64 = arith.constant 8388608 : i64
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.addptr %arg6, %0 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %2 = tt.load %1 : !tt.ptr<i32>
// CHECK-NEXT:     %3 = tt.addptr %1, %c1_i32 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %4 = tt.load %3 : !tt.ptr<i32>
// CHECK-NEXT:     %5 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg8 = %2 to %4 step %c1_i32  : i32 {
// CHECK-NEXT:       %6 = arith.divsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %7 = arith.remsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %8 = arith.divsi %6, %c8_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %6, %c8_i32 : i32
// CHECK-NEXT:       %10 = arith.muli %9, %arg7 : i32
// CHECK-NEXT:       %11 = arith.divsi %10, %c8_i32 : i32
// CHECK-NEXT:       %12 = arith.extsi %8 : i32 to i64
// CHECK-NEXT:       %13 = arith.muli %12, %c8388608_i64 : i64
// CHECK-NEXT:       %14 = arith.extsi %9 : i32 to i64
// CHECK-NEXT:       %15 = arith.muli %14, %c1048576_i64 : i64
// CHECK-NEXT:       %16 = arith.addi %13, %15 : i64
// CHECK-NEXT:       %17 = arith.extsi %11 : i32 to i64
// CHECK-NEXT:       %18 = arith.muli %17, %c1048576_i64 : i64
// CHECK-NEXT:       %19 = arith.addi %13, %18 : i64
// CHECK-NEXT:       %20 = tt.addptr %arg0, %16 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %21 = arith.muli %7, %c128_i32 : i32
// CHECK-NEXT:       %22 = tt.make_tensor_ptr %20, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:       %23 = tt.addptr %arg2, %19 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %23, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:       %25 = tt.addptr %arg1, %19 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %26 = tt.make_tensor_ptr %25, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:       %27 = tt.addptr %arg5, %16 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %28 = tt.make_tensor_ptr %27, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:       %29 = tt.splat %21 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %30 = arith.addi %29, %5 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %31 = tt.load %22 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:       %32:5 = scf.for %arg9 = %c0_i32 to %21 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %cst_3, %arg12 = %cst_0, %arg13 = %24, %arg14 = %26) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
// CHECK-NEXT:         %50 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %51 = tt.trans %50 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %52 = tt.dot %31, %51, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %53 = arith.mulf %52, %cst_2 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %54 = "tt.reduce"(%53) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %74 = arith.maximumf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %74 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %55 = arith.maximumf %arg12, %54 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %56 = tt.expand_dims %55 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %57 = tt.broadcast %56 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %58 = arith.subf %53, %57 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %59 = math.exp %58 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %60 = arith.truncf %59 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %61 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %62 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %74 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %74 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %63 = arith.subf %arg12, %55 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %64 = math.exp %63 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %65 = arith.mulf %arg10, %64 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %66 = arith.addf %65, %62 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %67 = tt.expand_dims %64 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %68 = tt.broadcast %67 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %69 = arith.mulf %arg11, %68 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %70 = tt.dot %60, %61, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %71 = arith.addf %70, %69 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         %72 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         %73 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         scf.yield %66, %71, %55, %72, %73 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %33 = arith.muli %7, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
// CHECK-NEXT:       %34 = arith.addi %7, %c1_i32 : i32
// CHECK-NEXT:       %35 = arith.muli %34, %c128_i32 : i32
// CHECK-NEXT:       %36 = tt.make_tensor_ptr %arg3, [%c8192_i64, %c8192_i64], [%c8192_i64, %c1_i64], [%21, %33] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
// CHECK-NEXT:       %37 = tt.advance %26, [%33, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:       %38 = tt.advance %24, [%33, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:       %39:6 = scf.for %arg9 = %33 to %35 step %c128_i32 iter_args(%arg10 = %36, %arg11 = %32#0, %arg12 = %32#1, %arg13 = %32#2, %arg14 = %38, %arg15 = %37) -> (!tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
// CHECK-NEXT:         %50 = tt.load %arg15 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %51 = tt.trans %50 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %52 = tt.dot %31, %51, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %53 = tt.load %arg10 {DataUse} : !tt.ptr<tensor<128x128xf32>>
// CHECK-NEXT:         %54 = arith.mulf %52, %cst_2 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %55 = arith.cmpf une, %53, %cst_3 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %56 = arith.select %55, %cst_1, %cst_3 {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
// CHECK-NEXT:         %57 = arith.addf %54, %56 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %58 = "tt.reduce"(%57) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %79 = arith.maximumf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %79 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %59 = arith.maximumf %arg13, %58 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %60 = tt.expand_dims %59 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %61 = tt.broadcast %60 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %62 = arith.subf %57, %61 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %63 = tt.advance %arg10, [%c0_i32, %c128_i32] : <tensor<128x128xf32>>
// CHECK-NEXT:         %64 = math.exp %62 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %65 = arith.truncf %64 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %66 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %67 = "tt.reduce"(%64) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %79 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %79 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %68 = arith.subf %arg13, %59 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %69 = math.exp %68 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %70 = arith.mulf %arg11, %69 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %71 = arith.addf %70, %67 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %72 = tt.expand_dims %69 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %74 = arith.mulf %arg12, %73 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %75 = tt.dot %65, %66, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %76 = arith.addf %75, %74 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         %77 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         %78 = tt.advance %arg15, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         scf.yield %63, %71, %76, %59, %77, %78 : !tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %40 = math.log %39#1 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %41 = arith.addf %39#3, %40 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %42 = tt.expand_dims %39#1 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %43 = tt.broadcast %42 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %44 = arith.divf %39#2, %43 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %45 = arith.muli %6, %c8192_i32 : i32
// CHECK-NEXT:       %46 = tt.addptr %arg4, %45 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %47 = tt.splat %46 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %48 = tt.addptr %47, %30 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       tt.store %48, %41 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %49 = arith.truncf %44 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       tt.store %28, %49 : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant dense<-1.000000e+04> : tensor<128x128xf32>
    %cst_2 = arith.constant dense<5.000000e-01> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg6, %0 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = tt.addptr %1, %c1_i32 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    scf.for %arg8 = %2 to %4 step %c1_i32  : i32 {
      %6 = arith.divsi %arg8, %c64_i32 : i32
      %7 = arith.remsi %arg8, %c64_i32 : i32
      %8 = arith.divsi %6, %c8_i32 : i32
      %9 = arith.remsi %6, %c8_i32 : i32
      %10 = arith.muli %9, %arg7 : i32
      %11 = arith.divsi %10, %c8_i32 : i32
      %12 = arith.extsi %8 : i32 to i64
      %13 = arith.muli %12, %c8388608_i64 : i64
      %14 = arith.extsi %9 : i32 to i64
      %15 = arith.muli %14, %c1048576_i64 : i64
      %16 = arith.addi %13, %15 : i64
      %17 = arith.extsi %11 : i32 to i64
      %18 = arith.muli %17, %c1048576_i64 : i64
      %19 = arith.addi %13, %18 : i64
      %20 = tt.addptr %arg0, %16 : !tt.ptr<bf16>, i64
      %21 = arith.muli %7, %c128_i32 : i32
      %22 = tt.make_tensor_ptr %20, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
      %23 = tt.addptr %arg2, %19 : !tt.ptr<bf16>, i64
      %24 = tt.make_tensor_ptr %23, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
      %25 = tt.addptr %arg1, %19 : !tt.ptr<bf16>, i64
      %26 = tt.make_tensor_ptr %25, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
      %27 = tt.addptr %arg5, %16 : !tt.ptr<bf16>, i64
      %28 = tt.make_tensor_ptr %27, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
      %29 = tt.splat %21 : i32 -> tensor<128xi32>
      %30 = arith.addi %29, %5 : tensor<128xi32>
      %31 = tt.load %22 : !tt.ptr<tensor<128x128xbf16>>
      %32:5 = scf.for %arg9 = %c0_i32 to %21 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %cst_3, %arg12 = %cst_0, %arg13 = %24, %arg14 = %26) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
        %50 = tt.load %arg14 : !tt.ptr<tensor<128x128xbf16>>
        %51 = tt.trans %50 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %52 = tt.dot %31, %51, %cst_3 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %53 = arith.mulf %52, %cst_2 : tensor<128x128xf32>
        %54 = "tt.reduce"(%53) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %74 = arith.maximumf %arg15, %arg16 : f32
          tt.reduce.return %74 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %55 = arith.maximumf %arg12, %54 : tensor<128xf32>
        %56 = tt.expand_dims %55 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %57 = tt.broadcast %56 : tensor<128x1xf32> -> tensor<128x128xf32>
        %58 = arith.subf %53, %57 : tensor<128x128xf32>
        %59 = math.exp %58 : tensor<128x128xf32>
        %60 = arith.truncf %59 : tensor<128x128xf32> to tensor<128x128xbf16>
        %61 = tt.load %arg13 : !tt.ptr<tensor<128x128xbf16>>
        %62 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %74 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %74 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %63 = arith.subf %arg12, %55 : tensor<128xf32>
        %64 = math.exp %63 : tensor<128xf32>
        %65 = arith.mulf %arg10, %64 : tensor<128xf32>
        %66 = arith.addf %65, %62 : tensor<128xf32>
        %67 = tt.expand_dims %64 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %68 = tt.broadcast %67 : tensor<128x1xf32> -> tensor<128x128xf32>
        %69 = arith.mulf %arg11, %68 : tensor<128x128xf32>
        %70 = tt.dot %60, %61, %cst_3 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %71 = arith.addf %70, %69 {triton_cv12.add_from_dot} : tensor<128x128xf32>
        %72 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        %73 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        scf.yield %66, %71, %55, %72, %73 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %33 = arith.muli %7, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %34 = arith.addi %7, %c1_i32 : i32
      %35 = arith.muli %34, %c128_i32 : i32
      %36 = tt.make_tensor_ptr %arg3, [%c8192_i64, %c8192_i64], [%c8192_i64, %c1_i64], [%21, %33] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
      %37 = tt.advance %26, [%33, %c0_i32] : <tensor<128x128xbf16>>
      %38 = tt.advance %24, [%33, %c0_i32] : <tensor<128x128xbf16>>
      %39:6 = scf.for %arg9 = %33 to %35 step %c128_i32 iter_args(%arg10 = %36, %arg11 = %32#0, %arg12 = %32#1, %arg13 = %32#2, %arg14 = %38, %arg15 = %37) -> (!tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
        %50 = tt.load %arg15 : !tt.ptr<tensor<128x128xbf16>>
        %51 = tt.trans %50 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %52 = tt.dot %31, %51, %cst_3 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %53 = tt.load %arg10 : !tt.ptr<tensor<128x128xf32>>
        %54 = arith.mulf %52, %cst_2 : tensor<128x128xf32>
        %55 = arith.cmpf une, %53, %cst_3 : tensor<128x128xf32>
        %56 = arith.select %55, %cst_1, %cst_3 : tensor<128x128xi1>, tensor<128x128xf32>
        %57 = arith.addf %54, %56 : tensor<128x128xf32>
        %58 = "tt.reduce"(%57) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %79 = arith.maximumf %arg16, %arg17 : f32
          tt.reduce.return %79 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %59 = arith.maximumf %arg13, %58 : tensor<128xf32>
        %60 = tt.expand_dims %59 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %61 = tt.broadcast %60 : tensor<128x1xf32> -> tensor<128x128xf32>
        %62 = arith.subf %57, %61 : tensor<128x128xf32>
        %63 = tt.advance %arg10, [%c0_i32, %c128_i32] : <tensor<128x128xf32>>
        %64 = math.exp %62 : tensor<128x128xf32>
        %65 = arith.truncf %64 : tensor<128x128xf32> to tensor<128x128xbf16>
        %66 = tt.load %arg14 : !tt.ptr<tensor<128x128xbf16>>
        %67 = "tt.reduce"(%64) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %79 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %79 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %68 = arith.subf %arg13, %59 : tensor<128xf32>
        %69 = math.exp %68 : tensor<128xf32>
        %70 = arith.mulf %arg11, %69 : tensor<128xf32>
        %71 = arith.addf %70, %67 : tensor<128xf32>
        %72 = tt.expand_dims %69 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.mulf %arg12, %73 : tensor<128x128xf32>
        %75 = tt.dot %65, %66, %cst_3 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %76 = arith.addf %75, %74 {triton_cv12.add_from_dot} : tensor<128x128xf32>
        %77 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        %78 = tt.advance %arg15, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        scf.yield %63, %71, %76, %59, %77, %78 : !tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %40 = math.log %39#1 : tensor<128xf32>
      %41 = arith.addf %39#3, %40 : tensor<128xf32>
      %42 = tt.expand_dims %39#1 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %43 = tt.broadcast %42 : tensor<128x1xf32> -> tensor<128x128xf32>
      %44 = arith.divf %39#2, %43 : tensor<128x128xf32>
      %45 = arith.muli %6, %c8192_i32 : i32
      %46 = tt.addptr %arg4, %45 : !tt.ptr<f32>, i32
      %47 = tt.splat %46 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %48 = tt.addptr %47, %30 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %48, %41 : tensor<128x!tt.ptr<f32>>
      %49 = arith.truncf %44 : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %28, %49 : !tt.ptr<tensor<128x128xbf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_fwd_fp8/norm_dot_false.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg9 = %0 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %2 = arith.divsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %3 = arith.remsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %4 = arith.divsi %2, %c8_i32 : i32
// CHECK-NEXT:       %5 = arith.remsi %2, %c8_i32 : i32
// CHECK-NEXT:       %6 = arith.extsi %4 : i32 to i64
// CHECK-NEXT:       %7 = arith.muli %6, %c1048576_i64 : i64
// CHECK-NEXT:       %8 = arith.extsi %5 : i32 to i64
// CHECK-NEXT:       %9 = arith.muli %8, %c131072_i64 : i64
// CHECK-NEXT:       %10 = arith.addi %7, %9 : i64
// CHECK-NEXT:       %11 = arith.muli %6, %c64_i64 : i64
// CHECK-NEXT:       %12 = arith.muli %8, %c8_i64 : i64
// CHECK-NEXT:       %13 = arith.addi %11, %12 : i64
// CHECK-NEXT:       %14 = tt.addptr %arg0, %10 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %15 = arith.muli %3, %c128_i32 : i32
// CHECK-NEXT:       %16 = tt.make_tensor_ptr %14, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %17 = tt.addptr %arg2, %10 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %18 = tt.make_tensor_ptr %17, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %19 = tt.addptr %arg1, %10 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %20 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %21 = tt.addptr %arg6, %13 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %22 = arith.divsi %15, %c128_i32 : i32
// CHECK-NEXT:       %23 = tt.make_tensor_ptr %21, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %24 = tt.addptr %arg7, %13 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %25 = tt.make_tensor_ptr %24, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %26 = tt.addptr %arg8, %13 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %27 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %28 = tt.addptr %arg4, %10 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %29 = tt.make_tensor_ptr %28, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
// CHECK-NEXT:       %30 = tt.splat %15 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %31 = arith.addi %30, %1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %32 = tt.load %16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %33 = tt.load %23 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %34 = tt.broadcast %33 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %35:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %25, %arg12 = %27, %arg13 = %cst, %arg14 = %cst_2, %arg15 = %cst_0, %arg16 = %18, %arg17 = %20) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %45 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %46 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %47 = tt.dot %32, %46, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %48 = tt.load %arg11 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %49 = arith.mulf %47, %34 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %50 = tt.broadcast %48 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %51 = arith.mulf %49, %50 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %52 = arith.mulf %51, %cst_1 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %78 = arith.maximumf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %78 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %54 = arith.maximumf %arg15, %53 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %55 = tt.expand_dims %54 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %56 = tt.broadcast %55 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %57 = arith.subf %52, %56 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %58 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %59 = math.exp %57 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %60 = tt.fp_to_fp %59 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %61 = tt.load %arg16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %62 = tt.dot %60, %61, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %63 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %64 = tt.broadcast %63 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %65 = arith.mulf %62, %64 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %66 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %67 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %78 = arith.addf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %78 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %68 = arith.subf %arg15, %54 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %69 = math.exp %68 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %70 = arith.mulf %arg13, %69 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %71 = arith.addf %70, %67 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %72 = tt.expand_dims %69 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %74 = arith.mulf %arg14, %73 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %75 = arith.addf %74, %65 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %76 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %77 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %58, %66, %71, %75, %54, %76, %77 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %36 = math.log %35#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %37 = arith.addf %35#4, %36 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %38 = tt.expand_dims %35#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %39 = tt.broadcast %38 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %40 = arith.divf %35#3, %39 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %41 = arith.muli %2, %c1024_i32 : i32
// CHECK-NEXT:       %42 = tt.addptr %arg3, %41 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %43 = tt.splat %42 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %44 = tt.addptr %43, %31 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       tt.store %44, %37 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %29, %40 : !tt.ptr<tensor<128x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c28_i32 = arith.constant 28 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i64 = arith.constant 8 : i64
    %c64_i64 = arith.constant 64 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    scf.for %arg9 = %0 to %c8192_i32 step %c28_i32  : i32 {
      %2 = arith.divsi %arg9, %c8_i32 : i32
      %3 = arith.remsi %arg9, %c8_i32 : i32
      %4 = arith.divsi %2, %c8_i32 : i32
      %5 = arith.remsi %2, %c8_i32 : i32
      %6 = arith.extsi %4 : i32 to i64
      %7 = arith.muli %6, %c1048576_i64 : i64
      %8 = arith.extsi %5 : i32 to i64
      %9 = arith.muli %8, %c131072_i64 : i64
      %10 = arith.addi %7, %9 : i64
      %11 = arith.muli %6, %c64_i64 : i64
      %12 = arith.muli %8, %c8_i64 : i64
      %13 = arith.addi %11, %12 : i64
      %14 = tt.addptr %arg0, %10 : !tt.ptr<f8E4M3FN>, i64
      %15 = arith.muli %3, %c128_i32 : i32
      %16 = tt.make_tensor_ptr %14, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %17 = tt.addptr %arg2, %10 : !tt.ptr<f8E4M3FN>, i64
      %18 = tt.make_tensor_ptr %17, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %19 = tt.addptr %arg1, %10 : !tt.ptr<f8E4M3FN>, i64
      %20 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %21 = tt.addptr %arg6, %13 : !tt.ptr<f32>, i64
      %22 = arith.divsi %15, %c128_i32 : i32
      %23 = tt.make_tensor_ptr %21, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %24 = tt.addptr %arg7, %13 : !tt.ptr<f32>, i64
      %25 = tt.make_tensor_ptr %24, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %26 = tt.addptr %arg8, %13 : !tt.ptr<f32>, i64
      %27 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %28 = tt.addptr %arg4, %10 : !tt.ptr<f32>, i64
      %29 = tt.make_tensor_ptr %28, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
      %30 = tt.splat %15 : i32 -> tensor<128xi32>
      %31 = arith.addi %30, %1 : tensor<128xi32>
      %32 = tt.load %16 : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %33 = tt.load %23 : !tt.ptr<tensor<1x1xf32>>
      %34 = tt.broadcast %33 : tensor<1x1xf32> -> tensor<128x128xf32>
      %35:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %25, %arg12 = %27, %arg13 = %cst, %arg14 = %cst_2, %arg15 = %cst_0, %arg16 = %18, %arg17 = %20) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %45 = tt.load %arg17 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %46 = tt.trans %45 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %47 = tt.dot %32, %46, %cst_2 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %48 = tt.load %arg11 : !tt.ptr<tensor<1x1xf32>>
        %49 = arith.mulf %47, %34 : tensor<128x128xf32>
        %50 = tt.broadcast %48 : tensor<1x1xf32> -> tensor<128x128xf32>
        %51 = arith.mulf %49, %50 : tensor<128x128xf32>
        %52 = arith.mulf %51, %cst_1 : tensor<128x128xf32>
        %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.maximumf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %54 = arith.maximumf %arg15, %53 : tensor<128xf32>
        %55 = tt.expand_dims %54 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %56 = tt.broadcast %55 : tensor<128x1xf32> -> tensor<128x128xf32>
        %57 = arith.subf %52, %56 : tensor<128x128xf32>
        %58 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %59 = math.exp %57 : tensor<128x128xf32>
        %60 = tt.fp_to_fp %59, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %61 = tt.load %arg16 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %62 = tt.dot %60, %61, %cst_2 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %63 = tt.load %arg12 : !tt.ptr<tensor<1x1xf32>>
        %64 = tt.broadcast %63 : tensor<1x1xf32> -> tensor<128x128xf32>
        %65 = arith.mulf %62, %64 : tensor<128x128xf32>
        %66 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %67 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.addf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %68 = arith.subf %arg15, %54 : tensor<128xf32>
        %69 = math.exp %68 : tensor<128xf32>
        %70 = arith.mulf %arg13, %69 : tensor<128xf32>
        %71 = arith.addf %70, %67 : tensor<128xf32>
        %72 = tt.expand_dims %69 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.mulf %arg14, %73 : tensor<128x128xf32>
        %75 = arith.addf %74, %65 : tensor<128x128xf32>
        %76 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %77 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %58, %66, %71, %75, %54, %76, %77 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %36 = math.log %35#2 : tensor<128xf32>
      %37 = arith.addf %35#4, %36 : tensor<128xf32>
      %38 = tt.expand_dims %35#2 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %39 = tt.broadcast %38 : tensor<128x1xf32> -> tensor<128x128xf32>
      %40 = arith.divf %35#3, %39 : tensor<128x128xf32>
      %41 = arith.muli %2, %c1024_i32 : i32
      %42 = tt.addptr %arg3, %41 : !tt.ptr<f32>, i32
      %43 = tt.splat %42 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %44 = tt.addptr %43, %31 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %44, %37 : tensor<128x!tt.ptr<f32>>
      tt.store %29, %40 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_fwd_fp8/norm_dot_true.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.addptr %arg6, %0 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %2 = tt.load %1 : !tt.ptr<i32>
// CHECK-NEXT:     %3 = tt.addptr %1, %c1_i32 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %4 = tt.load %3 : !tt.ptr<i32>
// CHECK-NEXT:     %5 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %6 = tt.bitcast %arg3 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK-NEXT:     scf.for %arg10 = %2 to %4 step %c1_i32  : i32 {
// CHECK-NEXT:       %7 = arith.divsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %8 = arith.remsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %9 = arith.divsi %7, %c8_i32 : i32
// CHECK-NEXT:       %10 = arith.remsi %7, %c8_i32 : i32
// CHECK-NEXT:       %11 = arith.extsi %9 : i32 to i64
// CHECK-NEXT:       %12 = arith.muli %11, %c1048576_i64 : i64
// CHECK-NEXT:       %13 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %14 = arith.muli %13, %c131072_i64 : i64
// CHECK-NEXT:       %15 = arith.addi %12, %14 : i64
// CHECK-NEXT:       %16 = arith.muli %11, %c64_i64 : i64
// CHECK-NEXT:       %17 = arith.muli %13, %c8_i64 : i64
// CHECK-NEXT:       %18 = arith.addi %16, %17 : i64
// CHECK-NEXT:       %19 = tt.addptr %arg0, %15 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %20 = arith.muli %8, %c128_i32 : i32
// CHECK-NEXT:       %21 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %22 = tt.addptr %arg2, %15 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %23 = tt.make_tensor_ptr %22, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %24 = tt.addptr %arg1, %15 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %25 = tt.make_tensor_ptr %24, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %26 = tt.addptr %arg7, %18 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %27 = arith.divsi %20, %c128_i32 : i32
// CHECK-NEXT:       %28 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %29 = tt.addptr %arg8, %18 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %30 = tt.make_tensor_ptr %29, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %31 = tt.addptr %arg9, %18 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %32 = tt.make_tensor_ptr %31, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %33 = tt.addptr %arg5, %15 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %34 = tt.make_tensor_ptr %33, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
// CHECK-NEXT:       %35 = tt.splat %20 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %36 = arith.addi %35, %5 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %37 = tt.load %21 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %38 = tt.load %28 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %39 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %40:7 = scf.for %arg11 = %c0_i32 to %20 step %c128_i32 iter_args(%arg12 = %30, %arg13 = %32, %arg14 = %cst, %arg15 = %cst_4, %arg16 = %cst_0, %arg17 = %23, %arg18 = %25) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %58 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %61 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %62 = arith.mulf %60, %39 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %65 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %91 = arith.maximumf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %91 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %67 = arith.maximumf %arg16, %66 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %68 = tt.expand_dims %67 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %69 = tt.broadcast %68 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %70 = arith.subf %65, %69 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %72 = math.exp %70 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %73 = tt.fp_to_fp %72 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %74 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %75 = tt.dot %73, %74, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %76 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %77 = tt.broadcast %76 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %78 = arith.mulf %75, %77 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %79 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %80 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %91 = arith.addf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %91 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %81 = arith.subf %arg16, %67 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %82 = math.exp %81 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %83 = arith.mulf %arg14, %82 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %84 = arith.addf %83, %80 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %85 = tt.expand_dims %82 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %86 = tt.broadcast %85 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %87 = arith.mulf %arg15, %86 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %88 = arith.addf %87, %78 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %89 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %90 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %71, %79, %84, %88, %67, %89, %90 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %41 = arith.muli %8, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
// CHECK-NEXT:       %42 = arith.addi %8, %c1_i32 : i32
// CHECK-NEXT:       %43 = arith.muli %42, %c128_i32 : i32
// CHECK-NEXT:       %44 = tt.make_tensor_ptr %6, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%20, %41] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
// CHECK-NEXT:       %45 = tt.advance %25, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %46 = tt.advance %23, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %47 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %48:8 = scf.for %arg11 = %41 to %43 step %c128_i32 iter_args(%arg12 = %44, %arg13 = %30, %arg14 = %32, %arg15 = %40#2, %arg16 = %40#3, %arg17 = %40#4, %arg18 = %46, %arg19 = %45) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %58 = tt.load %arg19 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %61 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %62 = arith.mulf %60, %47 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %65 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<128x128xi8>>
// CHECK-NEXT:         %66 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %67 = arith.cmpi ne, %65, %cst_1 {DataUse} : tensor<128x128xi8>
// CHECK-NEXT:         %68 = arith.select %67, %cst_2, %cst_4 {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
// CHECK-NEXT:         %69 = arith.addf %66, %68 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %70 = "tt.reduce"(%69) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %96 = arith.maximumf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %96 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %71 = arith.maximumf %arg17, %70 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %72 = tt.expand_dims %71 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %74 = arith.subf %69, %73 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %75 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
// CHECK-NEXT:         %76 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %77 = math.exp %74 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %78 = tt.fp_to_fp %77 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %79 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %80 = tt.dot %78, %79, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %81 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %82 = tt.broadcast %81 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %83 = arith.mulf %80, %82 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %84 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %85 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %96 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %96 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %86 = arith.subf %arg17, %71 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %87 = math.exp %86 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %88 = arith.mulf %arg15, %87 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %89 = arith.addf %88, %85 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %90 = tt.expand_dims %87 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.mulf %arg16, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = arith.addf %92, %83 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %94 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %95 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %75, %76, %84, %89, %93, %71, %94, %95 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %49 = math.log %48#3 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %50 = arith.addf %48#5, %49 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %51 = tt.expand_dims %48#3 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %52 = tt.broadcast %51 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %53 = arith.divf %48#4, %52 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %54 = arith.muli %7, %c1024_i32 : i32
// CHECK-NEXT:       %55 = tt.addptr %arg4, %54 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %56 = tt.splat %55 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %57 = tt.addptr %56, %36 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       tt.store %57, %50 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %34, %53 : !tt.ptr<tensor<128x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant dense<0> : tensor<128x128xi8>
    %cst_2 = arith.constant dense<-1.000000e+04> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i64 = arith.constant 8 : i64
    %c64_i64 = arith.constant 64 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg6, %0 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = tt.addptr %1, %c1_i32 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %6 = tt.bitcast %arg3 : !tt.ptr<i1> -> !tt.ptr<i8>
    scf.for %arg10 = %2 to %4 step %c1_i32  : i32 {
      %7 = arith.divsi %arg10, %c8_i32 : i32
      %8 = arith.remsi %arg10, %c8_i32 : i32
      %9 = arith.divsi %7, %c8_i32 : i32
      %10 = arith.remsi %7, %c8_i32 : i32
      %11 = arith.extsi %9 : i32 to i64
      %12 = arith.muli %11, %c1048576_i64 : i64
      %13 = arith.extsi %10 : i32 to i64
      %14 = arith.muli %13, %c131072_i64 : i64
      %15 = arith.addi %12, %14 : i64
      %16 = arith.muli %11, %c64_i64 : i64
      %17 = arith.muli %13, %c8_i64 : i64
      %18 = arith.addi %16, %17 : i64
      %19 = tt.addptr %arg0, %15 : !tt.ptr<f8E4M3FN>, i64
      %20 = arith.muli %8, %c128_i32 : i32
      %21 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %22 = tt.addptr %arg2, %15 : !tt.ptr<f8E4M3FN>, i64
      %23 = tt.make_tensor_ptr %22, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %24 = tt.addptr %arg1, %15 : !tt.ptr<f8E4M3FN>, i64
      %25 = tt.make_tensor_ptr %24, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %26 = tt.addptr %arg7, %18 : !tt.ptr<f32>, i64
      %27 = arith.divsi %20, %c128_i32 : i32
      %28 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %29 = tt.addptr %arg8, %18 : !tt.ptr<f32>, i64
      %30 = tt.make_tensor_ptr %29, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %31 = tt.addptr %arg9, %18 : !tt.ptr<f32>, i64
      %32 = tt.make_tensor_ptr %31, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %33 = tt.addptr %arg5, %15 : !tt.ptr<f32>, i64
      %34 = tt.make_tensor_ptr %33, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
      %35 = tt.splat %20 : i32 -> tensor<128xi32>
      %36 = arith.addi %35, %5 : tensor<128xi32>
      %37 = tt.load %21 : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %38 = tt.load %28 : !tt.ptr<tensor<1x1xf32>>
      %39 = tt.broadcast %38 : tensor<1x1xf32> -> tensor<128x128xf32>
      %40:7 = scf.for %arg11 = %c0_i32 to %20 step %c128_i32 iter_args(%arg12 = %30, %arg13 = %32, %arg14 = %cst, %arg15 = %cst_4, %arg16 = %cst_0, %arg17 = %23, %arg18 = %25) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg18 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg12 : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %39 : tensor<128x128xf32>
        %63 = tt.broadcast %61 : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 : tensor<128x128xf32>
        %65 = arith.mulf %64, %cst_3 : tensor<128x128xf32>
        %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.maximumf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %67 = arith.maximumf %arg16, %66 : tensor<128xf32>
        %68 = tt.expand_dims %67 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %69 = tt.broadcast %68 : tensor<128x1xf32> -> tensor<128x128xf32>
        %70 = arith.subf %65, %69 : tensor<128x128xf32>
        %71 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %72 = math.exp %70 : tensor<128x128xf32>
        %73 = tt.fp_to_fp %72, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %74 = tt.load %arg17 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %75 = tt.dot %73, %74, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %76 = tt.load %arg13 : !tt.ptr<tensor<1x1xf32>>
        %77 = tt.broadcast %76 : tensor<1x1xf32> -> tensor<128x128xf32>
        %78 = arith.mulf %75, %77 : tensor<128x128xf32>
        %79 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %80 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.addf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %81 = arith.subf %arg16, %67 : tensor<128xf32>
        %82 = math.exp %81 : tensor<128xf32>
        %83 = arith.mulf %arg14, %82 : tensor<128xf32>
        %84 = arith.addf %83, %80 : tensor<128xf32>
        %85 = tt.expand_dims %82 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %86 = tt.broadcast %85 : tensor<128x1xf32> -> tensor<128x128xf32>
        %87 = arith.mulf %arg15, %86 : tensor<128x128xf32>
        %88 = arith.addf %87, %78 : tensor<128x128xf32>
        %89 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %90 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %71, %79, %84, %88, %67, %89, %90 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %41 = arith.muli %8, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %42 = arith.addi %8, %c1_i32 : i32
      %43 = arith.muli %42, %c128_i32 : i32
      %44 = tt.make_tensor_ptr %6, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%20, %41] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
      %45 = tt.advance %25, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %46 = tt.advance %23, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %47 = tt.broadcast %38 : tensor<1x1xf32> -> tensor<128x128xf32>
      %48:8 = scf.for %arg11 = %41 to %43 step %c128_i32 iter_args(%arg12 = %44, %arg13 = %30, %arg14 = %32, %arg15 = %40#2, %arg16 = %40#3, %arg17 = %40#4, %arg18 = %46, %arg19 = %45) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg19 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg13 : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %47 : tensor<128x128xf32>
        %63 = tt.broadcast %61 : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 : tensor<128x128xf32>
        %65 = tt.load %arg12 : !tt.ptr<tensor<128x128xi8>>
        %66 = arith.mulf %64, %cst_3 : tensor<128x128xf32>
        %67 = arith.cmpi ne, %65, %cst_1 : tensor<128x128xi8>
        %68 = arith.select %67, %cst_2, %cst_4 : tensor<128x128xi1>, tensor<128x128xf32>
        %69 = arith.addf %66, %68 : tensor<128x128xf32>
        %70 = "tt.reduce"(%69) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.maximumf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %71 = arith.maximumf %arg17, %70 : tensor<128xf32>
        %72 = tt.expand_dims %71 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.subf %69, %73 : tensor<128x128xf32>
        %75 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
        %76 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %77 = math.exp %74 : tensor<128x128xf32>
        %78 = tt.fp_to_fp %77, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %79 = tt.load %arg18 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %80 = tt.dot %78, %79, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %81 = tt.load %arg14 : !tt.ptr<tensor<1x1xf32>>
        %82 = tt.broadcast %81 : tensor<1x1xf32> -> tensor<128x128xf32>
        %83 = arith.mulf %80, %82 : tensor<128x128xf32>
        %84 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %85 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.addf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %86 = arith.subf %arg17, %71 : tensor<128xf32>
        %87 = math.exp %86 : tensor<128xf32>
        %88 = arith.mulf %arg15, %87 : tensor<128xf32>
        %89 = arith.addf %88, %85 : tensor<128xf32>
        %90 = tt.expand_dims %87 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.mulf %arg16, %91 : tensor<128x128xf32>
        %93 = arith.addf %92, %83 : tensor<128x128xf32>
        %94 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %95 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %75, %76, %84, %89, %93, %71, %94, %95 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %49 = math.log %48#3 : tensor<128xf32>
      %50 = arith.addf %48#5, %49 : tensor<128xf32>
      %51 = tt.expand_dims %48#3 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %52 = tt.broadcast %51 : tensor<128x1xf32> -> tensor<128x128xf32>
      %53 = arith.divf %48#4, %52 : tensor<128x128xf32>
      %54 = arith.muli %7, %c1024_i32 : i32
      %55 = tt.addptr %arg4, %54 : !tt.ptr<f32>, i32
      %56 = tt.splat %55 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %57 = tt.addptr %56, %36 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %57, %50 : tensor<128x!tt.ptr<f32>>
      tt.store %34, %53 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_fwd_fp8_Q2/norm_dot_false.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg9 = %0 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %2 = arith.divsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %3 = arith.remsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %4 = arith.divsi %2, %c8_i32 : i32
// CHECK-NEXT:       %5 = arith.remsi %2, %c8_i32 : i32
// CHECK-NEXT:       %6 = arith.extsi %4 : i32 to i64
// CHECK-NEXT:       %7 = arith.muli %6, %c1048576_i64 : i64
// CHECK-NEXT:       %8 = arith.extsi %5 : i32 to i64
// CHECK-NEXT:       %9 = arith.muli %8, %c131072_i64 : i64
// CHECK-NEXT:       %10 = arith.addi %7, %9 : i64
// CHECK-NEXT:       %11 = arith.muli %6, %c64_i64 : i64
// CHECK-NEXT:       %12 = arith.muli %8, %c8_i64 : i64
// CHECK-NEXT:       %13 = arith.addi %11, %12 : i64
// CHECK-NEXT:       %14 = tt.addptr %arg0, %10 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %15 = arith.muli %3, %c128_i32 : i32
// CHECK-NEXT:       %16 = tt.make_tensor_ptr %14, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %17 = tt.addptr %arg2, %10 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %18 = tt.make_tensor_ptr %17, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %19 = tt.addptr %arg1, %10 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %20 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %21 = tt.addptr %arg6, %13 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %22 = arith.divsi %15, %c128_i32 : i32
// CHECK-NEXT:       %23 = tt.make_tensor_ptr %21, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %24 = tt.addptr %arg7, %13 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %25 = tt.make_tensor_ptr %24, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %26 = tt.addptr %arg8, %13 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %27 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %28 = tt.addptr %arg4, %10 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %29 = tt.make_tensor_ptr %28, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
// CHECK-NEXT:       %30 = tt.splat %15 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %31 = arith.addi %30, %1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %32 = tt.load %16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %33 = tt.load %23 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %34 = tt.broadcast %33 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %35:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %25, %arg12 = %27, %arg13 = %cst, %arg14 = %cst_2, %arg15 = %cst_0, %arg16 = %18, %arg17 = %20) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %45 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %46 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %47 = tt.dot %32, %46, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %48 = tt.load %arg11 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %49 = arith.mulf %47, %34 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %50 = tt.broadcast %48 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %51 = arith.mulf %49, %50 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %52 = arith.mulf %51, %cst_1 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %78 = arith.maximumf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %78 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %54 = arith.maximumf %arg15, %53 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %55 = tt.expand_dims %54 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %56 = tt.broadcast %55 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %57 = arith.subf %52, %56 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %58 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %59 = math.exp %57 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %60 = tt.fp_to_fp %59 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %61 = tt.load %arg16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %62 = tt.dot %60, %61, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %63 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %64 = tt.broadcast %63 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %65 = arith.mulf %62, %64 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %66 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %67 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %78 = arith.addf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %78 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %68 = arith.subf %arg15, %54 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %69 = math.exp %68 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %70 = arith.mulf %arg13, %69 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %71 = arith.addf %70, %67 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %72 = tt.expand_dims %69 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %74 = arith.mulf %arg14, %73 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %75 = arith.addf %74, %65 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %76 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %77 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %58, %66, %71, %75, %54, %76, %77 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %36 = math.log %35#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %37 = arith.addf %35#4, %36 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %38 = tt.expand_dims %35#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %39 = tt.broadcast %38 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %40 = arith.divf %35#3, %39 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %41 = arith.muli %2, %c1024_i32 : i32
// CHECK-NEXT:       %42 = tt.addptr %arg3, %41 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %43 = tt.splat %42 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %44 = tt.addptr %43, %31 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       tt.store %44, %37 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %29, %40 : !tt.ptr<tensor<128x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c28_i32 = arith.constant 28 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i64 = arith.constant 8 : i64
    %c64_i64 = arith.constant 64 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    scf.for %arg9 = %0 to %c8192_i32 step %c28_i32  : i32 {
      %2 = arith.divsi %arg9, %c8_i32 : i32
      %3 = arith.remsi %arg9, %c8_i32 : i32
      %4 = arith.divsi %2, %c8_i32 : i32
      %5 = arith.remsi %2, %c8_i32 : i32
      %6 = arith.extsi %4 : i32 to i64
      %7 = arith.muli %6, %c1048576_i64 : i64
      %8 = arith.extsi %5 : i32 to i64
      %9 = arith.muli %8, %c131072_i64 : i64
      %10 = arith.addi %7, %9 : i64
      %11 = arith.muli %6, %c64_i64 : i64
      %12 = arith.muli %8, %c8_i64 : i64
      %13 = arith.addi %11, %12 : i64
      %14 = tt.addptr %arg0, %10 : !tt.ptr<f8E4M3FN>, i64
      %15 = arith.muli %3, %c128_i32 : i32
      %16 = tt.make_tensor_ptr %14, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %17 = tt.addptr %arg2, %10 : !tt.ptr<f8E4M3FN>, i64
      %18 = tt.make_tensor_ptr %17, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %19 = tt.addptr %arg1, %10 : !tt.ptr<f8E4M3FN>, i64
      %20 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %21 = tt.addptr %arg6, %13 : !tt.ptr<f32>, i64
      %22 = arith.divsi %15, %c128_i32 : i32
      %23 = tt.make_tensor_ptr %21, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %24 = tt.addptr %arg7, %13 : !tt.ptr<f32>, i64
      %25 = tt.make_tensor_ptr %24, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %26 = tt.addptr %arg8, %13 : !tt.ptr<f32>, i64
      %27 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %28 = tt.addptr %arg4, %10 : !tt.ptr<f32>, i64
      %29 = tt.make_tensor_ptr %28, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
      %30 = tt.splat %15 : i32 -> tensor<128xi32>
      %31 = arith.addi %30, %1 : tensor<128xi32>
      %32 = tt.load %16 : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %33 = tt.load %23 : !tt.ptr<tensor<1x1xf32>>
      %34 = tt.broadcast %33 : tensor<1x1xf32> -> tensor<128x128xf32>
      %35:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %25, %arg12 = %27, %arg13 = %cst, %arg14 = %cst_2, %arg15 = %cst_0, %arg16 = %18, %arg17 = %20) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %45 = tt.load %arg17 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %46 = tt.trans %45 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %47 = tt.dot %32, %46, %cst_2 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %48 = tt.load %arg11 : !tt.ptr<tensor<1x1xf32>>
        %49 = arith.mulf %47, %34 : tensor<128x128xf32>
        %50 = tt.broadcast %48 : tensor<1x1xf32> -> tensor<128x128xf32>
        %51 = arith.mulf %49, %50 : tensor<128x128xf32>
        %52 = arith.mulf %51, %cst_1 : tensor<128x128xf32>
        %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.maximumf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %54 = arith.maximumf %arg15, %53 : tensor<128xf32>
        %55 = tt.expand_dims %54 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %56 = tt.broadcast %55 : tensor<128x1xf32> -> tensor<128x128xf32>
        %57 = arith.subf %52, %56 : tensor<128x128xf32>
        %58 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %59 = math.exp %57 : tensor<128x128xf32>
        %60 = tt.fp_to_fp %59, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %61 = tt.load %arg16 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %62 = tt.dot %60, %61, %cst_2 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %63 = tt.load %arg12 : !tt.ptr<tensor<1x1xf32>>
        %64 = tt.broadcast %63 : tensor<1x1xf32> -> tensor<128x128xf32>
        %65 = arith.mulf %62, %64 : tensor<128x128xf32>
        %66 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %67 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.addf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %68 = arith.subf %arg15, %54 : tensor<128xf32>
        %69 = math.exp %68 : tensor<128xf32>
        %70 = arith.mulf %arg13, %69 : tensor<128xf32>
        %71 = arith.addf %70, %67 : tensor<128xf32>
        %72 = tt.expand_dims %69 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.mulf %arg14, %73 : tensor<128x128xf32>
        %75 = arith.addf %74, %65 : tensor<128x128xf32>
        %76 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %77 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %58, %66, %71, %75, %54, %76, %77 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %36 = math.log %35#2 : tensor<128xf32>
      %37 = arith.addf %35#4, %36 : tensor<128xf32>
      %38 = tt.expand_dims %35#2 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %39 = tt.broadcast %38 : tensor<128x1xf32> -> tensor<128x128xf32>
      %40 = arith.divf %35#3, %39 : tensor<128x128xf32>
      %41 = arith.muli %2, %c1024_i32 : i32
      %42 = tt.addptr %arg3, %41 : !tt.ptr<f32>, i32
      %43 = tt.splat %42 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %44 = tt.addptr %43, %31 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %44, %37 : tensor<128x!tt.ptr<f32>>
      tt.store %29, %40 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/fa_fwd_fp8_Q2/norm_dot_true.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.addptr %arg6, %0 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %2 = tt.load %1 : !tt.ptr<i32>
// CHECK-NEXT:     %3 = tt.addptr %1, %c1_i32 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %4 = tt.load %3 : !tt.ptr<i32>
// CHECK-NEXT:     %5 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %6 = tt.bitcast %arg3 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK-NEXT:     scf.for %arg10 = %2 to %4 step %c1_i32  : i32 {
// CHECK-NEXT:       %7 = arith.divsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %8 = arith.remsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %9 = arith.divsi %7, %c8_i32 : i32
// CHECK-NEXT:       %10 = arith.remsi %7, %c8_i32 : i32
// CHECK-NEXT:       %11 = arith.extsi %9 : i32 to i64
// CHECK-NEXT:       %12 = arith.muli %11, %c1048576_i64 : i64
// CHECK-NEXT:       %13 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %14 = arith.muli %13, %c131072_i64 : i64
// CHECK-NEXT:       %15 = arith.addi %12, %14 : i64
// CHECK-NEXT:       %16 = arith.muli %11, %c64_i64 : i64
// CHECK-NEXT:       %17 = arith.muli %13, %c8_i64 : i64
// CHECK-NEXT:       %18 = arith.addi %16, %17 : i64
// CHECK-NEXT:       %19 = tt.addptr %arg0, %15 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %20 = arith.muli %8, %c128_i32 : i32
// CHECK-NEXT:       %21 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %22 = tt.addptr %arg2, %15 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %23 = tt.make_tensor_ptr %22, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %24 = tt.addptr %arg1, %15 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %25 = tt.make_tensor_ptr %24, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %26 = tt.addptr %arg7, %18 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %27 = arith.divsi %20, %c128_i32 : i32
// CHECK-NEXT:       %28 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %29 = tt.addptr %arg8, %18 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %30 = tt.make_tensor_ptr %29, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %31 = tt.addptr %arg9, %18 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %32 = tt.make_tensor_ptr %31, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %33 = tt.addptr %arg5, %15 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %34 = tt.make_tensor_ptr %33, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
// CHECK-NEXT:       %35 = tt.splat %20 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %36 = arith.addi %35, %5 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %37 = tt.load %21 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %38 = tt.load %28 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %39 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %40:7 = scf.for %arg11 = %c0_i32 to %20 step %c128_i32 iter_args(%arg12 = %30, %arg13 = %32, %arg14 = %cst, %arg15 = %cst_4, %arg16 = %cst_0, %arg17 = %23, %arg18 = %25) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %58 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %61 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %62 = arith.mulf %60, %39 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %65 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %91 = arith.maximumf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %91 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %67 = arith.maximumf %arg16, %66 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %68 = tt.expand_dims %67 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %69 = tt.broadcast %68 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %70 = arith.subf %65, %69 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %72 = math.exp %70 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %73 = tt.fp_to_fp %72 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %74 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %75 = tt.dot %73, %74, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %76 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %77 = tt.broadcast %76 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %78 = arith.mulf %75, %77 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %79 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %80 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %91 = arith.addf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %91 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %81 = arith.subf %arg16, %67 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %82 = math.exp %81 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %83 = arith.mulf %arg14, %82 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %84 = arith.addf %83, %80 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %85 = tt.expand_dims %82 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %86 = tt.broadcast %85 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %87 = arith.mulf %arg15, %86 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %88 = arith.addf %87, %78 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %89 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %90 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %71, %79, %84, %88, %67, %89, %90 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %41 = arith.muli %8, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
// CHECK-NEXT:       %42 = arith.addi %8, %c1_i32 : i32
// CHECK-NEXT:       %43 = arith.muli %42, %c128_i32 : i32
// CHECK-NEXT:       %44 = tt.make_tensor_ptr %6, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%20, %41] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
// CHECK-NEXT:       %45 = tt.advance %25, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %46 = tt.advance %23, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %47 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %48:8 = scf.for %arg11 = %41 to %43 step %c128_i32 iter_args(%arg12 = %44, %arg13 = %30, %arg14 = %32, %arg15 = %40#2, %arg16 = %40#3, %arg17 = %40#4, %arg18 = %46, %arg19 = %45) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %58 = tt.load %arg19 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %61 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %62 = arith.mulf %60, %47 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %65 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<128x128xi8>>
// CHECK-NEXT:         %66 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %67 = arith.cmpi ne, %65, %cst_1 {DataUse} : tensor<128x128xi8>
// CHECK-NEXT:         %68 = arith.select %67, %cst_2, %cst_4 {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
// CHECK-NEXT:         %69 = arith.addf %66, %68 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %70 = "tt.reduce"(%69) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %96 = arith.maximumf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %96 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %71 = arith.maximumf %arg17, %70 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %72 = tt.expand_dims %71 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %74 = arith.subf %69, %73 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %75 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
// CHECK-NEXT:         %76 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %77 = math.exp %74 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %78 = tt.fp_to_fp %77 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %79 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %80 = tt.dot %78, %79, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %81 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %82 = tt.broadcast %81 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %83 = arith.mulf %80, %82 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %84 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %85 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %96 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %96 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %86 = arith.subf %arg17, %71 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %87 = math.exp %86 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %88 = arith.mulf %arg15, %87 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %89 = arith.addf %88, %85 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %90 = tt.expand_dims %87 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.mulf %arg16, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = arith.addf %92, %83 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %94 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %95 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %75, %76, %84, %89, %93, %71, %94, %95 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %49 = math.log %48#3 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %50 = arith.addf %48#5, %49 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %51 = tt.expand_dims %48#3 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %52 = tt.broadcast %51 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %53 = arith.divf %48#4, %52 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %54 = arith.muli %7, %c1024_i32 : i32
// CHECK-NEXT:       %55 = tt.addptr %arg4, %54 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %56 = tt.splat %55 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %57 = tt.addptr %56, %36 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       tt.store %57, %50 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %34, %53 : !tt.ptr<tensor<128x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant dense<0> : tensor<128x128xi8>
    %cst_2 = arith.constant dense<-1.000000e+04> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i64 = arith.constant 8 : i64
    %c64_i64 = arith.constant 64 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg6, %0 : !tt.ptr<i32>, i32
    %2 = tt.load %1 : !tt.ptr<i32>
    %3 = tt.addptr %1, %c1_i32 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %6 = tt.bitcast %arg3 : !tt.ptr<i1> -> !tt.ptr<i8>
    scf.for %arg10 = %2 to %4 step %c1_i32  : i32 {
      %7 = arith.divsi %arg10, %c8_i32 : i32
      %8 = arith.remsi %arg10, %c8_i32 : i32
      %9 = arith.divsi %7, %c8_i32 : i32
      %10 = arith.remsi %7, %c8_i32 : i32
      %11 = arith.extsi %9 : i32 to i64
      %12 = arith.muli %11, %c1048576_i64 : i64
      %13 = arith.extsi %10 : i32 to i64
      %14 = arith.muli %13, %c131072_i64 : i64
      %15 = arith.addi %12, %14 : i64
      %16 = arith.muli %11, %c64_i64 : i64
      %17 = arith.muli %13, %c8_i64 : i64
      %18 = arith.addi %16, %17 : i64
      %19 = tt.addptr %arg0, %15 : !tt.ptr<f8E4M3FN>, i64
      %20 = arith.muli %8, %c128_i32 : i32
      %21 = tt.make_tensor_ptr %19, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %22 = tt.addptr %arg2, %15 : !tt.ptr<f8E4M3FN>, i64
      %23 = tt.make_tensor_ptr %22, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %24 = tt.addptr %arg1, %15 : !tt.ptr<f8E4M3FN>, i64
      %25 = tt.make_tensor_ptr %24, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
      %26 = tt.addptr %arg7, %18 : !tt.ptr<f32>, i64
      %27 = arith.divsi %20, %c128_i32 : i32
      %28 = tt.make_tensor_ptr %26, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %29 = tt.addptr %arg8, %18 : !tt.ptr<f32>, i64
      %30 = tt.make_tensor_ptr %29, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %31 = tt.addptr %arg9, %18 : !tt.ptr<f32>, i64
      %32 = tt.make_tensor_ptr %31, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
      %33 = tt.addptr %arg5, %15 : !tt.ptr<f32>, i64
      %34 = tt.make_tensor_ptr %33, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
      %35 = tt.splat %20 : i32 -> tensor<128xi32>
      %36 = arith.addi %35, %5 : tensor<128xi32>
      %37 = tt.load %21 : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %38 = tt.load %28 : !tt.ptr<tensor<1x1xf32>>
      %39 = tt.broadcast %38 : tensor<1x1xf32> -> tensor<128x128xf32>
      %40:7 = scf.for %arg11 = %c0_i32 to %20 step %c128_i32 iter_args(%arg12 = %30, %arg13 = %32, %arg14 = %cst, %arg15 = %cst_4, %arg16 = %cst_0, %arg17 = %23, %arg18 = %25) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg18 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg12 : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %39 : tensor<128x128xf32>
        %63 = tt.broadcast %61 : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 : tensor<128x128xf32>
        %65 = arith.mulf %64, %cst_3 : tensor<128x128xf32>
        %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.maximumf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %67 = arith.maximumf %arg16, %66 : tensor<128xf32>
        %68 = tt.expand_dims %67 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %69 = tt.broadcast %68 : tensor<128x1xf32> -> tensor<128x128xf32>
        %70 = arith.subf %65, %69 : tensor<128x128xf32>
        %71 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %72 = math.exp %70 : tensor<128x128xf32>
        %73 = tt.fp_to_fp %72, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %74 = tt.load %arg17 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %75 = tt.dot %73, %74, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %76 = tt.load %arg13 : !tt.ptr<tensor<1x1xf32>>
        %77 = tt.broadcast %76 : tensor<1x1xf32> -> tensor<128x128xf32>
        %78 = arith.mulf %75, %77 : tensor<128x128xf32>
        %79 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %80 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.addf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %81 = arith.subf %arg16, %67 : tensor<128xf32>
        %82 = math.exp %81 : tensor<128xf32>
        %83 = arith.mulf %arg14, %82 : tensor<128xf32>
        %84 = arith.addf %83, %80 : tensor<128xf32>
        %85 = tt.expand_dims %82 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %86 = tt.broadcast %85 : tensor<128x1xf32> -> tensor<128x128xf32>
        %87 = arith.mulf %arg15, %86 : tensor<128x128xf32>
        %88 = arith.addf %87, %78 : tensor<128x128xf32>
        %89 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %90 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %71, %79, %84, %88, %67, %89, %90 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %41 = arith.muli %8, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %42 = arith.addi %8, %c1_i32 : i32
      %43 = arith.muli %42, %c128_i32 : i32
      %44 = tt.make_tensor_ptr %6, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%20, %41] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
      %45 = tt.advance %25, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %46 = tt.advance %23, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %47 = tt.broadcast %38 : tensor<1x1xf32> -> tensor<128x128xf32>
      %48:8 = scf.for %arg11 = %41 to %43 step %c128_i32 iter_args(%arg12 = %44, %arg13 = %30, %arg14 = %32, %arg15 = %40#2, %arg16 = %40#3, %arg17 = %40#4, %arg18 = %46, %arg19 = %45) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg19 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg13 : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %47 : tensor<128x128xf32>
        %63 = tt.broadcast %61 : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 : tensor<128x128xf32>
        %65 = tt.load %arg12 : !tt.ptr<tensor<128x128xi8>>
        %66 = arith.mulf %64, %cst_3 : tensor<128x128xf32>
        %67 = arith.cmpi ne, %65, %cst_1 : tensor<128x128xi8>
        %68 = arith.select %67, %cst_2, %cst_4 : tensor<128x128xi1>, tensor<128x128xf32>
        %69 = arith.addf %66, %68 : tensor<128x128xf32>
        %70 = "tt.reduce"(%69) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.maximumf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %71 = arith.maximumf %arg17, %70 : tensor<128xf32>
        %72 = tt.expand_dims %71 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.subf %69, %73 : tensor<128x128xf32>
        %75 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
        %76 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %77 = math.exp %74 : tensor<128x128xf32>
        %78 = tt.fp_to_fp %77, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %79 = tt.load %arg18 : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %80 = tt.dot %78, %79, %cst_4 {triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %81 = tt.load %arg14 : !tt.ptr<tensor<1x1xf32>>
        %82 = tt.broadcast %81 : tensor<1x1xf32> -> tensor<128x128xf32>
        %83 = arith.mulf %80, %82 : tensor<128x128xf32>
        %84 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %85 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.addf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %86 = arith.subf %arg17, %71 : tensor<128xf32>
        %87 = math.exp %86 : tensor<128xf32>
        %88 = arith.mulf %arg15, %87 : tensor<128xf32>
        %89 = arith.addf %88, %85 : tensor<128xf32>
        %90 = tt.expand_dims %87 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.mulf %arg16, %91 : tensor<128x128xf32>
        %93 = arith.addf %92, %83 : tensor<128x128xf32>
        %94 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %95 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %75, %76, %84, %89, %93, %71, %94, %95 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %49 = math.log %48#3 : tensor<128xf32>
      %50 = arith.addf %48#5, %49 : tensor<128xf32>
      %51 = tt.expand_dims %48#3 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %52 = tt.broadcast %51 : tensor<128x1xf32> -> tensor<128x128xf32>
      %53 = arith.divf %48#4, %52 : tensor<128x128xf32>
      %54 = arith.muli %7, %c1024_i32 : i32
      %55 = tt.addptr %arg4, %54 : !tt.ptr<f32>, i32
      %56 = tt.splat %55 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %57 = tt.addptr %56, %36 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %57, %50 : tensor<128x!tt.ptr<f32>>
      tt.store %34, %53 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/gqa_pack_fwd/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_gqa_pack_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0xFF800000> : tensor<16xf32>
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<1x64xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<1x64xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x64xbf16>
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16x64xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16xf32>
// CHECK-NEXT:     %c-2_i32 = arith.constant -2 : i32
// CHECK-NEXT:     %c-4_i32 = arith.constant -4 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_5 = arith.constant 0.180336878 : f32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = arith.muli %1, %arg10 : i32
// CHECK-NEXT:     %3 = arith.muli %1, %arg13 : i32
// CHECK-NEXT:     %4 = arith.muli %0, %arg11 : i32
// CHECK-NEXT:     %5 = arith.addi %2, %4 : i32
// CHECK-NEXT:     %6 = tt.addptr %arg0, %5 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %7 = tt.addptr %arg1, %3 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %8 = tt.addptr %arg2, %3 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %9 = tt.addptr %arg3, %3 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %10 = tt.addptr %arg4, %3 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %11 = arith.muli %1, %arg25 : i32
// CHECK-NEXT:     %12 = arith.muli %0, %arg26 : i32
// CHECK-NEXT:     %13 = arith.addi %11, %12 : i32
// CHECK-NEXT:     %14 = tt.addptr %arg5, %13 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %15 = arith.muli %1, %arg28 : i32
// CHECK-NEXT:     %16 = arith.addi %15, %0 : i32
// CHECK-NEXT:     %17 = tt.addptr %arg6, %16 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %18 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %19 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %20 = tt.make_range {MetaUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT:     %21 = tt.expand_dims %20 {MetaUse, axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT:     %22 = tt.splat %arg12 {MetaUse} : i32 -> tensor<16x1xi32>
// CHECK-NEXT:     %23 = arith.muli %21, %22 {MetaUse} : tensor<16x1xi32>
// CHECK-NEXT:     %24 = tt.expand_dims %18 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %25 = tt.broadcast %23 {MetaUse} : tensor<16x1xi32> -> tensor<16x64xi32>
// CHECK-NEXT:     %26 = tt.broadcast %24 {MetaUse} : tensor<1x64xi32> -> tensor<16x64xi32>
// CHECK-NEXT:     %27 = arith.addi %25, %26 {MetaUse} : tensor<16x64xi32>
// CHECK-NEXT:     %28 = tt.splat %6 {MetaUse} : !tt.ptr<bf16> -> tensor<16x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %29 = tt.addptr %28, %27 {MetaUse} : tensor<16x64x!tt.ptr<bf16>>, tensor<16x64xi32>
// CHECK-NEXT:     %30 = tt.load %29 {DataUse} : tensor<16x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %31 = arith.truncf %cst_5 : f32 to bf16
// CHECK-NEXT:     %32 = tt.splat %31 {DataUse} : bf16 -> tensor<16x64xbf16>
// CHECK-NEXT:     %33 = arith.mulf %30, %32 {DataUse} : tensor<16x64xbf16>
// CHECK-NEXT:     %34 = arith.addi %0, %c-4_i32 : i32
// CHECK-NEXT:     %35 = arith.maxsi %34, %c0_i32 : i32
// CHECK-NEXT:     %36 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:     %37 = arith.minsi %arg8, %36 : i32
// CHECK-NEXT:     %38 = arith.subi %37, %35 : i32
// CHECK-NEXT:     %39 = arith.remsi %38, %c64_i32 : i32
// CHECK-NEXT:     %40 = arith.cmpi sgt, %39, %c0_i32 : i32
// CHECK-NEXT:     %41 = arith.addi %38, %c63_i32 : i32
// CHECK-NEXT:     %42 = arith.divsi %41, %c64_i32 : i32
// CHECK-NEXT:     %43 = scf.if %40 -> (i32) {
// CHECK-NEXT:       %72 = arith.subi %42, %c1_i32 : i32
// CHECK-NEXT:       scf.yield %72 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %42 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %44 = arith.addi %0, %c-2_i32 : i32
// CHECK-NEXT:     %45 = arith.maxsi %44, %c0_i32 : i32
// CHECK-NEXT:     %46 = tt.splat %7 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:     %47 = tt.splat %9 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:     %48 = tt.splat %arg17 {MetaUse} : i32 -> tensor<1x64xi32>
// CHECK-NEXT:     %49 = tt.expand_dims %18 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %50 = tt.broadcast %49 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %51 = tt.splat %arg23 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %52 = tt.broadcast %24 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %53 = tt.splat %8 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %54 = tt.splat %10 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %55:3 = scf.for %arg30 = %45 to %37 step %c1_i32 iter_args(%arg31 = %cst_3, %arg32 = %cst, %arg33 = %cst_4) -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
// CHECK-NEXT:       %72 = arith.muli %arg30, %arg14 : i32
// CHECK-NEXT:       %73 = tt.splat %72 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %74 = arith.addi %73, %18 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %75 = tt.addptr %46, %74 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %76 = tt.load %75 {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %77 = tt.expand_dims %76 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:       %78 = tt.broadcast %77 {DataUse} : tensor<1x64xbf16> -> tensor<16x64xbf16>
// CHECK-NEXT:       %79 = arith.mulf %33, %78 {DataUse} : tensor<16x64xbf16>
// CHECK-NEXT:       %80 = arith.muli %arg30, %arg20 : i32
// CHECK-NEXT:       %81 = tt.splat %80 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %82 = arith.addi %81, %18 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %83 = tt.addptr %47, %82 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %84 = tt.load %83 {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %85 = tt.expand_dims %84 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:       %86 = arith.extf %85 {DataUse} : tensor<1x64xbf16> to tensor<1x64xf32>
// CHECK-NEXT:       %87 = tt.broadcast %86 {DataUse} : tensor<1x64xf32> -> tensor<16x64xf32>
// CHECK-NEXT:       %88:3 = scf.for %arg34 = %c0_i32 to %43 step %c1_i32 iter_args(%arg35 = %arg31, %arg36 = %arg32, %arg37 = %arg33) -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
// CHECK-NEXT:         %90 = arith.muli %arg34, %c64_i32 : i32
// CHECK-NEXT:         %91 = arith.addi %35, %90 : i32
// CHECK-NEXT:         %92 = tt.splat %91 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %93 = arith.addi %92, %18 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %94 = tt.expand_dims %93 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:         %95 = arith.muli %94, %48 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:         %96 = tt.broadcast %95 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:         %97 = arith.addi %96, %50 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:         %98 = tt.expand_dims %93 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %99 = arith.muli %98, %51 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:         %101 = arith.addi %100, %52 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:         %102 = tt.addptr %53, %97 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:         %103 = tt.load %102 {DataUse} : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:         %104 = tt.addptr %54, %101 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:         %105 = tt.load %104 {DataUse} : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:         %106 = tt.dot %79, %103, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
// CHECK-NEXT:         %107 = "tt.reduce"(%106) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg38: f32, %arg39: f32):
// CHECK-NEXT:           %125 = arith.maxnumf %arg38, %arg39 : f32
// CHECK-NEXT:           tt.reduce.return %125 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x64xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %108 = arith.maxnumf %arg36, %107 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %109 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %110 = tt.broadcast %109 {DataUse} : tensor<16x1xf32> -> tensor<16x64xf32>
// CHECK-NEXT:         %111 = arith.subf %106, %110 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %112 = math.exp2 %111 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %113 = "tt.reduce"(%112) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg38: f32, %arg39: f32):
// CHECK-NEXT:           %125 = arith.addf %arg38, %arg39 : f32
// CHECK-NEXT:           tt.reduce.return %125 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x64xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %114 = arith.subf %arg36, %108 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %115 = math.exp2 %114 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %116 = arith.mulf %arg37, %115 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %117 = arith.addf %116, %113 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %118 = tt.expand_dims %115 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %119 = tt.broadcast %118 {DataUse} : tensor<16x1xf32> -> tensor<16x64xf32>
// CHECK-NEXT:         %120 = arith.mulf %arg35, %119 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %121 = arith.truncf %112 {DataUse} : tensor<16x64xf32> to tensor<16x64xbf16>
// CHECK-NEXT:         %122 = tt.dot %121, %105, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
// CHECK-NEXT:         %123 = arith.mulf %122, %87 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %124 = arith.addf %120, %123 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         scf.yield %124, %108, %117 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
// CHECK-NEXT:       } {DataUse, tt.num_stages = 1 : i32}
// CHECK-NEXT:       %89:3 = scf.if %40 -> (tensor<16xf32>, tensor<16xf32>, tensor<16x64xf32>) {
// CHECK-NEXT:         %90 = arith.muli %43, %c64_i32 : i32
// CHECK-NEXT:         %91 = arith.addi %35, %90 : i32
// CHECK-NEXT:         %92 = tt.splat %91 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %93 = tt.splat %91 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %94 = arith.addi %92, %18 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %95 = arith.addi %93, %19 {DataUse} : tensor<64xi32>
// CHECK-NEXT:         %96 = tt.expand_dims %94 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:         %97 = tt.splat %arg17 {MetaUse} : i32 -> tensor<1x64xi32>
// CHECK-NEXT:         %98 = arith.muli %96, %97 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:         %99 = tt.expand_dims %18 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %100 = tt.broadcast %98 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:         %101 = tt.broadcast %99 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:         %102 = arith.addi %100, %101 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:         %103 = tt.expand_dims %94 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %104 = tt.splat %arg23 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:         %105 = arith.muli %103, %104 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %106 = tt.broadcast %105 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:         %107 = tt.broadcast %24 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:         %108 = arith.addi %106, %107 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:         %109 = arith.subi %0, %c5_i32 : i32
// CHECK-NEXT:         %110 = tt.splat %109 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %111 = tt.splat %109 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %112 = arith.cmpi slt, %110, %94 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %113 = arith.cmpi slt, %111, %95 {DataUse} : tensor<64xi32>
// CHECK-NEXT:         %114 = tt.splat %0 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %115 = tt.splat %0 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %116 = arith.cmpi sle, %94, %114 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %117 = arith.cmpi sle, %95, %115 {DataUse} : tensor<64xi32>
// CHECK-NEXT:         %118 = arith.andi %112, %116 {MetaUse} : tensor<64xi1>
// CHECK-NEXT:         %119 = arith.andi %113, %117 {DataUse} : tensor<64xi1>
// CHECK-NEXT:         %120 = tt.expand_dims %118 {MetaUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:         %121 = tt.expand_dims %119 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:         %122 = tt.splat %8 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:         %123 = tt.addptr %122, %102 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:         %124 = tt.broadcast %120 {MetaUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:         %125 = tt.load %123, %124, %cst_2 {DataUse} : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:         %126 = tt.expand_dims %118 {MetaUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:         %127 = tt.splat %10 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:         %128 = tt.addptr %127, %108 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:         %129 = tt.broadcast %126 {MetaUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:         %130 = tt.load %128, %129, %cst_2 {DataUse} : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:         %131 = arith.select %121, %cst_0, %cst_1 {DataUse} : tensor<1x64xi1>, tensor<1x64xf32>
// CHECK-NEXT:         %132 = tt.broadcast %131 {DataUse} : tensor<1x64xf32> -> tensor<16x64xf32>
// CHECK-NEXT:         %133 = tt.dot %79, %125, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
// CHECK-NEXT:         %134 = arith.addf %133, %132 {DataUse, triton_cv12.add_from_dot} : tensor<16x64xf32>
// CHECK-NEXT:         %135 = "tt.reduce"(%134) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg34: f32, %arg35: f32):
// CHECK-NEXT:           %155 = arith.maxnumf %arg34, %arg35 : f32
// CHECK-NEXT:           tt.reduce.return %155 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x64xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %136 = arith.maxnumf %88#1, %135 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %137 = tt.expand_dims %136 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %138 = tt.broadcast %137 {DataUse} : tensor<16x1xf32> -> tensor<16x64xf32>
// CHECK-NEXT:         %139 = arith.subf %134, %138 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %140 = math.exp2 %139 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %141 = "tt.reduce"(%140) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg34: f32, %arg35: f32):
// CHECK-NEXT:           %155 = arith.addf %arg34, %arg35 : f32
// CHECK-NEXT:           tt.reduce.return %155 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x64xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %142 = arith.subf %88#1, %136 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %143 = math.exp2 %142 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %144 = arith.mulf %88#2, %143 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %145 = arith.addf %144, %141 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %146 = tt.expand_dims %143 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %147 = tt.broadcast %146 {DataUse} : tensor<16x1xf32> -> tensor<16x64xf32>
// CHECK-NEXT:         %148 = arith.mulf %88#0, %147 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %149 = arith.truncf %140 {DataUse} : tensor<16x64xf32> to tensor<16x64xbf16>
// CHECK-NEXT:         %150 = tt.dot %149, %130, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
// CHECK-NEXT:         %151 = arith.extf %85 {DataUse} : tensor<1x64xbf16> to tensor<1x64xf32>
// CHECK-NEXT:         %152 = tt.broadcast %151 {DataUse} : tensor<1x64xf32> -> tensor<16x64xf32>
// CHECK-NEXT:         %153 = arith.mulf %150, %152 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         %154 = arith.addf %148, %153 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:         scf.yield %136, %145, %154 : tensor<16xf32>, tensor<16xf32>, tensor<16x64xf32>
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %88#1, %88#2, %88#0 : tensor<16xf32>, tensor<16xf32>, tensor<16x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       scf.yield %89#2, %89#0, %89#1 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
// CHECK-NEXT:     } {DataUse, tt.num_stages = 16 : i32}
// CHECK-NEXT:     %56 = tt.expand_dims %55#2 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:     %57 = tt.broadcast %56 {DataUse} : tensor<16x1xf32> -> tensor<16x64xf32>
// CHECK-NEXT:     %58 = arith.divf %55#0, %57 {DataUse} : tensor<16x64xf32>
// CHECK-NEXT:     %59 = arith.truncf %58 {DataUse} : tensor<16x64xf32> to tensor<16x64xbf16>
// CHECK-NEXT:     %60 = tt.splat %arg27 {MetaUse} : i32 -> tensor<16x1xi32>
// CHECK-NEXT:     %61 = arith.muli %21, %60 {MetaUse} : tensor<16x1xi32>
// CHECK-NEXT:     %62 = tt.broadcast %61 {MetaUse} : tensor<16x1xi32> -> tensor<16x64xi32>
// CHECK-NEXT:     %63 = arith.addi %62, %26 {MetaUse} : tensor<16x64xi32>
// CHECK-NEXT:     %64 = tt.splat %14 {MetaUse} : !tt.ptr<bf16> -> tensor<16x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %65 = tt.addptr %64, %63 {MetaUse} : tensor<16x64x!tt.ptr<bf16>>, tensor<16x64xi32>
// CHECK-NEXT:     tt.store %65, %59 : tensor<16x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %66 = math.log %55#2 {DataUse} : tensor<16xf32>
// CHECK-NEXT:     %67 = arith.addf %55#1, %66 {DataUse} : tensor<16xf32>
// CHECK-NEXT:     %68 = tt.splat %arg29 {MetaUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:     %69 = arith.muli %20, %68 {MetaUse} : tensor<16xi32>
// CHECK-NEXT:     %70 = tt.splat %17 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:     %71 = tt.addptr %70, %69 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:     tt.store %71, %67 : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_gqa_pack_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<16xf32>
    %c5_i32 = arith.constant 5 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
    %cst_1 = arith.constant dense<-1.000000e+06> : tensor<1x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x64xbf16>
    %c63_i32 = arith.constant 63 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c-2_i32 = arith.constant -2 : i32
    %c-4_i32 = arith.constant -4 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_5 = arith.constant 0.180336878 : f32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg10 : i32
    %3 = arith.muli %1, %arg13 : i32
    %4 = arith.muli %0, %arg11 : i32
    %5 = arith.addi %2, %4 : i32
    %6 = tt.addptr %arg0, %5 : !tt.ptr<bf16>, i32
    %7 = tt.addptr %arg1, %3 : !tt.ptr<bf16>, i32
    %8 = tt.addptr %arg2, %3 : !tt.ptr<bf16>, i32
    %9 = tt.addptr %arg3, %3 : !tt.ptr<bf16>, i32
    %10 = tt.addptr %arg4, %3 : !tt.ptr<bf16>, i32
    %11 = arith.muli %1, %arg25 : i32
    %12 = arith.muli %0, %arg26 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = tt.addptr %arg5, %13 : !tt.ptr<bf16>, i32
    %15 = arith.muli %1, %arg28 : i32
    %16 = arith.addi %15, %0 : i32
    %17 = tt.addptr %arg6, %16 : !tt.ptr<f32>, i32
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %19 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %20 = tt.expand_dims %19 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %21 = tt.splat %arg12 : i32 -> tensor<16x1xi32>
    %22 = arith.muli %20, %21 : tensor<16x1xi32>
    %23 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %24 = tt.broadcast %22 : tensor<16x1xi32> -> tensor<16x64xi32>
    %25 = tt.broadcast %23 : tensor<1x64xi32> -> tensor<16x64xi32>
    %26 = arith.addi %24, %25 : tensor<16x64xi32>
    %27 = tt.splat %6 : !tt.ptr<bf16> -> tensor<16x64x!tt.ptr<bf16>>
    %28 = tt.addptr %27, %26 : tensor<16x64x!tt.ptr<bf16>>, tensor<16x64xi32>
    %29 = tt.load %28 : tensor<16x64x!tt.ptr<bf16>>
    %30 = arith.truncf %cst_5 : f32 to bf16
    %31 = tt.splat %30 : bf16 -> tensor<16x64xbf16>
    %32 = arith.mulf %29, %31 : tensor<16x64xbf16>
    %33 = arith.addi %0, %c-4_i32 : i32
    %34 = arith.maxsi %33, %c0_i32 : i32
    %35 = arith.addi %0, %c1_i32 : i32
    %36 = arith.minsi %arg8, %35 : i32
    %37 = arith.subi %36, %34 : i32
    %38 = arith.remsi %37, %c64_i32 : i32
    %39 = arith.cmpi sgt, %38, %c0_i32 : i32
    %40 = arith.addi %37, %c63_i32 : i32
    %41 = arith.divsi %40, %c64_i32 : i32
    %42 = scf.if %39 -> (i32) {
      %71 = arith.subi %41, %c1_i32 : i32
      scf.yield %71 : i32
    } else {
      scf.yield %41 : i32
    }
    %43 = arith.addi %0, %c-2_i32 : i32
    %44 = arith.maxsi %43, %c0_i32 : i32
    %45 = tt.splat %7 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %46 = tt.splat %9 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %47 = tt.splat %arg17 : i32 -> tensor<1x64xi32>
    %48 = tt.expand_dims %18 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %49 = tt.broadcast %48 : tensor<64x1xi32> -> tensor<64x64xi32>
    %50 = tt.splat %arg23 : i32 -> tensor<64x1xi32>
    %51 = tt.broadcast %23 : tensor<1x64xi32> -> tensor<64x64xi32>
    %52 = tt.splat %8 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %53 = tt.splat %10 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %54:3 = scf.for %arg30 = %44 to %36 step %c1_i32 iter_args(%arg31 = %cst_3, %arg32 = %cst, %arg33 = %cst_4) -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
      %71 = arith.muli %arg30, %arg14 : i32
      %72 = tt.splat %71 : i32 -> tensor<64xi32>
      %73 = arith.addi %72, %18 : tensor<64xi32>
      %74 = tt.addptr %45, %73 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %75 = tt.load %74 : tensor<64x!tt.ptr<bf16>>
      %76 = tt.expand_dims %75 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
      %77 = tt.broadcast %76 : tensor<1x64xbf16> -> tensor<16x64xbf16>
      %78 = arith.mulf %32, %77 : tensor<16x64xbf16>
      %79 = arith.muli %arg30, %arg20 : i32
      %80 = tt.splat %79 : i32 -> tensor<64xi32>
      %81 = arith.addi %80, %18 : tensor<64xi32>
      %82 = tt.addptr %46, %81 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %83 = tt.load %82 : tensor<64x!tt.ptr<bf16>>
      %84 = tt.expand_dims %83 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
      %85 = arith.extf %84 : tensor<1x64xbf16> to tensor<1x64xf32>
      %86 = tt.broadcast %85 : tensor<1x64xf32> -> tensor<16x64xf32>
      %87:3 = scf.for %arg34 = %c0_i32 to %42 step %c1_i32 iter_args(%arg35 = %arg31, %arg36 = %arg32, %arg37 = %arg33) -> (tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
        %89 = arith.muli %arg34, %c64_i32 : i32
        %90 = arith.addi %34, %89 : i32
        %91 = tt.splat %90 : i32 -> tensor<64xi32>
        %92 = arith.addi %91, %18 : tensor<64xi32>
        %93 = tt.expand_dims %92 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %94 = arith.muli %93, %47 : tensor<1x64xi32>
        %95 = tt.broadcast %94 : tensor<1x64xi32> -> tensor<64x64xi32>
        %96 = arith.addi %95, %49 : tensor<64x64xi32>
        %97 = tt.expand_dims %92 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %98 = arith.muli %97, %50 : tensor<64x1xi32>
        %99 = tt.broadcast %98 : tensor<64x1xi32> -> tensor<64x64xi32>
        %100 = arith.addi %99, %51 : tensor<64x64xi32>
        %101 = tt.addptr %52, %96 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
        %102 = tt.load %101 : tensor<64x64x!tt.ptr<bf16>>
        %103 = tt.addptr %53, %100 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
        %104 = tt.load %103 : tensor<64x64x!tt.ptr<bf16>>
        %105 = tt.dot %78, %102, %cst_3 {triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
        %106 = "tt.reduce"(%105) <{axis = 1 : i32}> ({
        ^bb0(%arg38: f32, %arg39: f32):
          %124 = arith.maxnumf %arg38, %arg39 : f32
          tt.reduce.return %124 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %107 = arith.maxnumf %arg36, %106 : tensor<16xf32>
        %108 = tt.expand_dims %107 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %109 = tt.broadcast %108 : tensor<16x1xf32> -> tensor<16x64xf32>
        %110 = arith.subf %105, %109 : tensor<16x64xf32>
        %111 = math.exp2 %110 : tensor<16x64xf32>
        %112 = "tt.reduce"(%111) <{axis = 1 : i32}> ({
        ^bb0(%arg38: f32, %arg39: f32):
          %124 = arith.addf %arg38, %arg39 : f32
          tt.reduce.return %124 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %113 = arith.subf %arg36, %107 : tensor<16xf32>
        %114 = math.exp2 %113 : tensor<16xf32>
        %115 = arith.mulf %arg37, %114 : tensor<16xf32>
        %116 = arith.addf %115, %112 : tensor<16xf32>
        %117 = tt.expand_dims %114 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %118 = tt.broadcast %117 : tensor<16x1xf32> -> tensor<16x64xf32>
        %119 = arith.mulf %arg35, %118 : tensor<16x64xf32>
        %120 = arith.truncf %111 : tensor<16x64xf32> to tensor<16x64xbf16>
        %121 = tt.dot %120, %104, %cst_3 {triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
        %122 = arith.mulf %121, %86 : tensor<16x64xf32>
        %123 = arith.addf %119, %122 : tensor<16x64xf32>
        scf.yield %123, %107, %116 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
      } {tt.num_stages = 1 : i32}
      %88:3 = scf.if %39 -> (tensor<16xf32>, tensor<16xf32>, tensor<16x64xf32>) {
        %89 = arith.muli %42, %c64_i32 : i32
        %90 = arith.addi %34, %89 : i32
        %91 = tt.splat %90 : i32 -> tensor<64xi32>
        %92 = arith.addi %91, %18 : tensor<64xi32>
        %93 = tt.expand_dims %92 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %94 = tt.splat %arg17 : i32 -> tensor<1x64xi32>
        %95 = arith.muli %93, %94 : tensor<1x64xi32>
        %96 = tt.expand_dims %18 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %97 = tt.broadcast %95 : tensor<1x64xi32> -> tensor<64x64xi32>
        %98 = tt.broadcast %96 : tensor<64x1xi32> -> tensor<64x64xi32>
        %99 = arith.addi %97, %98 : tensor<64x64xi32>
        %100 = tt.expand_dims %92 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %101 = tt.splat %arg23 : i32 -> tensor<64x1xi32>
        %102 = arith.muli %100, %101 : tensor<64x1xi32>
        %103 = tt.broadcast %102 : tensor<64x1xi32> -> tensor<64x64xi32>
        %104 = tt.broadcast %23 : tensor<1x64xi32> -> tensor<64x64xi32>
        %105 = arith.addi %103, %104 : tensor<64x64xi32>
        %106 = arith.subi %0, %c5_i32 : i32
        %107 = tt.splat %106 : i32 -> tensor<64xi32>
        %108 = arith.cmpi slt, %107, %92 : tensor<64xi32>
        %109 = tt.splat %0 : i32 -> tensor<64xi32>
        %110 = arith.cmpi sle, %92, %109 : tensor<64xi32>
        %111 = arith.andi %108, %110 : tensor<64xi1>
        %112 = tt.expand_dims %111 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
        %113 = tt.splat %8 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
        %114 = tt.addptr %113, %99 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
        %115 = tt.broadcast %112 : tensor<1x64xi1> -> tensor<64x64xi1>
        %116 = tt.load %114, %115, %cst_2 : tensor<64x64x!tt.ptr<bf16>>
        %117 = tt.expand_dims %111 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
        %118 = tt.splat %10 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
        %119 = tt.addptr %118, %105 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
        %120 = tt.broadcast %117 : tensor<64x1xi1> -> tensor<64x64xi1>
        %121 = tt.load %119, %120, %cst_2 : tensor<64x64x!tt.ptr<bf16>>
        %122 = arith.select %112, %cst_0, %cst_1 : tensor<1x64xi1>, tensor<1x64xf32>
        %123 = tt.broadcast %122 : tensor<1x64xf32> -> tensor<16x64xf32>
        %124 = tt.dot %78, %116, %cst_3 {triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
        %125 = arith.addf %124, %123 {triton_cv12.add_from_dot} : tensor<16x64xf32>
        %126 = "tt.reduce"(%125) <{axis = 1 : i32}> ({
        ^bb0(%arg34: f32, %arg35: f32):
          %146 = arith.maxnumf %arg34, %arg35 : f32
          tt.reduce.return %146 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %127 = arith.maxnumf %87#1, %126 : tensor<16xf32>
        %128 = tt.expand_dims %127 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %129 = tt.broadcast %128 : tensor<16x1xf32> -> tensor<16x64xf32>
        %130 = arith.subf %125, %129 : tensor<16x64xf32>
        %131 = math.exp2 %130 : tensor<16x64xf32>
        %132 = "tt.reduce"(%131) <{axis = 1 : i32}> ({
        ^bb0(%arg34: f32, %arg35: f32):
          %146 = arith.addf %arg34, %arg35 : f32
          tt.reduce.return %146 : f32
        }) : (tensor<16x64xf32>) -> tensor<16xf32>
        %133 = arith.subf %87#1, %127 : tensor<16xf32>
        %134 = math.exp2 %133 : tensor<16xf32>
        %135 = arith.mulf %87#2, %134 : tensor<16xf32>
        %136 = arith.addf %135, %132 : tensor<16xf32>
        %137 = tt.expand_dims %134 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %138 = tt.broadcast %137 : tensor<16x1xf32> -> tensor<16x64xf32>
        %139 = arith.mulf %87#0, %138 : tensor<16x64xf32>
        %140 = arith.truncf %131 : tensor<16x64xf32> to tensor<16x64xbf16>
        %141 = tt.dot %140, %121, %cst_3 {triton_cv12.normalized_dot} : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
        %142 = arith.extf %84 : tensor<1x64xbf16> to tensor<1x64xf32>
        %143 = tt.broadcast %142 : tensor<1x64xf32> -> tensor<16x64xf32>
        %144 = arith.mulf %141, %143 : tensor<16x64xf32>
        %145 = arith.addf %139, %144 : tensor<16x64xf32>
        scf.yield %127, %136, %145 : tensor<16xf32>, tensor<16xf32>, tensor<16x64xf32>
      } else {
        scf.yield %87#1, %87#2, %87#0 : tensor<16xf32>, tensor<16xf32>, tensor<16x64xf32>
      }
      scf.yield %88#2, %88#0, %88#1 : tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
    } {tt.num_stages = 16 : i32}
    %55 = tt.expand_dims %54#2 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %56 = tt.broadcast %55 : tensor<16x1xf32> -> tensor<16x64xf32>
    %57 = arith.divf %54#0, %56 : tensor<16x64xf32>
    %58 = arith.truncf %57 : tensor<16x64xf32> to tensor<16x64xbf16>
    %59 = tt.splat %arg27 : i32 -> tensor<16x1xi32>
    %60 = arith.muli %20, %59 : tensor<16x1xi32>
    %61 = tt.broadcast %60 : tensor<16x1xi32> -> tensor<16x64xi32>
    %62 = arith.addi %61, %25 : tensor<16x64xi32>
    %63 = tt.splat %14 : !tt.ptr<bf16> -> tensor<16x64x!tt.ptr<bf16>>
    %64 = tt.addptr %63, %62 : tensor<16x64x!tt.ptr<bf16>>, tensor<16x64xi32>
    tt.store %64, %58 : tensor<16x64x!tt.ptr<bf16>>
    %65 = math.log %54#2 : tensor<16xf32>
    %66 = arith.addf %54#1, %65 : tensor<16xf32>
    %67 = tt.splat %arg29 : i32 -> tensor<16xi32>
    %68 = arith.muli %19, %67 : tensor<16xi32>
    %69 = tt.splat %17 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %70 = tt.addptr %69, %68 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %70, %66 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/matmul_fp8/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_kernel_matmul_fp8_block_fastacc(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<32> : tensor<32x64xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<32> : tensor<64x32xi32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     tt.assert %true, "" : i1
// CHECK-NEXT:     tt.assert %true, "" : i1
// CHECK-NEXT:     tt.assert %true, "" : i1
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = arith.addi %arg3, %c63_i32 : i32
// CHECK-NEXT:     %3 = arith.divsi %2, %c64_i32 : i32
// CHECK-NEXT:     %4 = arith.addi %arg4, %c63_i32 : i32
// CHECK-NEXT:     %5 = arith.divsi %4, %c64_i32 : i32
// CHECK-NEXT:     %6 = arith.muli %5, %c8_i32 : i32
// CHECK-NEXT:     %7 = arith.divsi %0, %6 : i32
// CHECK-NEXT:     %8 = arith.muli %7, %c8_i32 : i32
// CHECK-NEXT:     %9 = arith.subi %3, %8 : i32
// CHECK-NEXT:     %10 = arith.minsi %9, %c8_i32 : i32
// CHECK-NEXT:     %11 = arith.remsi %0, %10 : i32
// CHECK-NEXT:     %12 = arith.addi %8, %11 : i32
// CHECK-NEXT:     %13 = arith.remsi %0, %6 : i32
// CHECK-NEXT:     %14 = arith.divsi %13, %10 : i32
// CHECK-NEXT:     %15 = arith.muli %12, %c64_i32 : i32
// CHECK-NEXT:     %16 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %17 = tt.splat %15 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %18 = arith.addi %17, %16 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %19 = arith.muli %14, %c64_i32 : i32
// CHECK-NEXT:     %20 = tt.splat %19 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %21 = arith.addi %20, %16 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %22 = tt.splat %arg3 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %23 = arith.remsi %18, %22 {MetaUse, tt.contiguity = dense<64> : tensor<1xi32>, tt.divisibility = dense<64> : tensor<1xi32>} : tensor<64xi32>
// CHECK-NEXT:     %24 = tt.splat %arg4 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %25 = arith.remsi %21, %24 {MetaUse, tt.contiguity = dense<64> : tensor<1xi32>, tt.divisibility = dense<64> : tensor<1xi32>} : tensor<64xi32>
// CHECK-NEXT:     %26 = arith.muli %1, %c32_i32 : i32
// CHECK-NEXT:     %27 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %28 = tt.splat %26 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:     %29 = arith.addi %28, %27 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:     %30 = tt.expand_dims %23 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %31 = tt.splat %arg11 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %32 = arith.muli %30, %31 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %33 = tt.expand_dims %29 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %34 = tt.broadcast %32 {MetaUse} : tensor<64x1xi32> -> tensor<64x32xi32>
// CHECK-NEXT:     %35 = tt.broadcast %33 {MetaUse} : tensor<1x32xi32> -> tensor<64x32xi32>
// CHECK-NEXT:     %36 = arith.addi %34, %35 {MetaUse} : tensor<64x32xi32>
// CHECK-NEXT:     %37 = tt.splat %arg0 {MetaUse} : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>>
// CHECK-NEXT:     %38 = tt.addptr %37, %36 {MetaUse} : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
// CHECK-NEXT:     %39 = tt.expand_dims %29 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %40 = tt.expand_dims %25 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %41 = tt.splat %arg12 {MetaUse} : i32 -> tensor<1x64xi32>
// CHECK-NEXT:     %42 = arith.muli %40, %41 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:     %43 = tt.broadcast %39 {MetaUse} : tensor<32x1xi32> -> tensor<32x64xi32>
// CHECK-NEXT:     %44 = tt.broadcast %42 {MetaUse} : tensor<1x64xi32> -> tensor<32x64xi32>
// CHECK-NEXT:     %45 = arith.addi %43, %44 {MetaUse} : tensor<32x64xi32>
// CHECK-NEXT:     %46 = tt.splat %arg1 {MetaUse} : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
// CHECK-NEXT:     %47 = tt.addptr %46, %45 {MetaUse} : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
// CHECK-NEXT:     %48 = arith.divsi %15, %c256_i32 : i32
// CHECK-NEXT:     %49 = arith.divsi %19, %c256_i32 : i32
// CHECK-NEXT:     %50 = arith.addi %arg5, %c31_i32 : i32
// CHECK-NEXT:     %51 = arith.divsi %50, %c32_i32 : i32
// CHECK-NEXT:     %52:3 = scf.for %arg16 = %c0_i32 to %51 step %c1_i32 iter_args(%arg17 = %cst, %arg18 = %38, %arg19 = %47) -> (tensor<64x64xf32>, tensor<64x32x!tt.ptr<f32>>, tensor<32x64x!tt.ptr<f32>>)  : i32 {
// CHECK-NEXT:       %69 = arith.muli %arg16, %c32_i32 : i32
// CHECK-NEXT:       %70 = arith.subi %arg5, %69 : i32
// CHECK-NEXT:       %71 = tt.load %arg18 {DataUse} : tensor<64x32x!tt.ptr<f32>>
// CHECK-NEXT:       %72 = tt.load %arg19 {DataUse} : tensor<32x64x!tt.ptr<f32>>
// CHECK-NEXT:       %73 = tt.dot %71, %72, %cst, inputPrecision = hf32 {DataUse, triton_cv12.normalized_dot} : tensor<64x32xf32> * tensor<32x64xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %74 = arith.addf %73, %arg17 {DataUse, triton_cv12.add_from_dot} : tensor<64x64xf32>
// CHECK-NEXT:       %75 = tt.addptr %arg18, %cst_1 {MetaUse} : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
// CHECK-NEXT:       %76 = tt.addptr %arg19, %cst_0 {MetaUse} : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
// CHECK-NEXT:       %77 = arith.addi %arg16, %1 : i32
// CHECK-NEXT:       %78 = arith.addi %77, %c1_i32 : i32
// CHECK-NEXT:       %79 = arith.remsi %78, %c8_i32 : i32
// CHECK-NEXT:       %80 = arith.cmpi eq, %79, %c0_i32 : i32
// CHECK-NEXT:       %81 = arith.cmpi slt, %70, %c32_i32 : i32
// CHECK-NEXT:       %82 = arith.ori %80, %81 : i1
// CHECK-NEXT:       %83 = scf.if %82 -> (tensor<64x64xf32>) {
// CHECK-NEXT:         %84 = arith.divsi %77, %c8_i32 : i32
// CHECK-NEXT:         %85 = arith.addi %84, %c1_i32 : i32
// CHECK-NEXT:         %86 = arith.muli %48, %arg14 : i32
// CHECK-NEXT:         %87 = tt.addptr %arg9, %86 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %88 = tt.addptr %87, %84 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %89 = tt.load %88 : !tt.ptr<f32>
// CHECK-NEXT:         %90 = arith.muli %49, %arg15 : i32
// CHECK-NEXT:         %91 = tt.addptr %arg10, %90 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %92 = tt.addptr %91, %84 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %93 = tt.load %92 : !tt.ptr<f32>
// CHECK-NEXT:         %94 = arith.mulf %89, %93 : f32
// CHECK-NEXT:         %95 = arith.addi %arg16, %c1_i32 : i32
// CHECK-NEXT:         %96 = arith.cmpi eq, %95, %51 : i32
// CHECK-NEXT:         %97 = scf.if %96 -> (f32) {
// CHECK-NEXT:           scf.yield %94 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %100 = tt.addptr %87, %85 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %101 = tt.load %100 : !tt.ptr<f32>
// CHECK-NEXT:           %102 = tt.addptr %91, %85 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %103 = tt.load %102 : !tt.ptr<f32>
// CHECK-NEXT:           %104 = arith.mulf %101, %103 : f32
// CHECK-NEXT:           %105 = arith.divf %94, %104 : f32
// CHECK-NEXT:           scf.yield %105 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         %98 = tt.splat %97 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:         %99 = arith.mulf %74, %98 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         scf.yield %99 : tensor<64x64xf32>
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %74 : tensor<64x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       scf.yield %83, %75, %76 : tensor<64x64xf32>, tensor<64x32x!tt.ptr<f32>>, tensor<32x64x!tt.ptr<f32>>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %53 = tt.expand_dims %18 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %54 = tt.splat %arg13 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %55 = arith.muli %53, %54 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %56 = tt.expand_dims %21 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %57 = tt.broadcast %55 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %58 = tt.broadcast %56 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %59 = arith.addi %57, %58 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:     %60 = tt.splat %arg2 {MetaUse} : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
// CHECK-NEXT:     %61 = tt.addptr %60, %59 {MetaUse} : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
// CHECK-NEXT:     %62 = arith.cmpi slt, %18, %22 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %63 = tt.expand_dims %62 {MetaUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %64 = arith.cmpi slt, %21, %24 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %65 = tt.expand_dims %64 {MetaUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %66 = tt.broadcast %63 {MetaUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %67 = tt.broadcast %65 {MetaUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %68 = arith.andi %66, %67 {MetaUse} : tensor<64x64xi1>
// CHECK-NEXT:     tt.store %61, %52#0, %68 : tensor<64x64x!tt.ptr<f32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_kernel_matmul_fp8_block_fastacc(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant dense<32> : tensor<32x64xi32>
    %cst_1 = arith.constant dense<32> : tensor<64x32xi32>
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %true = arith.constant true
    tt.assert %true, "" : i1
    tt.assert %true, "" : i1
    tt.assert %true, "" : i1
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.addi %arg3, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.addi %arg4, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %0, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %0, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c64_i32 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %17 = tt.splat %15 : i32 -> tensor<64xi32>
    %18 = arith.addi %17, %16 : tensor<64xi32>
    %19 = arith.muli %14, %c64_i32 : i32
    %20 = tt.splat %19 : i32 -> tensor<64xi32>
    %21 = arith.addi %20, %16 : tensor<64xi32>
    %22 = tt.splat %arg3 : i32 -> tensor<64xi32>
    %23 = arith.remsi %18, %22 {tt.contiguity = dense<64> : tensor<1xi32>, tt.divisibility = dense<64> : tensor<1xi32>} : tensor<64xi32>
    %24 = tt.splat %arg4 : i32 -> tensor<64xi32>
    %25 = arith.remsi %21, %24 {tt.contiguity = dense<64> : tensor<1xi32>, tt.divisibility = dense<64> : tensor<1xi32>} : tensor<64xi32>
    %26 = arith.muli %1, %c32_i32 : i32
    %27 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %28 = tt.splat %26 : i32 -> tensor<32xi32>
    %29 = arith.addi %28, %27 : tensor<32xi32>
    %30 = tt.expand_dims %23 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %31 = tt.splat %arg11 : i32 -> tensor<64x1xi32>
    %32 = arith.muli %30, %31 : tensor<64x1xi32>
    %33 = tt.expand_dims %29 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %34 = tt.broadcast %32 : tensor<64x1xi32> -> tensor<64x32xi32>
    %35 = tt.broadcast %33 : tensor<1x32xi32> -> tensor<64x32xi32>
    %36 = arith.addi %34, %35 : tensor<64x32xi32>
    %37 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>>
    %38 = tt.addptr %37, %36 : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
    %39 = tt.expand_dims %29 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %40 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %41 = tt.splat %arg12 : i32 -> tensor<1x64xi32>
    %42 = arith.muli %40, %41 : tensor<1x64xi32>
    %43 = tt.broadcast %39 : tensor<32x1xi32> -> tensor<32x64xi32>
    %44 = tt.broadcast %42 : tensor<1x64xi32> -> tensor<32x64xi32>
    %45 = arith.addi %43, %44 : tensor<32x64xi32>
    %46 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>>
    %47 = tt.addptr %46, %45 : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
    %48 = arith.divsi %15, %c256_i32 : i32
    %49 = arith.divsi %19, %c256_i32 : i32
    %50 = arith.addi %arg5, %c31_i32 : i32
    %51 = arith.divsi %50, %c32_i32 : i32
    %52:3 = scf.for %arg16 = %c0_i32 to %51 step %c1_i32 iter_args(%arg17 = %cst, %arg18 = %38, %arg19 = %47) -> (tensor<64x64xf32>, tensor<64x32x!tt.ptr<f32>>, tensor<32x64x!tt.ptr<f32>>)  : i32 {
      %69 = arith.muli %arg16, %c32_i32 : i32
      %70 = arith.subi %arg5, %69 : i32
      %71 = tt.load %arg18 : tensor<64x32x!tt.ptr<f32>>
      %72 = tt.load %arg19 : tensor<32x64x!tt.ptr<f32>>
      %73 = tt.dot %71, %72, %cst, inputPrecision = hf32 {triton_cv12.normalized_dot} : tensor<64x32xf32> * tensor<32x64xf32> -> tensor<64x64xf32>
      %74 = arith.addf %73, %arg17 {triton_cv12.add_from_dot} : tensor<64x64xf32>
      %75 = tt.addptr %arg18, %cst_1 : tensor<64x32x!tt.ptr<f32>>, tensor<64x32xi32>
      %76 = tt.addptr %arg19, %cst_0 : tensor<32x64x!tt.ptr<f32>>, tensor<32x64xi32>
      %77 = arith.addi %arg16, %1 : i32
      %78 = arith.addi %77, %c1_i32 : i32
      %79 = arith.remsi %78, %c8_i32 : i32
      %80 = arith.cmpi eq, %79, %c0_i32 : i32
      %81 = arith.cmpi slt, %70, %c32_i32 : i32
      %82 = arith.ori %80, %81 : i1
      %83 = scf.if %82 -> (tensor<64x64xf32>) {
        %84 = arith.divsi %77, %c8_i32 : i32
        %85 = arith.addi %84, %c1_i32 : i32
        %86 = arith.muli %48, %arg14 : i32
        %87 = tt.addptr %arg9, %86 : !tt.ptr<f32>, i32
        %88 = tt.addptr %87, %84 : !tt.ptr<f32>, i32
        %89 = tt.load %88 : !tt.ptr<f32>
        %90 = arith.muli %49, %arg15 : i32
        %91 = tt.addptr %arg10, %90 : !tt.ptr<f32>, i32
        %92 = tt.addptr %91, %84 : !tt.ptr<f32>, i32
        %93 = tt.load %92 : !tt.ptr<f32>
        %94 = arith.mulf %89, %93 : f32
        %95 = arith.addi %arg16, %c1_i32 : i32
        %96 = arith.cmpi eq, %95, %51 : i32
        %97 = scf.if %96 -> (f32) {
          scf.yield %94 : f32
        } else {
          %100 = tt.addptr %87, %85 : !tt.ptr<f32>, i32
          %101 = tt.load %100 : !tt.ptr<f32>
          %102 = tt.addptr %91, %85 : !tt.ptr<f32>, i32
          %103 = tt.load %102 : !tt.ptr<f32>
          %104 = arith.mulf %101, %103 : f32
          %105 = arith.divf %94, %104 : f32
          scf.yield %105 : f32
        }
        %98 = tt.splat %97 : f32 -> tensor<64x64xf32>
        %99 = arith.mulf %74, %98 : tensor<64x64xf32>
        scf.yield %99 : tensor<64x64xf32>
      } else {
        scf.yield %74 : tensor<64x64xf32>
      }
      scf.yield %83, %75, %76 : tensor<64x64xf32>, tensor<64x32x!tt.ptr<f32>>, tensor<32x64x!tt.ptr<f32>>
    }
    %53 = tt.expand_dims %18 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %54 = tt.splat %arg13 : i32 -> tensor<64x1xi32>
    %55 = arith.muli %53, %54 : tensor<64x1xi32>
    %56 = tt.expand_dims %21 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %57 = tt.broadcast %55 : tensor<64x1xi32> -> tensor<64x64xi32>
    %58 = tt.broadcast %56 : tensor<1x64xi32> -> tensor<64x64xi32>
    %59 = arith.addi %57, %58 : tensor<64x64xi32>
    %60 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %61 = tt.addptr %60, %59 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %62 = arith.cmpi slt, %18, %22 : tensor<64xi32>
    %63 = tt.expand_dims %62 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %64 = arith.cmpi slt, %21, %24 : tensor<64xi32>
    %65 = tt.expand_dims %64 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %66 = tt.broadcast %63 : tensor<64x1xi1> -> tensor<64x64xi1>
    %67 = tt.broadcast %65 : tensor<1x64xi1> -> tensor<64x64xi1>
    %68 = arith.andi %66, %67 : tensor<64x64xi1>
    tt.store %61, %52#0, %68 : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/prepare_wy/norm_dot.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module {
// CHECK-NEXT:   module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:     tt.func public @prepare_wy_repr_bwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32) attributes {noinline = false} {
// CHECK-NEXT:       %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:       %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:       %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:       %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:       %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:       %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:       %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:       %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:       %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:       %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:       %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:       %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:       %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:       %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:       %0 = tt.get_program_id x : i32
// CHECK-NEXT:       %1 = tt.get_program_id y : i32
// CHECK-NEXT:       %2 = arith.divsi %1, %c32_i32 : i32
// CHECK-NEXT:       %3 = arith.remsi %1, %c32_i32 : i32
// CHECK-NEXT:       %4 = arith.muli %2, %arg11 : i32
// CHECK-NEXT:       %5 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:       %6 = arith.addi %5, %3 : i32
// CHECK-NEXT:       %7 = tt.addptr %arg2, %6 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %8 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:       %9 = arith.extsi %arg11 : i32 to i64
// CHECK-NEXT:       %10 = tt.make_tensor_ptr %7, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:       %11 = tt.addptr %arg9, %6 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %12 = tt.make_tensor_ptr %11, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:       %13 = arith.muli %6, %c64_i32 : i32
// CHECK-NEXT:       %14 = tt.addptr %arg4, %13 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %15 = tt.make_tensor_ptr %14, [%c64_i64, %9], [%c1_i64, %c2048_i64], [%c0_i32, %8] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %16 = tt.load %10 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:       %17 = tt.load %15 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %18 = tt.addptr %arg3, %6 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %19 = tt.make_tensor_ptr %18, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:       %20 = tt.load %19 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:       %21 = math.exp %20 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:       %22 = arith.muli %6, %c128_i32 : i32
// CHECK-NEXT:       %23 = tt.addptr %arg0, %22 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %24 = tt.addptr %arg7, %22 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %25 = tt.addptr %arg5, %22 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %26 = arith.mulf %16, %21 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:       %27 = tt.expand_dims %26 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:       %28 = tt.broadcast %27 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %29 = arith.extf %27 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:       %30 = tt.broadcast %29 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %31 = tt.expand_dims %21 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:       %32 = arith.extf %31 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:       %33 = tt.broadcast %32 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %34:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_0, %arg14 = %cst, %arg15 = %cst) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
// CHECK-NEXT:         %98 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:         %99 = tt.make_tensor_ptr %23, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %100 = tt.make_tensor_ptr %24, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %101 = tt.make_tensor_ptr %25, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %102 = tt.load %99 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %103 = arith.mulf %102, %28 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:         %104 = tt.load %101 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %105 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %106 = tt.dot %104, %105, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %107 = tt.dot %17, %104, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %108 = arith.mulf %107, %30 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %109 = arith.extf %102 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:         %110 = arith.mulf %107, %109 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %111 = arith.mulf %110, %33 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %112 = "tt.reduce"(%111) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %119 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %119 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %113 = arith.addf %arg14, %112 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %114 = arith.extf %103 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:         %115 = arith.mulf %107, %114 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %116 = "tt.reduce"(%115) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %119 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %119 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %117 = arith.addf %arg15, %116 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %118 = arith.truncf %108 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:         tt.store %100, %118 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         scf.yield %106, %113, %117 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %35 = arith.muli %6, %c128_i32 : i32
// CHECK-NEXT:       %36 = tt.addptr %arg1, %35 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %37 = tt.addptr %arg8, %35 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %38 = tt.addptr %arg6, %35 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %39 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:       %40 = tt.broadcast %39 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %41 = arith.extf %39 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:       %42 = tt.broadcast %41 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %43:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %34#0, %arg14 = %34#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
// CHECK-NEXT:         %98 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:         %99 = tt.make_tensor_ptr %36, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %100 = tt.make_tensor_ptr %37, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %101 = tt.make_tensor_ptr %38, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %102 = tt.load %99 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %103 = arith.mulf %102, %40 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:         %104 = tt.load %101 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %105 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %106 = tt.dot %104, %105, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %107 = tt.dot %17, %104, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %108 = arith.mulf %107, %42 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %109 = arith.extf %102 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:         %110 = arith.mulf %107, %109 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %111 = "tt.reduce"(%110) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %114 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %114 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %112 = arith.addf %arg14, %111 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %113 = arith.truncf %108 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:         tt.store %100, %113 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         scf.yield %106, %112 : tensor<64x64xf32>, tensor<64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %44 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:       %45 = tt.splat %8 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %46 = arith.addi %45, %44 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %47 = tt.splat %arg11 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %48 = arith.cmpi slt, %46, %47 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %49 = tt.expand_dims %46 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:       %50 = tt.expand_dims %46 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:       %51 = tt.broadcast %49 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:       %52 = tt.broadcast %50 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:       %53 = arith.cmpi sgt, %51, %52 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:       %54 = tt.expand_dims %48 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:       %55 = tt.expand_dims %48 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:       %56 = tt.broadcast %54 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:       %57 = tt.broadcast %55 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:       %58 = arith.andi %56, %57 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:       %59 = arith.andi %53, %58 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:       %60 = arith.select %59, %43#0, %cst_0 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:       %61 = arith.truncf %60 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %62 = tt.dot %61, %17, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %63 = arith.truncf %62 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %64 = tt.dot %17, %63, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %65 = tt.expand_dims %20 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:       %66 = tt.expand_dims %20 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:       %67 = tt.broadcast %65 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %68 = tt.broadcast %66 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %69 = arith.subf %67, %68 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:       %70 = arith.extf %69 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %71 = math.exp %70 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %72 = arith.mulf %64, %71 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %73 = arith.subf %cst_0, %72 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %74 = arith.select %59, %73, %cst_0 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:       %75 = arith.truncf %74 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       gpu.barrier
// CHECK-NEXT:       %76 = arith.muli %6, %c128_i32 : i32
// CHECK-NEXT:       %77 = tt.addptr %arg0, %76 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %78 = tt.addptr %arg7, %76 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %79 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:       %80 = tt.broadcast %79 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %81 = arith.extf %79 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:       %82 = tt.broadcast %81 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %83:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_0, %arg14 = %43#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
// CHECK-NEXT:         %98 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:         %99 = tt.make_tensor_ptr %77, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %100 = tt.make_tensor_ptr %78, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %101 = tt.load %99 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %102 = tt.trans %101 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %103 = arith.mulf %101, %80 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:         %104 = tt.dot %101, %102, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %105 = tt.dot %75, %101, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %106 = arith.extf %101 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:         %107 = arith.mulf %105, %106 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %108 = "tt.reduce"(%107) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %119 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %119 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %109 = arith.addf %arg14, %108 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %110 = arith.mulf %105, %82 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %111 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %112 = tt.dot %111, %75, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %113 = tt.trans %112 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK-NEXT:         %114 = arith.addf %110, %113 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %115 = tt.load %100 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %116 = arith.extf %115 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:         %117 = arith.addf %114, %116 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %118 = arith.truncf %117 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:         tt.store %100, %118 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         scf.yield %104, %109 : tensor<64x64xf32>, tensor<64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %84 = arith.truncf %83#1 {DataUse} : tensor<64xf32> to tensor<64xbf16>
// CHECK-NEXT:       tt.store %12, %84 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:       %85 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:       %86 = arith.extf %85 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:       %87 = tt.broadcast %86 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %88 = arith.mulf %83#0, %87 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %89 = arith.extf %75 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %90 = arith.mulf %89, %88 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %91 = tt.addptr %arg10, %6 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %92 = tt.make_tensor_ptr %91, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:       %93 = "tt.reduce"(%90) <{axis = 1 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:         %98 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:         tt.reduce.return %98 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:       %94 = "tt.reduce"(%90) <{axis = 0 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:         %98 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:         tt.reduce.return %98 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:       %95 = arith.subf %93, %94 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %96 = arith.addf %34#2, %95 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %97 = arith.truncf %96 {DataUse} : tensor<64xf32> to tensor<64xbf16>
// CHECK-NEXT:       tt.store %92, %97 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:       tt.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.disable_auto_tile_and_bind_subblock} {
// CHECK-NEXT:     tt.func public @prepare_wy_repr_bwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32) attributes {noinline = false} {
// CHECK-NEXT:       %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x64xf32>
// CHECK-NEXT:       %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:       %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:       %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:       %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:       %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:       %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:       %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:       %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:       %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:       %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:       %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:       %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:       %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:       %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:       %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:       %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:       %2 = affine.apply #map()[%1]
// CHECK-NEXT:       %3 = arith.index_cast %2 : index to i32
// CHECK-NEXT:       %4 = tt.get_program_id x : i32
// CHECK-NEXT:       %5 = tt.get_program_id y : i32
// CHECK-NEXT:       %6 = arith.divsi %5, %c32_i32 : i32
// CHECK-NEXT:       %7 = arith.remsi %5, %c32_i32 : i32
// CHECK-NEXT:       %8 = arith.muli %6, %arg11 : i32
// CHECK-NEXT:       %9 = arith.muli %8, %c32_i32 : i32
// CHECK-NEXT:       %10 = arith.addi %9, %7 : i32
// CHECK-NEXT:       %11 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %12 = arith.muli %4, %c64_i32 : i32
// CHECK-NEXT:       %13 = arith.extsi %arg11 : i32 to i64
// CHECK-NEXT:       %14 = arith.addi %12, %3 : i32
// CHECK-NEXT:       %15 = tt.make_tensor_ptr %11, [%13], [%c32_i64], [%14] {order = array<i32: 0>} : <tensor<32xbf16>>
// CHECK-NEXT:       %16 = tt.addptr %arg9, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %17 = tt.make_tensor_ptr %16, [%13], [%c32_i64], [%14] {order = array<i32: 0>, tiled_op} : <tensor<32xbf16>>
// CHECK-NEXT:       %18 = arith.muli %10, %c64_i32 : i32
// CHECK-NEXT:       %19 = tt.addptr %arg4, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %20 = tt.make_tensor_ptr %19, [%c64_i64, %13], [%c1_i64, %c2048_i64], [%c0_i32, %12] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %21 = tt.load %15 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xbf16>>
// CHECK-NEXT:       %22 = tt.load %20 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %23, [%13], [%c32_i64], [%12] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:       %25 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %25[%2] [32] [1] {DataUse, should_kept_slice} : tensor<64xbf16> to tensor<32xbf16>
// CHECK-NEXT:       %26 = math.exp %extracted_slice {DataUse} : tensor<32xbf16>
// CHECK-NEXT:       %27 = arith.muli %10, %c128_i32 : i32
// CHECK-NEXT:       %28 = tt.addptr %arg0, %27 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %29 = tt.addptr %arg7, %27 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %30 = tt.addptr %arg5, %27 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %31 = arith.mulf %21, %26 {DataUse} : tensor<32xbf16>
// CHECK-NEXT:       %32 = tt.expand_dims %31 {DataUse, axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
// CHECK-NEXT:       %33 = tt.broadcast %32 {DataUse} : tensor<32x1xbf16> -> tensor<32x64xbf16>
// CHECK-NEXT:       %34 = arith.extf %32 {DataUse} : tensor<32x1xbf16> to tensor<32x1xf32>
// CHECK-NEXT:       %35 = tt.broadcast %34 {DataUse} : tensor<32x1xf32> -> tensor<32x64xf32>
// CHECK-NEXT:       %36 = tt.expand_dims %26 {DataUse, axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
// CHECK-NEXT:       %37 = arith.extf %36 {DataUse} : tensor<32x1xbf16> to tensor<32x1xf32>
// CHECK-NEXT:       %38 = tt.broadcast %37 {DataUse} : tensor<32x1xf32> -> tensor<32x64xf32>
// CHECK-NEXT:       %39:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_1, %arg14 = %cst_0, %arg15 = %cst_0) -> (tensor<64x64xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
// CHECK-NEXT:         %107 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:         %108 = tt.make_tensor_ptr %28, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
// CHECK-NEXT:         %109 = tt.make_tensor_ptr %29, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:         %110 = tt.make_tensor_ptr %30, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %107] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %111 = tt.load %108 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:         %112 = arith.mulf %111, %33 {DataUse} : tensor<32x64xbf16>
// CHECK-NEXT:         %113 = tt.load %110 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %114 = memref_ext.alloc_workspace() : memref<64x64xbf16>
// CHECK-NEXT:         %subview_17 = memref.subview %114[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         bufferization.materialize_in_destination %112 in writable %subview_17 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
// CHECK-NEXT:         %alloc_18 = memref.alloc() : memref<64x64xbf16>
// CHECK-NEXT:         memref.copy %114, %alloc_18 : memref<64x64xbf16> to memref<64x64xbf16>
// CHECK-NEXT:         %115 = bufferization.to_tensor %alloc_18 restrict writable {DataUse} : memref<64x64xbf16>
// CHECK-NEXT:         %116 = tt.trans %115 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %117 = tt.dot %113, %116, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %118 = tt.dot %22, %113, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %119 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:         bufferization.materialize_in_destination %118 in writable %119 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:         %subview_19 = memref.subview %119[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         %alloc_20 = memref.alloc() : memref<32x64xf32>
// CHECK-NEXT:         memref.copy %subview_19, %alloc_20 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
// CHECK-NEXT:         %120 = bufferization.to_tensor %alloc_20 restrict writable {DataUse} : memref<32x64xf32>
// CHECK-NEXT:         %121 = arith.mulf %120, %35 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %122 = arith.extf %111 {DataUse} : tensor<32x64xbf16> to tensor<32x64xf32>
// CHECK-NEXT:         %123 = arith.mulf %120, %122 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %124 = arith.mulf %123, %38 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %125 = "tt.reduce"(%124) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %132 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %132 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x64xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %126 = arith.addf %arg14, %125 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %127 = arith.extf %112 {DataUse} : tensor<32x64xbf16> to tensor<32x64xf32>
// CHECK-NEXT:         %128 = arith.mulf %120, %127 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %129 = "tt.reduce"(%128) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %132 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %132 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x64xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %130 = arith.addf %arg15, %129 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %131 = arith.truncf %121 {DataUse} : tensor<32x64xf32> to tensor<32x64xbf16>
// CHECK-NEXT:         tt.store %109, %131 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:         scf.yield %117, %126, %130 : tensor<64x64xf32>, tensor<32xf32>, tensor<32xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %40 = tt.addptr %arg1, %27 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %41 = tt.addptr %arg8, %27 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %42 = tt.addptr %arg6, %27 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %43 = tt.expand_dims %21 {DataUse, axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
// CHECK-NEXT:       %44 = tt.broadcast %43 {DataUse} : tensor<32x1xbf16> -> tensor<32x64xbf16>
// CHECK-NEXT:       %45 = arith.extf %43 {DataUse} : tensor<32x1xbf16> to tensor<32x1xf32>
// CHECK-NEXT:       %46 = tt.broadcast %45 {DataUse} : tensor<32x1xf32> -> tensor<32x64xf32>
// CHECK-NEXT:       %47:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %39#0, %arg14 = %39#1) -> (tensor<64x64xf32>, tensor<32xf32>)  : i32 {
// CHECK-NEXT:         %107 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:         %108 = tt.make_tensor_ptr %40, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
// CHECK-NEXT:         %109 = tt.make_tensor_ptr %41, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:         %110 = tt.make_tensor_ptr %42, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %107] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %111 = tt.load %108 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:         %112 = arith.mulf %111, %44 {DataUse} : tensor<32x64xbf16>
// CHECK-NEXT:         %113 = tt.load %110 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %114 = memref_ext.alloc_workspace() : memref<64x64xbf16>
// CHECK-NEXT:         %subview_17 = memref.subview %114[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         bufferization.materialize_in_destination %112 in writable %subview_17 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
// CHECK-NEXT:         %alloc_18 = memref.alloc() : memref<64x64xbf16>
// CHECK-NEXT:         memref.copy %114, %alloc_18 : memref<64x64xbf16> to memref<64x64xbf16>
// CHECK-NEXT:         %115 = bufferization.to_tensor %alloc_18 restrict writable {DataUse} : memref<64x64xbf16>
// CHECK-NEXT:         %116 = tt.trans %115 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %117 = tt.dot %113, %116, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %118 = tt.dot %22, %113, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %119 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:         bufferization.materialize_in_destination %118 in writable %119 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:         %subview_19 = memref.subview %119[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         %alloc_20 = memref.alloc() : memref<32x64xf32>
// CHECK-NEXT:         memref.copy %subview_19, %alloc_20 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
// CHECK-NEXT:         %120 = bufferization.to_tensor %alloc_20 restrict writable {DataUse} : memref<32x64xf32>
// CHECK-NEXT:         %121 = arith.mulf %120, %46 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %122 = arith.extf %111 {DataUse} : tensor<32x64xbf16> to tensor<32x64xf32>
// CHECK-NEXT:         %123 = arith.mulf %120, %122 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %124 = "tt.reduce"(%123) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %127 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %127 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x64xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %125 = arith.addf %arg14, %124 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %126 = arith.truncf %121 {DataUse} : tensor<32x64xf32> to tensor<32x64xbf16>
// CHECK-NEXT:         tt.store %109, %126 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:         scf.yield %117, %125 : tensor<64x64xf32>, tensor<32xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %48 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:       %49 = tt.splat %12 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %50 = arith.addi %49, %48 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %51 = tt.splat %arg11 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %52 = arith.cmpi slt, %50, %51 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %extracted_slice_2 = tensor.extract_slice %50[%2] [32] [1] {DataUse, should_kept_slice} : tensor<64xi32> to tensor<32xi32>
// CHECK-NEXT:       %53 = tt.expand_dims %extracted_slice_2 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:       %54 = tt.expand_dims %50 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:       %55 = tt.broadcast %53 {DataUse} : tensor<32x1xi32> -> tensor<32x64xi32>
// CHECK-NEXT:       %56 = tt.broadcast %54 {DataUse} : tensor<1x64xi32> -> tensor<32x64xi32>
// CHECK-NEXT:       %57 = arith.cmpi sgt, %55, %56 {DataUse} : tensor<32x64xi32>
// CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %52[%2] [32] [1] {DataUse, should_kept_slice} : tensor<64xi1> to tensor<32xi1>
// CHECK-NEXT:       %58 = tt.expand_dims %extracted_slice_3 {DataUse, axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
// CHECK-NEXT:       %59 = tt.expand_dims %52 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:       %60 = tt.broadcast %58 {DataUse} : tensor<32x1xi1> -> tensor<32x64xi1>
// CHECK-NEXT:       %61 = tt.broadcast %59 {DataUse} : tensor<1x64xi1> -> tensor<32x64xi1>
// CHECK-NEXT:       %62 = arith.andi %60, %61 {DataUse} : tensor<32x64xi1>
// CHECK-NEXT:       %63 = arith.andi %57, %62 {DataUse} : tensor<32x64xi1>
// CHECK-NEXT:       %64 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %47#0 in writable %64 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:       %subview = memref.subview %64[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       %alloc = memref.alloc() : memref<32x64xf32>
// CHECK-NEXT:       memref.copy %subview, %alloc : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
// CHECK-NEXT:       %65 = bufferization.to_tensor %alloc restrict writable {DataUse} : memref<32x64xf32>
// CHECK-NEXT:       %66 = arith.select %63, %65, %cst {DataUse} : tensor<32x64xi1>, tensor<32x64xf32>
// CHECK-NEXT:       %67 = arith.truncf %66 {DataUse} : tensor<32x64xf32> to tensor<32x64xbf16>
// CHECK-NEXT:       %68 = memref_ext.alloc_workspace() : memref<64x64xbf16>
// CHECK-NEXT:       %subview_4 = memref.subview %68[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %67 in writable %subview_4 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_5 = memref.alloc() : memref<64x64xbf16>
// CHECK-NEXT:       memref.copy %68, %alloc_5 : memref<64x64xbf16> to memref<64x64xbf16>
// CHECK-NEXT:       %69 = bufferization.to_tensor %alloc_5 restrict writable {DataUse} : memref<64x64xbf16>
// CHECK-NEXT:       %70 = tt.dot %69, %22, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %71 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %70 in writable %71 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:       %subview_6 = memref.subview %71[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       %alloc_7 = memref.alloc() : memref<32x64xf32>
// CHECK-NEXT:       memref.copy %subview_6, %alloc_7 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
// CHECK-NEXT:       %72 = bufferization.to_tensor %alloc_7 restrict writable {DataUse} : memref<32x64xf32>
// CHECK-NEXT:       %73 = arith.truncf %72 {DataUse} : tensor<32x64xf32> to tensor<32x64xbf16>
// CHECK-NEXT:       %74 = memref_ext.alloc_workspace() : memref<64x64xbf16>
// CHECK-NEXT:       %subview_8 = memref.subview %74[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %73 in writable %subview_8 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_9 = memref.alloc() : memref<64x64xbf16>
// CHECK-NEXT:       memref.copy %74, %alloc_9 : memref<64x64xbf16> to memref<64x64xbf16>
// CHECK-NEXT:       %75 = bufferization.to_tensor %alloc_9 restrict writable {DataUse} : memref<64x64xbf16>
// CHECK-NEXT:       %76 = tt.dot %22, %75, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %77 = tt.expand_dims %extracted_slice {DataUse, axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
// CHECK-NEXT:       %78 = tt.expand_dims %25 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:       %79 = tt.broadcast %77 {DataUse} : tensor<32x1xbf16> -> tensor<32x64xbf16>
// CHECK-NEXT:       %80 = tt.broadcast %78 {DataUse} : tensor<1x64xbf16> -> tensor<32x64xbf16>
// CHECK-NEXT:       %81 = arith.subf %79, %80 {DataUse} : tensor<32x64xbf16>
// CHECK-NEXT:       %82 = arith.extf %81 {DataUse} : tensor<32x64xbf16> to tensor<32x64xf32>
// CHECK-NEXT:       %83 = math.exp %82 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:       %84 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %76 in writable %84 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:       %subview_10 = memref.subview %84[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       %alloc_11 = memref.alloc() : memref<32x64xf32>
// CHECK-NEXT:       memref.copy %subview_10, %alloc_11 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
// CHECK-NEXT:       %85 = bufferization.to_tensor %alloc_11 restrict writable {DataUse} : memref<32x64xf32>
// CHECK-NEXT:       %86 = arith.mulf %85, %83 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:       %87 = arith.subf %cst, %86 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:       %88 = arith.select %63, %87, %cst {DataUse} : tensor<32x64xi1>, tensor<32x64xf32>
// CHECK-NEXT:       %89 = arith.truncf %88 {DataUse} : tensor<32x64xf32> to tensor<32x64xbf16>
// CHECK-NEXT:       gpu.barrier
// CHECK-NEXT:       %90:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_1, %arg14 = %47#1) -> (tensor<64x64xf32>, tensor<32xf32>)  : i32 {
// CHECK-NEXT:         %107 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:         %108 = tt.make_tensor_ptr %28, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %107] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:         %109 = tt.make_tensor_ptr %28, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
// CHECK-NEXT:         %110 = tt.make_tensor_ptr %29, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
// CHECK-NEXT:         %111 = tt.make_tensor_ptr %29, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:         %112 = tt.load %109 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:         %113 = tt.load %108 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:         %114 = tt.trans %113 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %115 = arith.mulf %112, %44 {DataUse} : tensor<32x64xbf16>
// CHECK-NEXT:         %116 = tt.dot %113, %114, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %117 = memref_ext.alloc_workspace() : memref<64x64xbf16>
// CHECK-NEXT:         %subview_17 = memref.subview %117[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         bufferization.materialize_in_destination %89 in writable %subview_17 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
// CHECK-NEXT:         %alloc_18 = memref.alloc() : memref<64x64xbf16>
// CHECK-NEXT:         memref.copy %117, %alloc_18 : memref<64x64xbf16> to memref<64x64xbf16>
// CHECK-NEXT:         %118 = bufferization.to_tensor %alloc_18 restrict writable {DataUse} : memref<64x64xbf16>
// CHECK-NEXT:         %119 = tt.dot %118, %113, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %120 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:         bufferization.materialize_in_destination %119 in writable %120 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:         %subview_19 = memref.subview %120[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         %alloc_20 = memref.alloc() : memref<32x64xf32>
// CHECK-NEXT:         memref.copy %subview_19, %alloc_20 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
// CHECK-NEXT:         %121 = bufferization.to_tensor %alloc_20 restrict writable {DataUse} : memref<32x64xf32>
// CHECK-NEXT:         %122 = arith.extf %112 {DataUse} : tensor<32x64xbf16> to tensor<32x64xf32>
// CHECK-NEXT:         %123 = arith.mulf %121, %122 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %124 = "tt.reduce"(%123) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %139 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %139 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x64xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %125 = arith.addf %arg14, %124 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %126 = arith.mulf %121, %46 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %127 = memref_ext.alloc_workspace() : memref<64x64xbf16>
// CHECK-NEXT:         %subview_21 = memref.subview %127[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         bufferization.materialize_in_destination %115 in writable %subview_21 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
// CHECK-NEXT:         %alloc_22 = memref.alloc() : memref<64x64xbf16>
// CHECK-NEXT:         memref.copy %127, %alloc_22 : memref<64x64xbf16> to memref<64x64xbf16>
// CHECK-NEXT:         %128 = bufferization.to_tensor %alloc_22 restrict writable {DataUse} : memref<64x64xbf16>
// CHECK-NEXT:         %129 = tt.trans %128 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:         %130 = tt.dot %129, %118, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %131 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:         bufferization.materialize_in_destination %130 in writable %131 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:         %subview_23 = memref.subview %131[0, %2] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:         %alloc_24 = memref.alloc() : memref<64x32xf32>
// CHECK-NEXT:         memref.copy %subview_23, %alloc_24 : memref<64x32xf32, strided<[64, 1], offset: ?>> to memref<64x32xf32>
// CHECK-NEXT:         %132 = bufferization.to_tensor %alloc_24 restrict writable {DataUse} : memref<64x32xf32>
// CHECK-NEXT:         %133 = tt.trans %132 {DataUse, order = array<i32: 1, 0>} : tensor<64x32xf32> -> tensor<32x64xf32>
// CHECK-NEXT:         %134 = arith.addf %126, %133 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %135 = tt.load %110 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:         %136 = arith.extf %135 {DataUse} : tensor<32x64xbf16> to tensor<32x64xf32>
// CHECK-NEXT:         %137 = arith.addf %134, %136 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:         %138 = arith.truncf %137 {DataUse} : tensor<32x64xf32> to tensor<32x64xbf16>
// CHECK-NEXT:         tt.store %111, %138 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:         scf.yield %116, %125 : tensor<64x64xf32>, tensor<32xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %91 = arith.truncf %90#1 {DataUse} : tensor<32xf32> to tensor<32xbf16>
// CHECK-NEXT:       tt.store %17, %91 {boundaryCheck = array<i32: 0>, tiled_op} : !tt.ptr<tensor<32xbf16>>
// CHECK-NEXT:       %92 = memref_ext.alloc_workspace() : memref<64x64xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %90#0 in writable %92 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
// CHECK-NEXT:       %subview_12 = memref.subview %92[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       %alloc_13 = memref.alloc() : memref<32x64xf32>
// CHECK-NEXT:       memref.copy %subview_12, %alloc_13 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
// CHECK-NEXT:       %93 = bufferization.to_tensor %alloc_13 restrict writable {DataUse} : memref<32x64xf32>
// CHECK-NEXT:       %94 = arith.mulf %93, %46 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:       %95 = arith.extf %89 {DataUse} : tensor<32x64xbf16> to tensor<32x64xf32>
// CHECK-NEXT:       %96 = arith.mulf %95, %94 {DataUse} : tensor<32x64xf32>
// CHECK-NEXT:       %97 = tt.addptr %arg10, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %98 = tt.make_tensor_ptr %97, [%13], [%c32_i64], [%14] {order = array<i32: 0>, tiled_op} : <tensor<32xbf16>>
// CHECK-NEXT:       %99 = "tt.reduce"(%96) <{axis = 1 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:         %107 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:         tt.reduce.return %107 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<32x64xf32>) -> tensor<32xf32>
// CHECK-NEXT:       %100 = "tt.reduce"(%96) <{axis = 0 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:         %107 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:         tt.reduce.return %107 : f32
// CHECK-NEXT:       }) {DataUse, tiled_op} : (tensor<32x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:       %101 = memref_ext.alloc_workspace() : memref<2x64xf32>
// CHECK-NEXT:       %subview_14 = memref.subview %101[%1, 0] [1, 64] [1, 1] : memref<2x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       %expanded = tensor.expand_shape %100 {{\[\[}}0, 1]] output_shape [1, 64] {DataUse} : tensor<64xf32> into tensor<1x64xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %expanded in writable %subview_14 : (tensor<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>>) -> ()
// CHECK-NEXT:       hivm.hir.sync_block[<ALL_SUB_VECTOR>] tvector_pipe = <PIPE_ALL>
// CHECK-NEXT:       %subview_15 = memref.subview %101[0, %2] [2, 32] [1, 1] : memref<2x64xf32> to memref<2x32xf32, strided<[64, 1], offset: ?>>
// CHECK-NEXT:       %alloc_16 = memref.alloc() : memref<2x32xf32>
// CHECK-NEXT:       memref.copy %subview_15, %alloc_16 : memref<2x32xf32, strided<[64, 1], offset: ?>> to memref<2x32xf32>
// CHECK-NEXT:       %102 = bufferization.to_tensor %alloc_16 restrict writable {DataUse} : memref<2x32xf32>
// CHECK-NEXT:       %103 = "tt.reduce"(%102) <{axis = 0 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:         %107 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:         tt.reduce.return %107 : f32
// CHECK-NEXT:       }) {DataUse, tiled_op} : (tensor<2x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:       %104 = arith.subf %99, %103 {DataUse} : tensor<32xf32>
// CHECK-NEXT:       %105 = arith.addf %39#2, %104 {DataUse} : tensor<32xf32>
// CHECK-NEXT:       %106 = arith.truncf %105 {DataUse} : tensor<32xf32> to tensor<32xbf16>
// CHECK-NEXT:       tt.store %98, %106 {boundaryCheck = array<i32: 0>, tiled_op} : !tt.ptr<tensor<32xbf16>>
// CHECK-NEXT:       tt.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
    tt.func public @prepare_wy_repr_bwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32) attributes {noinline = false} {
      %c2_i32 = arith.constant 2 : i32
      %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
      %c1_i32 = arith.constant 1 : i32
      %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
      %c4096_i64 = arith.constant 4096 : i64
      %c128_i64 = arith.constant 128 : i64
      %c128_i32 = arith.constant 128 : i32
      %c0_i32 = arith.constant 0 : i32
      %c2048_i64 = arith.constant 2048 : i64
      %c1_i64 = arith.constant 1 : i64
      %c64_i64 = arith.constant 64 : i64
      %c32_i64 = arith.constant 32 : i64
      %c64_i32 = arith.constant 64 : i32
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = tt.get_program_id y : i32
      %2 = arith.divsi %1, %c32_i32 : i32
      %3 = arith.remsi %1, %c32_i32 : i32
      %4 = arith.muli %2, %arg11 : i32
      %5 = arith.muli %4, %c32_i32 : i32
      %6 = arith.addi %5, %3 : i32
      %7 = tt.addptr %arg2, %6 : !tt.ptr<bf16>, i32
      %8 = arith.muli %0, %c64_i32 : i32
      %9 = arith.extsi %arg11 : i32 to i64
      %10 = tt.make_tensor_ptr %7, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
      %11 = tt.addptr %arg9, %6 : !tt.ptr<bf16>, i32
      %12 = tt.make_tensor_ptr %11, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
      %13 = arith.muli %6, %c64_i32 : i32
      %14 = tt.addptr %arg4, %13 : !tt.ptr<bf16>, i32
      %15 = tt.make_tensor_ptr %14, [%c64_i64, %9], [%c1_i64, %c2048_i64], [%c0_i32, %8] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
      %16 = tt.load %10 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
      %17 = tt.load %15 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %18 = tt.addptr %arg3, %6 : !tt.ptr<bf16>, i32
      %19 = tt.make_tensor_ptr %18, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
      %20 = tt.load %19 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
      %21 = math.exp %20 : tensor<64xbf16>
      %22 = arith.muli %6, %c128_i32 : i32
      %23 = tt.addptr %arg0, %22 : !tt.ptr<bf16>, i32
      %24 = tt.addptr %arg7, %22 : !tt.ptr<bf16>, i32
      %25 = tt.addptr %arg5, %22 : !tt.ptr<bf16>, i32
      %26 = arith.mulf %16, %21 : tensor<64xbf16>
      %27 = tt.expand_dims %26 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
      %28 = tt.broadcast %27 : tensor<64x1xbf16> -> tensor<64x64xbf16>
      %29 = arith.extf %27 : tensor<64x1xbf16> to tensor<64x1xf32>
      %30 = tt.broadcast %29 : tensor<64x1xf32> -> tensor<64x64xf32>
      %31 = tt.expand_dims %21 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
      %32 = arith.extf %31 : tensor<64x1xbf16> to tensor<64x1xf32>
      %33 = tt.broadcast %32 : tensor<64x1xf32> -> tensor<64x64xf32>
      %34:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_0, %arg14 = %cst, %arg15 = %cst) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
        %98 = arith.muli %arg12, %c64_i32 : i32
        %99 = tt.make_tensor_ptr %23, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %100 = tt.make_tensor_ptr %24, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %101 = tt.make_tensor_ptr %25, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %102 = tt.load %99 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %103 = arith.mulf %102, %28 : tensor<64x64xbf16>
        %104 = tt.load %101 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %105 = tt.trans %103 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %106 = tt.dot %104, %105, %arg13 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %107 = tt.dot %17, %104, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %108 = arith.mulf %107, %30 : tensor<64x64xf32>
        %109 = arith.extf %102 : tensor<64x64xbf16> to tensor<64x64xf32>
        %110 = arith.mulf %107, %109 : tensor<64x64xf32>
        %111 = arith.mulf %110, %33 : tensor<64x64xf32>
        %112 = "tt.reduce"(%111) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %119 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %119 : f32
        }) : (tensor<64x64xf32>) -> tensor<64xf32>
        %113 = arith.addf %arg14, %112 : tensor<64xf32>
        %114 = arith.extf %103 : tensor<64x64xbf16> to tensor<64x64xf32>
        %115 = arith.mulf %107, %114 : tensor<64x64xf32>
        %116 = "tt.reduce"(%115) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %119 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %119 : f32
        }) : (tensor<64x64xf32>) -> tensor<64xf32>
        %117 = arith.addf %arg15, %116 : tensor<64xf32>
        %118 = arith.truncf %108 : tensor<64x64xf32> to tensor<64x64xbf16>
        tt.store %100, %118 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
        scf.yield %106, %113, %117 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
      }
      %35 = arith.muli %6, %c128_i32 : i32
      %36 = tt.addptr %arg1, %35 : !tt.ptr<bf16>, i32
      %37 = tt.addptr %arg8, %35 : !tt.ptr<bf16>, i32
      %38 = tt.addptr %arg6, %35 : !tt.ptr<bf16>, i32
      %39 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
      %40 = tt.broadcast %39 : tensor<64x1xbf16> -> tensor<64x64xbf16>
      %41 = arith.extf %39 : tensor<64x1xbf16> to tensor<64x1xf32>
      %42 = tt.broadcast %41 : tensor<64x1xf32> -> tensor<64x64xf32>
      %43:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %34#0, %arg14 = %34#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
        %98 = arith.muli %arg12, %c64_i32 : i32
        %99 = tt.make_tensor_ptr %36, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %100 = tt.make_tensor_ptr %37, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %101 = tt.make_tensor_ptr %38, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %102 = tt.load %99 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %103 = arith.mulf %102, %40 : tensor<64x64xbf16>
        %104 = tt.load %101 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %105 = tt.trans %103 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %106 = tt.dot %104, %105, %arg13 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %107 = tt.dot %17, %104, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %108 = arith.mulf %107, %42 : tensor<64x64xf32>
        %109 = arith.extf %102 : tensor<64x64xbf16> to tensor<64x64xf32>
        %110 = arith.mulf %107, %109 : tensor<64x64xf32>
        %111 = "tt.reduce"(%110) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %114 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %114 : f32
        }) : (tensor<64x64xf32>) -> tensor<64xf32>
        %112 = arith.addf %arg14, %111 : tensor<64xf32>
        %113 = arith.truncf %108 : tensor<64x64xf32> to tensor<64x64xbf16>
        tt.store %100, %113 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
        scf.yield %106, %112 : tensor<64x64xf32>, tensor<64xf32>
      }
      %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %45 = tt.splat %8 : i32 -> tensor<64xi32>
      %46 = arith.addi %45, %44 : tensor<64xi32>
      %47 = tt.splat %arg11 : i32 -> tensor<64xi32>
      %48 = arith.cmpi slt, %46, %47 : tensor<64xi32>
      %49 = tt.expand_dims %46 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %50 = tt.expand_dims %46 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %51 = tt.broadcast %49 : tensor<64x1xi32> -> tensor<64x64xi32>
      %52 = tt.broadcast %50 : tensor<1x64xi32> -> tensor<64x64xi32>
      %53 = arith.cmpi sgt, %51, %52 : tensor<64x64xi32>
      %54 = tt.expand_dims %48 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
      %55 = tt.expand_dims %48 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %56 = tt.broadcast %54 : tensor<64x1xi1> -> tensor<64x64xi1>
      %57 = tt.broadcast %55 : tensor<1x64xi1> -> tensor<64x64xi1>
      %58 = arith.andi %56, %57 : tensor<64x64xi1>
      %59 = arith.andi %53, %58 : tensor<64x64xi1>
      %60 = arith.select %59, %43#0, %cst_0 : tensor<64x64xi1>, tensor<64x64xf32>
      %61 = arith.truncf %60 : tensor<64x64xf32> to tensor<64x64xbf16>
      %62 = tt.dot %61, %17, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %63 = arith.truncf %62 : tensor<64x64xf32> to tensor<64x64xbf16>
      %64 = tt.dot %17, %63, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %65 = tt.expand_dims %20 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
      %66 = tt.expand_dims %20 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
      %67 = tt.broadcast %65 : tensor<64x1xbf16> -> tensor<64x64xbf16>
      %68 = tt.broadcast %66 : tensor<1x64xbf16> -> tensor<64x64xbf16>
      %69 = arith.subf %67, %68 : tensor<64x64xbf16>
      %70 = arith.extf %69 : tensor<64x64xbf16> to tensor<64x64xf32>
      %71 = math.exp %70 : tensor<64x64xf32>
      %72 = arith.mulf %64, %71 : tensor<64x64xf32>
      %73 = arith.subf %cst_0, %72 : tensor<64x64xf32>
      %74 = arith.select %59, %73, %cst_0 : tensor<64x64xi1>, tensor<64x64xf32>
      %75 = arith.truncf %74 : tensor<64x64xf32> to tensor<64x64xbf16>
      gpu.barrier
      %76 = arith.muli %6, %c128_i32 : i32
      %77 = tt.addptr %arg0, %76 : !tt.ptr<bf16>, i32
      %78 = tt.addptr %arg7, %76 : !tt.ptr<bf16>, i32
      %79 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
      %80 = tt.broadcast %79 : tensor<64x1xbf16> -> tensor<64x64xbf16>
      %81 = arith.extf %79 : tensor<64x1xbf16> to tensor<64x1xf32>
      %82 = tt.broadcast %81 : tensor<64x1xf32> -> tensor<64x64xf32>
      %83:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_0, %arg14 = %43#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
        %98 = arith.muli %arg12, %c64_i32 : i32
        %99 = tt.make_tensor_ptr %77, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %100 = tt.make_tensor_ptr %78, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %101 = tt.load %99 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %102 = tt.trans %101 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %103 = arith.mulf %101, %80 : tensor<64x64xbf16>
        %104 = tt.dot %101, %102, %arg13 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %105 = tt.dot %75, %101, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %106 = arith.extf %101 : tensor<64x64xbf16> to tensor<64x64xf32>
        %107 = arith.mulf %105, %106 : tensor<64x64xf32>
        %108 = "tt.reduce"(%107) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %119 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %119 : f32
        }) : (tensor<64x64xf32>) -> tensor<64xf32>
        %109 = arith.addf %arg14, %108 : tensor<64xf32>
        %110 = arith.mulf %105, %82 : tensor<64x64xf32>
        %111 = tt.trans %103 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %112 = tt.dot %111, %75, %cst_0 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %113 = tt.trans %112 {order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
        %114 = arith.addf %110, %113 : tensor<64x64xf32>
        %115 = tt.load %100 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %116 = arith.extf %115 : tensor<64x64xbf16> to tensor<64x64xf32>
        %117 = arith.addf %114, %116 : tensor<64x64xf32>
        %118 = arith.truncf %117 : tensor<64x64xf32> to tensor<64x64xbf16>
        tt.store %100, %118 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
        scf.yield %104, %109 : tensor<64x64xf32>, tensor<64xf32>
      }
      %84 = arith.truncf %83#1 : tensor<64xf32> to tensor<64xbf16>
      tt.store %12, %84 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xbf16>>
      %85 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
      %86 = arith.extf %85 : tensor<64x1xbf16> to tensor<64x1xf32>
      %87 = tt.broadcast %86 : tensor<64x1xf32> -> tensor<64x64xf32>
      %88 = arith.mulf %83#0, %87 : tensor<64x64xf32>
      %89 = arith.extf %75 : tensor<64x64xbf16> to tensor<64x64xf32>
      %90 = arith.mulf %89, %88 : tensor<64x64xf32>
      %91 = tt.addptr %arg10, %6 : !tt.ptr<bf16>, i32
      %92 = tt.make_tensor_ptr %91, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
      %93 = "tt.reduce"(%90) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %98 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %98 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %94 = "tt.reduce"(%90) <{axis = 0 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %98 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %98 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %95 = arith.subf %93, %94 : tensor<64xf32>
      %96 = arith.addf %34#2, %95 : tensor<64xf32>
      %97 = arith.truncf %96 : tensor<64xf32> to tensor<64xbf16>
      tt.store %92, %97 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xbf16>>
      tt.return
    }
  }
  module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.disable_auto_tile_and_bind_subblock} {
    tt.func public @prepare_wy_repr_bwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32) attributes {noinline = false} {
      %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
      %cst_0 = arith.constant dense<0.000000e+00> : tensor<32xf32>
      %c32_i32 = arith.constant 32 : i32
      %c64_i32 = arith.constant 64 : i32
      %c32_i64 = arith.constant 32 : i64
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2048_i64 = arith.constant 2048 : i64
      %c0_i32 = arith.constant 0 : i32
      %c128_i32 = arith.constant 128 : i32
      %c128_i64 = arith.constant 128 : i64
      %c4096_i64 = arith.constant 4096 : i64
      %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %0 = hivm.hir.get_sub_block_idx -> i64
      %1 = arith.index_cast %0 : i64 to index
      %2 = affine.apply #map()[%1]
      %3 = arith.index_cast %2 : index to i32
      %4 = tt.get_program_id x : i32
      %5 = tt.get_program_id y : i32
      %6 = arith.divsi %5, %c32_i32 : i32
      %7 = arith.remsi %5, %c32_i32 : i32
      %8 = arith.muli %6, %arg11 : i32
      %9 = arith.muli %8, %c32_i32 : i32
      %10 = arith.addi %9, %7 : i32
      %11 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
      %12 = arith.muli %4, %c64_i32 : i32
      %13 = arith.extsi %arg11 : i32 to i64
      %14 = arith.addi %12, %3 : i32
      %15 = tt.make_tensor_ptr %11, [%13], [%c32_i64], [%14] {order = array<i32: 0>} : <tensor<32xbf16>>
      %16 = tt.addptr %arg9, %10 : !tt.ptr<bf16>, i32
      %17 = tt.make_tensor_ptr %16, [%13], [%c32_i64], [%14] {order = array<i32: 0>, tiled_op} : <tensor<32xbf16>>
      %18 = arith.muli %10, %c64_i32 : i32
      %19 = tt.addptr %arg4, %18 : !tt.ptr<bf16>, i32
      %20 = tt.make_tensor_ptr %19, [%c64_i64, %13], [%c1_i64, %c2048_i64], [%c0_i32, %12] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
      %21 = tt.load %15 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<32xbf16>>
      %22 = tt.load %20 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
      %24 = tt.make_tensor_ptr %23, [%13], [%c32_i64], [%12] {order = array<i32: 0>} : <tensor<64xbf16>>
      %25 = tt.load %24 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
      %extracted_slice = tensor.extract_slice %25[%2] [32] [1] {should_kept_slice} : tensor<64xbf16> to tensor<32xbf16>
      %26 = math.exp %extracted_slice : tensor<32xbf16>
      %27 = arith.muli %10, %c128_i32 : i32
      %28 = tt.addptr %arg0, %27 : !tt.ptr<bf16>, i32
      %29 = tt.addptr %arg7, %27 : !tt.ptr<bf16>, i32
      %30 = tt.addptr %arg5, %27 : !tt.ptr<bf16>, i32
      %31 = arith.mulf %21, %26 : tensor<32xbf16>
      %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
      %33 = tt.broadcast %32 : tensor<32x1xbf16> -> tensor<32x64xbf16>
      %34 = arith.extf %32 : tensor<32x1xbf16> to tensor<32x1xf32>
      %35 = tt.broadcast %34 : tensor<32x1xf32> -> tensor<32x64xf32>
      %36 = tt.expand_dims %26 {axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
      %37 = arith.extf %36 : tensor<32x1xbf16> to tensor<32x1xf32>
      %38 = tt.broadcast %37 : tensor<32x1xf32> -> tensor<32x64xf32>
      %39:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_1, %arg14 = %cst_0, %arg15 = %cst_0) -> (tensor<64x64xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
        %107 = arith.muli %arg12, %c64_i32 : i32
        %108 = tt.make_tensor_ptr %28, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
        %109 = tt.make_tensor_ptr %29, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
        %110 = tt.make_tensor_ptr %30, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %107] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %111 = tt.load %108 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
        %112 = arith.mulf %111, %33 : tensor<32x64xbf16>
        %113 = tt.load %110 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %114 = memref_ext.alloc_workspace() : memref<64x64xbf16>
        %subview_17 = memref.subview %114[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %112 in writable %subview_17 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
        %alloc_18 = memref.alloc() : memref<64x64xbf16>
        memref.copy %114, %alloc_18 : memref<64x64xbf16> to memref<64x64xbf16>
        %115 = bufferization.to_tensor %alloc_18 restrict writable : memref<64x64xbf16>
        %116 = tt.trans %115 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %117 = tt.dot %113, %116, %arg13 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %118 = tt.dot %22, %113, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %119 = memref_ext.alloc_workspace() : memref<64x64xf32>
        bufferization.materialize_in_destination %118 in writable %119 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        %subview_19 = memref.subview %119[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
        %alloc_20 = memref.alloc() : memref<32x64xf32>
        memref.copy %subview_19, %alloc_20 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
        %120 = bufferization.to_tensor %alloc_20 restrict writable : memref<32x64xf32>
        %121 = arith.mulf %120, %35 : tensor<32x64xf32>
        %122 = arith.extf %111 : tensor<32x64xbf16> to tensor<32x64xf32>
        %123 = arith.mulf %120, %122 : tensor<32x64xf32>
        %124 = arith.mulf %123, %38 : tensor<32x64xf32>
        %125 = "tt.reduce"(%124) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %132 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %132 : f32
        }) : (tensor<32x64xf32>) -> tensor<32xf32>
        %126 = arith.addf %arg14, %125 : tensor<32xf32>
        %127 = arith.extf %112 : tensor<32x64xbf16> to tensor<32x64xf32>
        %128 = arith.mulf %120, %127 : tensor<32x64xf32>
        %129 = "tt.reduce"(%128) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %132 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %132 : f32
        }) : (tensor<32x64xf32>) -> tensor<32xf32>
        %130 = arith.addf %arg15, %129 : tensor<32xf32>
        %131 = arith.truncf %121 : tensor<32x64xf32> to tensor<32x64xbf16>
        tt.store %109, %131 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
        scf.yield %117, %126, %130 : tensor<64x64xf32>, tensor<32xf32>, tensor<32xf32>
      }
      %40 = tt.addptr %arg1, %27 : !tt.ptr<bf16>, i32
      %41 = tt.addptr %arg8, %27 : !tt.ptr<bf16>, i32
      %42 = tt.addptr %arg6, %27 : !tt.ptr<bf16>, i32
      %43 = tt.expand_dims %21 {axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
      %44 = tt.broadcast %43 : tensor<32x1xbf16> -> tensor<32x64xbf16>
      %45 = arith.extf %43 : tensor<32x1xbf16> to tensor<32x1xf32>
      %46 = tt.broadcast %45 : tensor<32x1xf32> -> tensor<32x64xf32>
      %47:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %39#0, %arg14 = %39#1) -> (tensor<64x64xf32>, tensor<32xf32>)  : i32 {
        %107 = arith.muli %arg12, %c64_i32 : i32
        %108 = tt.make_tensor_ptr %40, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
        %109 = tt.make_tensor_ptr %41, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
        %110 = tt.make_tensor_ptr %42, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %107] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %111 = tt.load %108 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
        %112 = arith.mulf %111, %44 : tensor<32x64xbf16>
        %113 = tt.load %110 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %114 = memref_ext.alloc_workspace() : memref<64x64xbf16>
        %subview_17 = memref.subview %114[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %112 in writable %subview_17 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
        %alloc_18 = memref.alloc() : memref<64x64xbf16>
        memref.copy %114, %alloc_18 : memref<64x64xbf16> to memref<64x64xbf16>
        %115 = bufferization.to_tensor %alloc_18 restrict writable : memref<64x64xbf16>
        %116 = tt.trans %115 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %117 = tt.dot %113, %116, %arg13 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %118 = tt.dot %22, %113, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %119 = memref_ext.alloc_workspace() : memref<64x64xf32>
        bufferization.materialize_in_destination %118 in writable %119 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        %subview_19 = memref.subview %119[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
        %alloc_20 = memref.alloc() : memref<32x64xf32>
        memref.copy %subview_19, %alloc_20 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
        %120 = bufferization.to_tensor %alloc_20 restrict writable : memref<32x64xf32>
        %121 = arith.mulf %120, %46 : tensor<32x64xf32>
        %122 = arith.extf %111 : tensor<32x64xbf16> to tensor<32x64xf32>
        %123 = arith.mulf %120, %122 : tensor<32x64xf32>
        %124 = "tt.reduce"(%123) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %127 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %127 : f32
        }) : (tensor<32x64xf32>) -> tensor<32xf32>
        %125 = arith.addf %arg14, %124 : tensor<32xf32>
        %126 = arith.truncf %121 : tensor<32x64xf32> to tensor<32x64xbf16>
        tt.store %109, %126 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
        scf.yield %117, %125 : tensor<64x64xf32>, tensor<32xf32>
      }
      %48 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %49 = tt.splat %12 : i32 -> tensor<64xi32>
      %50 = arith.addi %49, %48 : tensor<64xi32>
      %51 = tt.splat %arg11 : i32 -> tensor<64xi32>
      %52 = arith.cmpi slt, %50, %51 : tensor<64xi32>
      %extracted_slice_2 = tensor.extract_slice %50[%2] [32] [1] {should_kept_slice} : tensor<64xi32> to tensor<32xi32>
      %53 = tt.expand_dims %extracted_slice_2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %54 = tt.expand_dims %50 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %55 = tt.broadcast %53 : tensor<32x1xi32> -> tensor<32x64xi32>
      %56 = tt.broadcast %54 : tensor<1x64xi32> -> tensor<32x64xi32>
      %57 = arith.cmpi sgt, %55, %56 : tensor<32x64xi32>
      %extracted_slice_3 = tensor.extract_slice %52[%2] [32] [1] {should_kept_slice} : tensor<64xi1> to tensor<32xi1>
      %58 = tt.expand_dims %extracted_slice_3 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
      %59 = tt.expand_dims %52 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
      %60 = tt.broadcast %58 : tensor<32x1xi1> -> tensor<32x64xi1>
      %61 = tt.broadcast %59 : tensor<1x64xi1> -> tensor<32x64xi1>
      %62 = arith.andi %60, %61 : tensor<32x64xi1>
      %63 = arith.andi %57, %62 : tensor<32x64xi1>
      %64 = memref_ext.alloc_workspace() : memref<64x64xf32>
      bufferization.materialize_in_destination %47#0 in writable %64 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
      %subview = memref.subview %64[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x64xf32>
      memref.copy %subview, %alloc : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
      %65 = bufferization.to_tensor %alloc restrict writable : memref<32x64xf32>
      %66 = arith.select %63, %65, %cst : tensor<32x64xi1>, tensor<32x64xf32>
      %67 = arith.truncf %66 : tensor<32x64xf32> to tensor<32x64xbf16>
      %68 = memref_ext.alloc_workspace() : memref<64x64xbf16>
      %subview_4 = memref.subview %68[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
      bufferization.materialize_in_destination %67 in writable %subview_4 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
      %alloc_5 = memref.alloc() : memref<64x64xbf16>
      memref.copy %68, %alloc_5 : memref<64x64xbf16> to memref<64x64xbf16>
      %69 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xbf16>
      %70 = tt.dot %69, %22, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %71 = memref_ext.alloc_workspace() : memref<64x64xf32>
      bufferization.materialize_in_destination %70 in writable %71 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
      %subview_6 = memref.subview %71[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %alloc_7 = memref.alloc() : memref<32x64xf32>
      memref.copy %subview_6, %alloc_7 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
      %72 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x64xf32>
      %73 = arith.truncf %72 : tensor<32x64xf32> to tensor<32x64xbf16>
      %74 = memref_ext.alloc_workspace() : memref<64x64xbf16>
      %subview_8 = memref.subview %74[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
      bufferization.materialize_in_destination %73 in writable %subview_8 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
      %alloc_9 = memref.alloc() : memref<64x64xbf16>
      memref.copy %74, %alloc_9 : memref<64x64xbf16> to memref<64x64xbf16>
      %75 = bufferization.to_tensor %alloc_9 restrict writable : memref<64x64xbf16>
      %76 = tt.dot %22, %75, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %77 = tt.expand_dims %extracted_slice {axis = 1 : i32} : tensor<32xbf16> -> tensor<32x1xbf16>
      %78 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
      %79 = tt.broadcast %77 : tensor<32x1xbf16> -> tensor<32x64xbf16>
      %80 = tt.broadcast %78 : tensor<1x64xbf16> -> tensor<32x64xbf16>
      %81 = arith.subf %79, %80 : tensor<32x64xbf16>
      %82 = arith.extf %81 : tensor<32x64xbf16> to tensor<32x64xf32>
      %83 = math.exp %82 : tensor<32x64xf32>
      %84 = memref_ext.alloc_workspace() : memref<64x64xf32>
      bufferization.materialize_in_destination %76 in writable %84 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
      %subview_10 = memref.subview %84[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %alloc_11 = memref.alloc() : memref<32x64xf32>
      memref.copy %subview_10, %alloc_11 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
      %85 = bufferization.to_tensor %alloc_11 restrict writable : memref<32x64xf32>
      %86 = arith.mulf %85, %83 : tensor<32x64xf32>
      %87 = arith.subf %cst, %86 : tensor<32x64xf32>
      %88 = arith.select %63, %87, %cst : tensor<32x64xi1>, tensor<32x64xf32>
      %89 = arith.truncf %88 : tensor<32x64xf32> to tensor<32x64xbf16>
      gpu.barrier
      %90:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_1, %arg14 = %47#1) -> (tensor<64x64xf32>, tensor<32xf32>)  : i32 {
        %107 = arith.muli %arg12, %c64_i32 : i32
        %108 = tt.make_tensor_ptr %28, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %107] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
        %109 = tt.make_tensor_ptr %28, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
        %110 = tt.make_tensor_ptr %29, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>} : <tensor<32x64xbf16>>
        %111 = tt.make_tensor_ptr %29, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%14, %107] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
        %112 = tt.load %109 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
        %113 = tt.load %108 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
        %114 = tt.trans %113 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %115 = arith.mulf %112, %44 : tensor<32x64xbf16>
        %116 = tt.dot %113, %114, %arg13 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %117 = memref_ext.alloc_workspace() : memref<64x64xbf16>
        %subview_17 = memref.subview %117[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %89 in writable %subview_17 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
        %alloc_18 = memref.alloc() : memref<64x64xbf16>
        memref.copy %117, %alloc_18 : memref<64x64xbf16> to memref<64x64xbf16>
        %118 = bufferization.to_tensor %alloc_18 restrict writable : memref<64x64xbf16>
        %119 = tt.dot %118, %113, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %120 = memref_ext.alloc_workspace() : memref<64x64xf32>
        bufferization.materialize_in_destination %119 in writable %120 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        %subview_19 = memref.subview %120[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
        %alloc_20 = memref.alloc() : memref<32x64xf32>
        memref.copy %subview_19, %alloc_20 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
        %121 = bufferization.to_tensor %alloc_20 restrict writable : memref<32x64xf32>
        %122 = arith.extf %112 : tensor<32x64xbf16> to tensor<32x64xf32>
        %123 = arith.mulf %121, %122 : tensor<32x64xf32>
        %124 = "tt.reduce"(%123) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %139 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %139 : f32
        }) : (tensor<32x64xf32>) -> tensor<32xf32>
        %125 = arith.addf %arg14, %124 : tensor<32xf32>
        %126 = arith.mulf %121, %46 : tensor<32x64xf32>
        %127 = memref_ext.alloc_workspace() : memref<64x64xbf16>
        %subview_21 = memref.subview %127[%2, 0] [32, 64] [1, 1] : memref<64x64xbf16> to memref<32x64xbf16, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %115 in writable %subview_21 : (tensor<32x64xbf16>, memref<32x64xbf16, strided<[64, 1], offset: ?>>) -> ()
        %alloc_22 = memref.alloc() : memref<64x64xbf16>
        memref.copy %127, %alloc_22 : memref<64x64xbf16> to memref<64x64xbf16>
        %128 = bufferization.to_tensor %alloc_22 restrict writable : memref<64x64xbf16>
        %129 = tt.trans %128 {order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
        %130 = tt.dot %129, %118, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        %131 = memref_ext.alloc_workspace() : memref<64x64xf32>
        bufferization.materialize_in_destination %130 in writable %131 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
        %subview_23 = memref.subview %131[0, %2] [64, 32] [1, 1] : memref<64x64xf32> to memref<64x32xf32, strided<[64, 1], offset: ?>>
        %alloc_24 = memref.alloc() : memref<64x32xf32>
        memref.copy %subview_23, %alloc_24 : memref<64x32xf32, strided<[64, 1], offset: ?>> to memref<64x32xf32>
        %132 = bufferization.to_tensor %alloc_24 restrict writable : memref<64x32xf32>
        %133 = tt.trans %132 {order = array<i32: 1, 0>} : tensor<64x32xf32> -> tensor<32x64xf32>
        %134 = arith.addf %126, %133 : tensor<32x64xf32>
        %135 = tt.load %110 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xbf16>>
        %136 = arith.extf %135 : tensor<32x64xbf16> to tensor<32x64xf32>
        %137 = arith.addf %134, %136 : tensor<32x64xf32>
        %138 = arith.truncf %137 : tensor<32x64xf32> to tensor<32x64xbf16>
        tt.store %111, %138 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
        scf.yield %116, %125 : tensor<64x64xf32>, tensor<32xf32>
      }
      %91 = arith.truncf %90#1 : tensor<32xf32> to tensor<32xbf16>
      tt.store %17, %91 {boundaryCheck = array<i32: 0>, tiled_op} : !tt.ptr<tensor<32xbf16>>
      %92 = memref_ext.alloc_workspace() : memref<64x64xf32>
      bufferization.materialize_in_destination %90#0 in writable %92 : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
      %subview_12 = memref.subview %92[%2, 0] [32, 64] [1, 1] : memref<64x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<32x64xf32>
      memref.copy %subview_12, %alloc_13 : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<32x64xf32>
      %93 = bufferization.to_tensor %alloc_13 restrict writable : memref<32x64xf32>
      %94 = arith.mulf %93, %46 : tensor<32x64xf32>
      %95 = arith.extf %89 : tensor<32x64xbf16> to tensor<32x64xf32>
      %96 = arith.mulf %95, %94 : tensor<32x64xf32>
      %97 = tt.addptr %arg10, %10 : !tt.ptr<bf16>, i32
      %98 = tt.make_tensor_ptr %97, [%13], [%c32_i64], [%14] {order = array<i32: 0>, tiled_op} : <tensor<32xbf16>>
      %99 = "tt.reduce"(%96) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %107 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %107 : f32
      }) : (tensor<32x64xf32>) -> tensor<32xf32>
      %100 = "tt.reduce"(%96) <{axis = 0 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %107 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %107 : f32
      }) {tiled_op} : (tensor<32x64xf32>) -> tensor<64xf32>
      %101 = memref_ext.alloc_workspace() : memref<2x64xf32>
      %subview_14 = memref.subview %101[%1, 0] [1, 64] [1, 1] : memref<2x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %expanded = tensor.expand_shape %100 [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
      bufferization.materialize_in_destination %expanded in writable %subview_14 : (tensor<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>>) -> ()
      hivm.hir.sync_block[<ALL_SUB_VECTOR>] tvector_pipe = <PIPE_ALL>
      %subview_15 = memref.subview %101[0, %2] [2, 32] [1, 1] : memref<2x64xf32> to memref<2x32xf32, strided<[64, 1], offset: ?>>
      %alloc_16 = memref.alloc() : memref<2x32xf32>
      memref.copy %subview_15, %alloc_16 : memref<2x32xf32, strided<[64, 1], offset: ?>> to memref<2x32xf32>
      %102 = bufferization.to_tensor %alloc_16 restrict writable : memref<2x32xf32>
      %103 = "tt.reduce"(%102) <{axis = 0 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %107 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %107 : f32
      }) {tiled_op} : (tensor<2x32xf32>) -> tensor<32xf32>
      %104 = arith.subf %99, %103 : tensor<32xf32>
      %105 = arith.addf %39#2, %104 : tensor<32xf32>
      %106 = arith.truncf %105 : tensor<32xf32> to tensor<32xbf16>
      tt.store %98, %106 {boundaryCheck = array<i32: 0>, tiled_op} : !tt.ptr<tensor<32xbf16>>
      tt.return
    }
  }
}

// -----
// Source: gaoyou/0701/recompute_w_u/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @recompute_w_u_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = arith.divsi %1, %c32_i32 : i32
// CHECK-NEXT:     %3 = arith.remsi %1, %c32_i32 : i32
// CHECK-NEXT:     %4 = arith.muli %2, %arg7 : i32
// CHECK-NEXT:     %5 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:     %6 = tt.addptr %arg2, %5 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %7 = tt.addptr %6, %3 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %8 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:     %9 = arith.extsi %arg7 : i32 to i64
// CHECK-NEXT:     %10 = tt.make_tensor_ptr %7, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %11 = tt.load %10 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %12 = arith.addi %5, %3 : i32
// CHECK-NEXT:     %13 = arith.muli %12, %c64_i32 : i32
// CHECK-NEXT:     %14 = tt.addptr %arg5, %13 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %15 = tt.make_tensor_ptr %14, [%9, %c64_i64], [%c2048_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:     %16 = tt.load %15 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:     %17 = arith.muli %12, %c128_i32 : i32
// CHECK-NEXT:     %18 = tt.addptr %arg1, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %19 = tt.make_tensor_ptr %18, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %20 = tt.addptr %arg4, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %21 = tt.make_tensor_ptr %20, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %22 = tt.load %19 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %23 = tt.expand_dims %11 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %24 = tt.broadcast %23 {DataUse} : tensor<64x1xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:     %25 = arith.mulf %22, %24 {DataUse} : tensor<64x128xbf16>
// CHECK-NEXT:     %26 = tt.dot %16, %25, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %27 = arith.truncf %26 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     tt.store %21, %27 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %28 = tt.addptr %arg6, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %29 = tt.make_tensor_ptr %28, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %30 = tt.load %29 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %31 = arith.extf %30 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %32 = math.exp %31 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %33 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %34 = tt.make_tensor_ptr %33, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %35 = tt.addptr %arg3, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %36 = tt.make_tensor_ptr %35, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %37 = tt.load %34 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %38 = arith.mulf %37, %24 {DataUse} : tensor<64x128xbf16>
// CHECK-NEXT:     %39 = tt.expand_dims %32 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %40 = arith.extf %38 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
// CHECK-NEXT:     %41 = tt.broadcast %39 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %42 = arith.mulf %40, %41 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %43 = arith.truncf %42 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     %44 = tt.dot %16, %43, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %45 = arith.truncf %44 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     tt.store %36, %45 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @recompute_w_u_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %c4096_i64 = arith.constant 4096 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.remsi %1, %c32_i32 : i32
    %4 = arith.muli %2, %arg7 : i32
    %5 = arith.muli %4, %c32_i32 : i32
    %6 = tt.addptr %arg2, %5 : !tt.ptr<bf16>, i32
    %7 = tt.addptr %6, %3 : !tt.ptr<bf16>, i32
    %8 = arith.muli %0, %c64_i32 : i32
    %9 = arith.extsi %arg7 : i32 to i64
    %10 = tt.make_tensor_ptr %7, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
    %11 = tt.load %10 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %12 = arith.addi %5, %3 : i32
    %13 = arith.muli %12, %c64_i32 : i32
    %14 = tt.addptr %arg5, %13 : !tt.ptr<bf16>, i32
    %15 = tt.make_tensor_ptr %14, [%9, %c64_i64], [%c2048_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
    %16 = tt.load %15 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
    %17 = arith.muli %12, %c128_i32 : i32
    %18 = tt.addptr %arg1, %17 : !tt.ptr<bf16>, i32
    %19 = tt.make_tensor_ptr %18, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %20 = tt.addptr %arg4, %17 : !tt.ptr<bf16>, i32
    %21 = tt.make_tensor_ptr %20, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %22 = tt.load %19 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %23 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %24 = tt.broadcast %23 : tensor<64x1xbf16> -> tensor<64x128xbf16>
    %25 = arith.mulf %22, %24 : tensor<64x128xbf16>
    %26 = tt.dot %16, %25, %cst {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %27 = arith.truncf %26 : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %21, %27 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    %28 = tt.addptr %arg6, %12 : !tt.ptr<bf16>, i32
    %29 = tt.make_tensor_ptr %28, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
    %30 = tt.load %29 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %31 = arith.extf %30 : tensor<64xbf16> to tensor<64xf32>
    %32 = math.exp %31 : tensor<64xf32>
    %33 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i32
    %34 = tt.make_tensor_ptr %33, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %35 = tt.addptr %arg3, %17 : !tt.ptr<bf16>, i32
    %36 = tt.make_tensor_ptr %35, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %37 = tt.load %34 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %38 = arith.mulf %37, %24 : tensor<64x128xbf16>
    %39 = tt.expand_dims %32 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %40 = arith.extf %38 : tensor<64x128xbf16> to tensor<64x128xf32>
    %41 = tt.broadcast %39 : tensor<64x1xf32> -> tensor<64x128xf32>
    %42 = arith.mulf %40, %41 : tensor<64x128xf32>
    %43 = arith.truncf %42 : tensor<64x128xf32> to tensor<64x128xbf16>
    %44 = tt.dot %16, %43, %cst {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %45 = arith.truncf %44 : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %36, %45 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/scaled_dot_kkt/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_scaled_dot_kkt_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_program_id y : i32
// CHECK-NEXT:     %2 = arith.divsi %1, %c32_i32 : i32
// CHECK-NEXT:     %3 = arith.remsi %1, %c32_i32 : i32
// CHECK-NEXT:     %4 = arith.muli %2, %arg4 : i32
// CHECK-NEXT:     %5 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:     %6 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %7 = tt.splat %5 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %8 = arith.addi %7, %6 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %9 = tt.splat %arg4 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %10 = arith.cmpi slt, %8, %9 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %11 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:     %12 = tt.addptr %arg2, %11 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %13 = tt.addptr %12, %3 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %14 = arith.extsi %arg4 : i32 to i64
// CHECK-NEXT:     %15 = tt.make_tensor_ptr %13, [%14], [%c32_i64], [%5] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %16 = tt.load %15 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %17 = arith.addi %11, %3 : i32
// CHECK-NEXT:     %18 = arith.muli %17, %c128_i32 : i32
// CHECK-NEXT:     %19 = tt.addptr %arg0, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %20 = tt.make_tensor_ptr %19, [%14, %c128_i64], [%c4096_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %21 = tt.load %20 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %22 = tt.trans %21 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:     %23 = tt.dot %21, %22, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %24 = tt.addptr %arg1, %11 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %25 = tt.addptr %24, %3 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %26 = tt.make_tensor_ptr %25, [%14], [%c32_i64], [%5] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %27 = tt.load %26 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %28 = tt.expand_dims %27 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %29 = tt.expand_dims %27 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:     %30 = tt.broadcast %28 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %31 = tt.broadcast %29 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %32 = arith.subf %30, %31 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:     %33 = arith.extf %32 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %34 = math.exp %33 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %35 = arith.mulf %23, %34 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %36 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %37 = arith.extf %36 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:     %38 = tt.broadcast %37 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:     %39 = arith.mulf %35, %38 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %40 = tt.expand_dims %8 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %41 = tt.expand_dims %8 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %42 = tt.broadcast %40 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %43 = tt.broadcast %41 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %44 = arith.cmpi sgt, %42, %43 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %45 = tt.expand_dims %10 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %46 = tt.expand_dims %10 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %47 = tt.broadcast %45 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %48 = tt.broadcast %46 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %49 = arith.andi %47, %48 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %50 = arith.andi %44, %49 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %51 = arith.select %50, %39, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %52 = arith.muli %17, %c64_i32 : i32
// CHECK-NEXT:     %53 = tt.addptr %arg3, %52 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %54 = tt.make_tensor_ptr %53, [%14, %c64_i64], [%c2048_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:     %55 = arith.truncf %51 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     tt.store %54, %55 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_scaled_dot_kkt_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %c2048_i64 = arith.constant 2048 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c1_i64 = arith.constant 1 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c128_i64 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i64 = arith.constant 32 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.remsi %1, %c32_i32 : i32
    %4 = arith.muli %2, %arg4 : i32
    %5 = arith.muli %0, %c64_i32 : i32
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %7 = tt.splat %5 : i32 -> tensor<64xi32>
    %8 = arith.addi %7, %6 : tensor<64xi32>
    %9 = tt.splat %arg4 : i32 -> tensor<64xi32>
    %10 = arith.cmpi slt, %8, %9 : tensor<64xi32>
    %11 = arith.muli %4, %c32_i32 : i32
    %12 = tt.addptr %arg2, %11 : !tt.ptr<bf16>, i32
    %13 = tt.addptr %12, %3 : !tt.ptr<bf16>, i32
    %14 = arith.extsi %arg4 : i32 to i64
    %15 = tt.make_tensor_ptr %13, [%14], [%c32_i64], [%5] {order = array<i32: 0>} : <tensor<64xbf16>>
    %16 = tt.load %15 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %17 = arith.addi %11, %3 : i32
    %18 = arith.muli %17, %c128_i32 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<bf16>, i32
    %20 = tt.make_tensor_ptr %19, [%14, %c128_i64], [%c4096_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %21 = tt.load %20 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %22 = tt.trans %21 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
    %23 = tt.dot %21, %22, %cst {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
    %24 = tt.addptr %arg1, %11 : !tt.ptr<bf16>, i32
    %25 = tt.addptr %24, %3 : !tt.ptr<bf16>, i32
    %26 = tt.make_tensor_ptr %25, [%14], [%c32_i64], [%5] {order = array<i32: 0>} : <tensor<64xbf16>>
    %27 = tt.load %26 {boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %28 = tt.expand_dims %27 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %29 = tt.expand_dims %27 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
    %30 = tt.broadcast %28 : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %31 = tt.broadcast %29 : tensor<1x64xbf16> -> tensor<64x64xbf16>
    %32 = arith.subf %30, %31 : tensor<64x64xbf16>
    %33 = arith.extf %32 : tensor<64x64xbf16> to tensor<64x64xf32>
    %34 = math.exp %33 : tensor<64x64xf32>
    %35 = arith.mulf %23, %34 : tensor<64x64xf32>
    %36 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %37 = arith.extf %36 : tensor<64x1xbf16> to tensor<64x1xf32>
    %38 = tt.broadcast %37 : tensor<64x1xf32> -> tensor<64x64xf32>
    %39 = arith.mulf %35, %38 : tensor<64x64xf32>
    %40 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %41 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %42 = tt.broadcast %40 : tensor<64x1xi32> -> tensor<64x64xi32>
    %43 = tt.broadcast %41 : tensor<1x64xi32> -> tensor<64x64xi32>
    %44 = arith.cmpi sgt, %42, %43 : tensor<64x64xi32>
    %45 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %46 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %47 = tt.broadcast %45 : tensor<64x1xi1> -> tensor<64x64xi1>
    %48 = tt.broadcast %46 : tensor<1x64xi1> -> tensor<64x64xi1>
    %49 = arith.andi %47, %48 : tensor<64x64xi1>
    %50 = arith.andi %44, %49 : tensor<64x64xi1>
    %51 = arith.select %50, %39, %cst : tensor<64x64xi1>, tensor<64x64xf32>
    %52 = arith.muli %17, %c64_i32 : i32
    %53 = tt.addptr %arg3, %52 : !tt.ptr<bf16>, i32
    %54 = tt.make_tensor_ptr %53, [%14, %c64_i64], [%c2048_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
    %55 = arith.truncf %51 : tensor<64x64xf32> to tensor<64x64xbf16>
    tt.store %54, %55 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_kv_norm_dot_false.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_11 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %cst_12 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %3 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %4 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %5 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %6 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %7 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg9 = %0 to %c64_i32 step %1  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %10 = arith.muli %8, %c524288_i32 : i32
// CHECK-NEXT:       %11 = tt.addptr %arg1, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %12 = tt.addptr %11, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %13 = arith.muli %9, %c64_i32 : i32
// CHECK-NEXT:       %14 = tt.splat %13 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %15 = arith.addi %14, %3 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %16 = tt.expand_dims %15 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:       %17 = arith.muli %16, %cst_14 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %18 = tt.splat %12 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %19 = tt.addptr %18, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %20 = tt.broadcast %19 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %21 = tt.addptr %20, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %22 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %23 = tt.addptr %22, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %24 = tt.splat %23 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %25 = tt.addptr %24, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %26 = tt.broadcast %25 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %27 = tt.addptr %26, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %28 = tt.addptr %arg5, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %29 = tt.addptr %28, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %30 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %31 = tt.addptr %30, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %32 = tt.broadcast %31 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %33 = tt.addptr %32, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %34 = tt.addptr %arg6, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %35 = tt.addptr %34, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %36 = tt.splat %35 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %37 = tt.addptr %36, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %38 = tt.broadcast %37 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %39 = tt.addptr %38, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %40 = arith.cmpi slt, %16, %cst_13 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %41 = tt.broadcast %40 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:       %42 = tt.load %21, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %43 = tt.load %27, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %44 = arith.muli %8, %c2621440_i32 : i32
// CHECK-NEXT:       %45 = tt.addptr %arg0, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %46 = tt.addptr %arg3, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %47 = arith.muli %8, %c20480_i32 : i32
// CHECK-NEXT:       %48 = tt.addptr %arg4, %47 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %49 = tt.addptr %arg7, %47 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %50 = tt.expand_dims %15 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:       %51 = tt.broadcast %50 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:       %52 = arith.cmpi slt, %50, %cst_8 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:       %53 = tt.broadcast %52 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %54 = tt.trans %42 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %55 = tt.trans %43 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %56:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_12, %arg12 = %cst_12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:         %59 = arith.muli %arg10, %c524288_i32 : i32
// CHECK-NEXT:         %60 = tt.addptr %45, %59 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %61 = tt.splat %60 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %62 = tt.addptr %46, %59 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %63 = tt.splat %62 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %64 = arith.muli %arg10, %c4096_i32 : i32
// CHECK-NEXT:         %65 = tt.addptr %48, %64 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %66 = tt.splat %65 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %67 = tt.addptr %49, %64 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %68 = tt.splat %67 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %69:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:           %70 = arith.muli %arg13, %c128_i32 : i32
// CHECK-NEXT:           %71 = tt.splat %70 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %72 = arith.addi %71, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %73 = tt.expand_dims %72 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %74 = arith.muli %73, %cst_11 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %75 = tt.addptr %61, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %76 = tt.broadcast %75 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %77 = tt.addptr %76, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %78 = tt.addptr %63, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %79 = tt.broadcast %78 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %80 = tt.addptr %79, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %81 = tt.addptr %66, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %82 = tt.addptr %68, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %83 = arith.muli %73, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %84 = tt.addptr %7, %83 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:           %85 = tt.broadcast %84 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:           %86 = tt.addptr %85, %51 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:           %87 = arith.cmpi slt, %73, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %88 = arith.cmpi slt, %72, %cst_9 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %89 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:           %90 = arith.andi %89, %53 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:           %91 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %92 = tt.load %77, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %93 = tt.load %80, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %94 = tt.load %82, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %95 = tt.load %81, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %96 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %97 = tt.load %96, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %98 = tt.dot %92, %54, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %99 = arith.mulf %98, %cst_5 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %100 = arith.sitofp %97 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:           %101 = arith.subf %cst_4, %100 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %102 = arith.mulf %101, %cst_3 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %103 = arith.subf %99, %102 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %104 = tt.expand_dims %94 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %105 = tt.broadcast %104 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %106 = arith.subf %103, %105 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %107 = math.exp %106 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %108 = arith.truncf %107 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %109 = tt.trans %108 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %110 = tt.dot %109, %93, %arg14 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %111 = tt.dot %93, %55, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %112 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %113 = tt.broadcast %112 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %114 = arith.subf %111, %113 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %115 = arith.mulf %107, %114 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %116 = arith.truncf %115 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %117 = tt.trans %116 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %118 = tt.dot %117, %92, %cst_12 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %119 = arith.mulf %118, %cst_2 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %120 = arith.addf %arg15, %119 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           scf.yield %110, %120 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         scf.yield %69#0, %69#1 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %57 = arith.truncf %56#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %33, %57, %41 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %58 = arith.truncf %56#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %39, %58, %41 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.0883883461> : tensor<64x128xf32>
    %cst_3 = arith.constant dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_8 = arith.constant dense<4096> : tensor<1x64xi32>
    %cst_9 = arith.constant dense<4096> : tensor<128xi32>
    %cst_10 = arith.constant dense<4096> : tensor<128x1xi32>
    %c20480_i32 = arith.constant 20480 : i32
    %cst_11 = arith.constant dense<128> : tensor<128x1xi32>
    %c2621440_i32 = arith.constant 2621440 : i32
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %cst_13 = arith.constant dense<4096> : tensor<64x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %cst_14 = arith.constant dense<128> : tensor<64x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.expand_dims %2 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %5 = tt.broadcast %4 : tensor<1x128xi32> -> tensor<64x128xi32>
    %6 = tt.broadcast %4 : tensor<1x128xi32> -> tensor<128x128xi32>
    %7 = tt.splat %arg8 : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg9 = %0 to %c64_i32 step %1  : i32 {
      %8 = arith.divsi %arg9, %c64_i32 : i32
      %9 = arith.remsi %arg9, %c64_i32 : i32
      %10 = arith.muli %8, %c524288_i32 : i32
      %11 = tt.addptr %arg1, %10 : !tt.ptr<bf16>, i32
      %12 = tt.addptr %11, %c0_i32 : !tt.ptr<bf16>, i32
      %13 = arith.muli %9, %c64_i32 : i32
      %14 = tt.splat %13 : i32 -> tensor<64xi32>
      %15 = arith.addi %14, %3 : tensor<64xi32>
      %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %17 = arith.muli %16, %cst_14 : tensor<64x1xi32>
      %18 = tt.splat %12 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %19 = tt.addptr %18, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %20 = tt.broadcast %19 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %21 = tt.addptr %20, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %22 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
      %23 = tt.addptr %22, %c0_i32 : !tt.ptr<bf16>, i32
      %24 = tt.splat %23 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %25 = tt.addptr %24, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %26 = tt.broadcast %25 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %27 = tt.addptr %26, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %28 = tt.addptr %arg5, %10 : !tt.ptr<bf16>, i32
      %29 = tt.addptr %28, %c0_i32 : !tt.ptr<bf16>, i32
      %30 = tt.splat %29 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %31 = tt.addptr %30, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %32 = tt.broadcast %31 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %33 = tt.addptr %32, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %34 = tt.addptr %arg6, %10 : !tt.ptr<bf16>, i32
      %35 = tt.addptr %34, %c0_i32 : !tt.ptr<bf16>, i32
      %36 = tt.splat %35 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %37 = tt.addptr %36, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %38 = tt.broadcast %37 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %39 = tt.addptr %38, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %40 = arith.cmpi slt, %16, %cst_13 : tensor<64x1xi32>
      %41 = tt.broadcast %40 : tensor<64x1xi1> -> tensor<64x128xi1>
      %42 = tt.load %21, %41, %cst_1 : tensor<64x128x!tt.ptr<bf16>>
      %43 = tt.load %27, %41, %cst_1 : tensor<64x128x!tt.ptr<bf16>>
      %44 = arith.muli %8, %c2621440_i32 : i32
      %45 = tt.addptr %arg0, %44 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %arg3, %44 : !tt.ptr<bf16>, i32
      %47 = arith.muli %8, %c20480_i32 : i32
      %48 = tt.addptr %arg4, %47 : !tt.ptr<f32>, i32
      %49 = tt.addptr %arg7, %47 : !tt.ptr<f32>, i32
      %50 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %51 = tt.broadcast %50 : tensor<1x64xi32> -> tensor<128x64xi32>
      %52 = arith.cmpi slt, %50, %cst_8 : tensor<1x64xi32>
      %53 = tt.broadcast %52 : tensor<1x64xi1> -> tensor<128x64xi1>
      %54 = tt.trans %42 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %55 = tt.trans %43 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %56:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_12, %arg12 = %cst_12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
        %59 = arith.muli %arg10, %c524288_i32 : i32
        %60 = tt.addptr %45, %59 : !tt.ptr<bf16>, i32
        %61 = tt.splat %60 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %62 = tt.addptr %46, %59 : !tt.ptr<bf16>, i32
        %63 = tt.splat %62 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %64 = arith.muli %arg10, %c4096_i32 : i32
        %65 = tt.addptr %48, %64 : !tt.ptr<f32>, i32
        %66 = tt.splat %65 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %67 = tt.addptr %49, %64 : !tt.ptr<f32>, i32
        %68 = tt.splat %67 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %69:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
          %70 = arith.muli %arg13, %c128_i32 : i32
          %71 = tt.splat %70 : i32 -> tensor<128xi32>
          %72 = arith.addi %71, %2 : tensor<128xi32>
          %73 = tt.expand_dims %72 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %74 = arith.muli %73, %cst_11 : tensor<128x1xi32>
          %75 = tt.addptr %61, %74 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %76 = tt.broadcast %75 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %77 = tt.addptr %76, %6 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %78 = tt.addptr %63, %74 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %79 = tt.broadcast %78 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %80 = tt.addptr %79, %6 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %81 = tt.addptr %66, %72 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %82 = tt.addptr %68, %72 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %83 = arith.muli %73, %cst_10 : tensor<128x1xi32>
          %84 = tt.addptr %7, %83 : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
          %85 = tt.broadcast %84 : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
          %86 = tt.addptr %85, %51 : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
          %87 = arith.cmpi slt, %73, %cst_10 : tensor<128x1xi32>
          %88 = arith.cmpi slt, %72, %cst_9 : tensor<128xi32>
          %89 = tt.broadcast %87 : tensor<128x1xi1> -> tensor<128x64xi1>
          %90 = arith.andi %89, %53 : tensor<128x64xi1>
          %91 = tt.broadcast %87 : tensor<128x1xi1> -> tensor<128x128xi1>
          %92 = tt.load %77, %91, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
          %93 = tt.load %80, %91, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
          %94 = tt.load %82, %88, %cst_7 : tensor<128x!tt.ptr<f32>>
          %95 = tt.load %81, %88, %cst_7 : tensor<128x!tt.ptr<f32>>
          %96 = tt.bitcast %86 : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
          %97 = tt.load %96, %90, %cst {was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
          %98 = tt.dot %92, %54, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %99 = arith.mulf %98, %cst_5 : tensor<128x64xf32>
          %100 = arith.sitofp %97 : tensor<128x64xi8> to tensor<128x64xf32>
          %101 = arith.subf %cst_4, %100 : tensor<128x64xf32>
          %102 = arith.mulf %101, %cst_3 : tensor<128x64xf32>
          %103 = arith.subf %99, %102 : tensor<128x64xf32>
          %104 = tt.expand_dims %94 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %105 = tt.broadcast %104 : tensor<128x1xf32> -> tensor<128x64xf32>
          %106 = arith.subf %103, %105 : tensor<128x64xf32>
          %107 = math.exp %106 : tensor<128x64xf32>
          %108 = arith.truncf %107 : tensor<128x64xf32> to tensor<128x64xbf16>
          %109 = tt.trans %108 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %110 = tt.dot %109, %93, %arg14 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %111 = tt.dot %93, %55, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %112 = tt.expand_dims %95 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %113 = tt.broadcast %112 : tensor<128x1xf32> -> tensor<128x64xf32>
          %114 = arith.subf %111, %113 : tensor<128x64xf32>
          %115 = arith.mulf %107, %114 : tensor<128x64xf32>
          %116 = arith.truncf %115 : tensor<128x64xf32> to tensor<128x64xbf16>
          %117 = tt.trans %116 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %118 = tt.dot %117, %92, %cst_12 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %119 = arith.mulf %118, %cst_2 : tensor<64x128xf32>
          %120 = arith.addf %arg15, %119 : tensor<64x128xf32>
          scf.yield %110, %120 : tensor<64x128xf32>, tensor<64x128xf32>
        }
        scf.yield %69#0, %69#1 : tensor<64x128xf32>, tensor<64x128xf32>
      }
      %57 = arith.truncf %56#1 : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %33, %57, %41 : tensor<64x128x!tt.ptr<bf16>>
      %58 = arith.truncf %56#0 : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %39, %58, %41 : tensor<64x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_kv_norm_dot_true.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_11 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %cst_12 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %3 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %4 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %5 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %6 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %7 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg9 = %0 to %c64_i32 step %1  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %10 = arith.muli %8, %c524288_i32 : i32
// CHECK-NEXT:       %11 = tt.addptr %arg1, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %12 = tt.addptr %11, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %13 = arith.muli %9, %c64_i32 : i32
// CHECK-NEXT:       %14 = tt.splat %13 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %15 = arith.addi %14, %3 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %16 = tt.expand_dims %15 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:       %17 = arith.muli %16, %cst_14 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %18 = tt.splat %12 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %19 = tt.addptr %18, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %20 = tt.broadcast %19 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %21 = tt.addptr %20, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %22 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %23 = tt.addptr %22, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %24 = tt.splat %23 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %25 = tt.addptr %24, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %26 = tt.broadcast %25 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %27 = tt.addptr %26, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %28 = tt.addptr %arg5, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %29 = tt.addptr %28, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %30 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %31 = tt.addptr %30, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %32 = tt.broadcast %31 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %33 = tt.addptr %32, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %34 = tt.addptr %arg6, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %35 = tt.addptr %34, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %36 = tt.splat %35 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %37 = tt.addptr %36, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %38 = tt.broadcast %37 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %39 = tt.addptr %38, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %40 = arith.cmpi slt, %16, %cst_13 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %41 = tt.broadcast %40 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:       %42 = tt.load %21, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %43 = tt.load %27, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %44 = arith.muli %8, %c2621440_i32 : i32
// CHECK-NEXT:       %45 = tt.addptr %arg0, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %46 = tt.addptr %arg3, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %47 = arith.muli %8, %c20480_i32 : i32
// CHECK-NEXT:       %48 = tt.addptr %arg4, %47 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %49 = tt.addptr %arg7, %47 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %50 = tt.expand_dims %15 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:       %51 = tt.broadcast %50 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:       %52 = arith.cmpi slt, %50, %cst_8 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:       %53 = tt.broadcast %52 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %54 = tt.trans %42 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %55 = tt.trans %43 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %56:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_12, %arg12 = %cst_12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:         %59 = arith.muli %arg10, %c524288_i32 : i32
// CHECK-NEXT:         %60 = tt.addptr %45, %59 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %61 = tt.splat %60 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %62 = tt.addptr %46, %59 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %63 = tt.splat %62 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %64 = arith.muli %arg10, %c4096_i32 : i32
// CHECK-NEXT:         %65 = tt.addptr %48, %64 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %66 = tt.splat %65 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %67 = tt.addptr %49, %64 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %68 = tt.splat %67 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %69:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:           %70 = arith.muli %arg13, %c128_i32 : i32
// CHECK-NEXT:           %71 = tt.splat %70 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %72 = arith.addi %71, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %73 = tt.expand_dims %72 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %74 = arith.muli %73, %cst_11 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %75 = tt.addptr %61, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %76 = tt.broadcast %75 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %77 = tt.addptr %76, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %78 = tt.addptr %63, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %79 = tt.broadcast %78 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %80 = tt.addptr %79, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %81 = tt.addptr %66, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %82 = tt.addptr %68, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %83 = arith.muli %73, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %84 = tt.addptr %7, %83 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:           %85 = tt.broadcast %84 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:           %86 = tt.addptr %85, %51 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:           %87 = arith.cmpi slt, %73, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %88 = arith.cmpi slt, %72, %cst_9 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %89 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:           %90 = arith.andi %89, %53 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:           %91 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %92 = tt.load %77, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %93 = tt.load %80, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %94 = tt.load %82, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %95 = tt.load %81, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %96 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %97 = tt.load %96, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %98 = tt.dot %92, %54, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %99 = arith.mulf %98, %cst_5 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %100 = arith.sitofp %97 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:           %101 = arith.subf %cst_4, %100 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %102 = arith.mulf %101, %cst_3 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %103 = arith.subf %99, %102 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %104 = tt.expand_dims %94 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %105 = tt.broadcast %104 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %106 = arith.subf %103, %105 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %107 = math.exp %106 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %108 = arith.truncf %107 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %109 = tt.trans %108 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %110 = tt.dot %109, %93, %arg14 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %111 = tt.dot %93, %55, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %112 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %113 = tt.broadcast %112 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %114 = arith.subf %111, %113 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %115 = arith.mulf %107, %114 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %116 = arith.truncf %115 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %117 = tt.trans %116 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %118 = tt.dot %117, %92, %cst_12 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %119 = arith.mulf %118, %cst_2 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %120 = arith.addf %arg15, %119 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           scf.yield %110, %120 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         scf.yield %69#0, %69#1 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %57 = arith.truncf %56#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %33, %57, %41 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %58 = arith.truncf %56#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %39, %58, %41 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.0883883461> : tensor<64x128xf32>
    %cst_3 = arith.constant dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_8 = arith.constant dense<4096> : tensor<1x64xi32>
    %cst_9 = arith.constant dense<4096> : tensor<128xi32>
    %cst_10 = arith.constant dense<4096> : tensor<128x1xi32>
    %c20480_i32 = arith.constant 20480 : i32
    %cst_11 = arith.constant dense<128> : tensor<128x1xi32>
    %c2621440_i32 = arith.constant 2621440 : i32
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %cst_13 = arith.constant dense<4096> : tensor<64x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %cst_14 = arith.constant dense<128> : tensor<64x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.expand_dims %2 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %5 = tt.broadcast %4 : tensor<1x128xi32> -> tensor<64x128xi32>
    %6 = tt.broadcast %4 : tensor<1x128xi32> -> tensor<128x128xi32>
    %7 = tt.splat %arg8 : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg9 = %0 to %c64_i32 step %1  : i32 {
      %8 = arith.divsi %arg9, %c64_i32 : i32
      %9 = arith.remsi %arg9, %c64_i32 : i32
      %10 = arith.muli %8, %c524288_i32 : i32
      %11 = tt.addptr %arg1, %10 : !tt.ptr<bf16>, i32
      %12 = tt.addptr %11, %c0_i32 : !tt.ptr<bf16>, i32
      %13 = arith.muli %9, %c64_i32 : i32
      %14 = tt.splat %13 : i32 -> tensor<64xi32>
      %15 = arith.addi %14, %3 : tensor<64xi32>
      %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %17 = arith.muli %16, %cst_14 : tensor<64x1xi32>
      %18 = tt.splat %12 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %19 = tt.addptr %18, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %20 = tt.broadcast %19 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %21 = tt.addptr %20, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %22 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
      %23 = tt.addptr %22, %c0_i32 : !tt.ptr<bf16>, i32
      %24 = tt.splat %23 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %25 = tt.addptr %24, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %26 = tt.broadcast %25 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %27 = tt.addptr %26, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %28 = tt.addptr %arg5, %10 : !tt.ptr<bf16>, i32
      %29 = tt.addptr %28, %c0_i32 : !tt.ptr<bf16>, i32
      %30 = tt.splat %29 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %31 = tt.addptr %30, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %32 = tt.broadcast %31 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %33 = tt.addptr %32, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %34 = tt.addptr %arg6, %10 : !tt.ptr<bf16>, i32
      %35 = tt.addptr %34, %c0_i32 : !tt.ptr<bf16>, i32
      %36 = tt.splat %35 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %37 = tt.addptr %36, %17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %38 = tt.broadcast %37 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %39 = tt.addptr %38, %5 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %40 = arith.cmpi slt, %16, %cst_13 : tensor<64x1xi32>
      %41 = tt.broadcast %40 : tensor<64x1xi1> -> tensor<64x128xi1>
      %42 = tt.load %21, %41, %cst_1 : tensor<64x128x!tt.ptr<bf16>>
      %43 = tt.load %27, %41, %cst_1 : tensor<64x128x!tt.ptr<bf16>>
      %44 = arith.muli %8, %c2621440_i32 : i32
      %45 = tt.addptr %arg0, %44 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %arg3, %44 : !tt.ptr<bf16>, i32
      %47 = arith.muli %8, %c20480_i32 : i32
      %48 = tt.addptr %arg4, %47 : !tt.ptr<f32>, i32
      %49 = tt.addptr %arg7, %47 : !tt.ptr<f32>, i32
      %50 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %51 = tt.broadcast %50 : tensor<1x64xi32> -> tensor<128x64xi32>
      %52 = arith.cmpi slt, %50, %cst_8 : tensor<1x64xi32>
      %53 = tt.broadcast %52 : tensor<1x64xi1> -> tensor<128x64xi1>
      %54 = tt.trans %42 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %55 = tt.trans %43 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %56:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_12, %arg12 = %cst_12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
        %59 = arith.muli %arg10, %c524288_i32 : i32
        %60 = tt.addptr %45, %59 : !tt.ptr<bf16>, i32
        %61 = tt.splat %60 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %62 = tt.addptr %46, %59 : !tt.ptr<bf16>, i32
        %63 = tt.splat %62 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %64 = arith.muli %arg10, %c4096_i32 : i32
        %65 = tt.addptr %48, %64 : !tt.ptr<f32>, i32
        %66 = tt.splat %65 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %67 = tt.addptr %49, %64 : !tt.ptr<f32>, i32
        %68 = tt.splat %67 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %69:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
          %70 = arith.muli %arg13, %c128_i32 : i32
          %71 = tt.splat %70 : i32 -> tensor<128xi32>
          %72 = arith.addi %71, %2 : tensor<128xi32>
          %73 = tt.expand_dims %72 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %74 = arith.muli %73, %cst_11 : tensor<128x1xi32>
          %75 = tt.addptr %61, %74 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %76 = tt.broadcast %75 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %77 = tt.addptr %76, %6 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %78 = tt.addptr %63, %74 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %79 = tt.broadcast %78 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %80 = tt.addptr %79, %6 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %81 = tt.addptr %66, %72 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %82 = tt.addptr %68, %72 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %83 = arith.muli %73, %cst_10 : tensor<128x1xi32>
          %84 = tt.addptr %7, %83 : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
          %85 = tt.broadcast %84 : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
          %86 = tt.addptr %85, %51 : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
          %87 = arith.cmpi slt, %73, %cst_10 : tensor<128x1xi32>
          %88 = arith.cmpi slt, %72, %cst_9 : tensor<128xi32>
          %89 = tt.broadcast %87 : tensor<128x1xi1> -> tensor<128x64xi1>
          %90 = arith.andi %89, %53 : tensor<128x64xi1>
          %91 = tt.broadcast %87 : tensor<128x1xi1> -> tensor<128x128xi1>
          %92 = tt.load %77, %91, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
          %93 = tt.load %80, %91, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
          %94 = tt.load %82, %88, %cst_7 : tensor<128x!tt.ptr<f32>>
          %95 = tt.load %81, %88, %cst_7 : tensor<128x!tt.ptr<f32>>
          %96 = tt.bitcast %86 : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
          %97 = tt.load %96, %90, %cst {was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
          %98 = tt.dot %92, %54, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %99 = arith.mulf %98, %cst_5 : tensor<128x64xf32>
          %100 = arith.sitofp %97 : tensor<128x64xi8> to tensor<128x64xf32>
          %101 = arith.subf %cst_4, %100 : tensor<128x64xf32>
          %102 = arith.mulf %101, %cst_3 : tensor<128x64xf32>
          %103 = arith.subf %99, %102 : tensor<128x64xf32>
          %104 = tt.expand_dims %94 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %105 = tt.broadcast %104 : tensor<128x1xf32> -> tensor<128x64xf32>
          %106 = arith.subf %103, %105 : tensor<128x64xf32>
          %107 = math.exp %106 : tensor<128x64xf32>
          %108 = arith.truncf %107 : tensor<128x64xf32> to tensor<128x64xbf16>
          %109 = tt.trans %108 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %110 = tt.dot %109, %93, %arg14 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %111 = tt.dot %93, %55, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %112 = tt.expand_dims %95 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %113 = tt.broadcast %112 : tensor<128x1xf32> -> tensor<128x64xf32>
          %114 = arith.subf %111, %113 : tensor<128x64xf32>
          %115 = arith.mulf %107, %114 : tensor<128x64xf32>
          %116 = arith.truncf %115 : tensor<128x64xf32> to tensor<128x64xbf16>
          %117 = tt.trans %116 {order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %118 = tt.dot %117, %92, %cst_12 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %119 = arith.mulf %118, %cst_2 : tensor<64x128xf32>
          %120 = arith.addf %arg15, %119 : tensor<64x128xf32>
          scf.yield %110, %120 : tensor<64x128xf32>, tensor<64x128xf32>
        }
        scf.yield %69#0, %69#1 : tensor<64x128xf32>, tensor<64x128xf32>
      }
      %57 = arith.truncf %56#1 : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %33, %57, %41 : tensor<64x128x!tt.ptr<bf16>>
      %58 = arith.truncf %56#0 : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %39, %58, %41 : tensor<64x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_q_norm_dot_false.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %5 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %6 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %7 = tt.splat %arg7 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg8 = %0 to %c160_i32 step %1  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg8, %c160_i32 : i32
// CHECK-NEXT:       %9 = arith.divsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %10 = arith.remsi %9, %c5_i32 : i32
// CHECK-NEXT:       %11 = arith.remsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %12 = arith.muli %8, %c2621440_i32 : i32
// CHECK-NEXT:       %13 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %14 = arith.muli %10, %c524288_i32 : i32
// CHECK-NEXT:       %15 = tt.addptr %13, %14 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %16 = arith.muli %11, %c128_i32 : i32
// CHECK-NEXT:       %17 = tt.splat %16 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %18 = arith.addi %17, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %19 = tt.expand_dims %18 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %20 = arith.muli %19, %cst_14 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %21 = tt.splat %15 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %22 = tt.addptr %21, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %23 = tt.broadcast %22 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %24 = tt.addptr %23, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %25 = tt.addptr %arg3, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %26 = tt.addptr %25, %14 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %28 = tt.addptr %27, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %29 = tt.broadcast %28 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %30 = tt.addptr %29, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %31 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %32 = tt.addptr %31, %14 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %33 = tt.splat %32 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %34 = tt.addptr %33, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %35 = tt.broadcast %34 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %36 = tt.addptr %35, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %37 = arith.muli %8, %c20480_i32 : i32
// CHECK-NEXT:       %38 = tt.addptr %arg4, %37 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %39 = arith.muli %10, %c4096_i32 : i32
// CHECK-NEXT:       %40 = tt.addptr %38, %39 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %41 = tt.splat %40 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %42 = tt.addptr %41, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %43 = tt.addptr %arg6, %37 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %44 = tt.addptr %43, %39 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %45 = tt.splat %44 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %46 = tt.addptr %45, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %47 = arith.cmpi slt, %19, %cst_13 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %48 = arith.cmpi slt, %18, %cst_12 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %49 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %50 = tt.load %24, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %51 = tt.load %30, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %52 = tt.load %46, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %53 = tt.load %42, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %54 = arith.muli %8, %c524288_i32 : i32
// CHECK-NEXT:       %55 = tt.addptr %arg1, %54 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %56 = arith.divsi %10, %c5_i32 : i32
// CHECK-NEXT:       %57 = arith.muli %56, %c524288_i32 : i32
// CHECK-NEXT:       %58 = tt.addptr %55, %57 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %59 = tt.splat %58 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %60 = tt.addptr %arg2, %54 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %61 = tt.addptr %60, %57 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %62 = tt.splat %61 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %63 = arith.muli %19, %cst_13 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %64 = tt.addptr %7, %63 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %65 = tt.broadcast %64 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:       %66 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %67 = tt.expand_dims %52 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %68 = tt.broadcast %67 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %69 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %70 = tt.broadcast %69 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %71 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_11) -> (tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %73 = arith.muli %arg9, %c64_i32 : i32
// CHECK-NEXT:         %74 = tt.splat %73 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %75 = arith.addi %74, %5 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %76 = tt.expand_dims %75 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %77 = arith.muli %76, %cst_9 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %78 = tt.addptr %59, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %79 = tt.broadcast %78 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %80 = tt.addptr %79, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %81 = tt.addptr %62, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %82 = tt.broadcast %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %83 = tt.addptr %82, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %84 = tt.expand_dims %75 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:         %85 = tt.broadcast %84 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:         %86 = tt.addptr %65, %85 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:         %87 = arith.cmpi slt, %76, %cst_8 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %88 = arith.cmpi slt, %84, %cst_7 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:         %89 = tt.broadcast %88 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:         %90 = arith.andi %66, %89 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:         %91 = tt.broadcast %87 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:         %92 = tt.load %80, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %93 = tt.load %83, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %94 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %95 = tt.load %94, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %96 = tt.trans %92 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %97 = tt.dot %50, %96, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %98 = arith.mulf %97, %cst_5 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %99 = arith.sitofp %95 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:         %100 = arith.subf %cst_4, %99 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %101 = arith.mulf %100, %cst_3 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %102 = arith.subf %98, %101 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %103 = arith.subf %102, %68 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %104 = math.exp %103 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %105 = tt.trans %93 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %106 = tt.dot %51, %105, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %107 = arith.subf %106, %70 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %108 = arith.mulf %104, %107 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %109 = arith.truncf %108 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:         %110 = tt.dot %109, %92, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %111 = arith.mulf %110, %cst_2 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %112 = arith.addf %arg10, %111 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %112 : tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %72 = arith.truncf %71 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       tt.store %36, %72, %49 : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_2 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant dense<4096> : tensor<1x64xi32>
    %cst_8 = arith.constant dense<4096> : tensor<64x1xi32>
    %cst_9 = arith.constant dense<128> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_12 = arith.constant dense<4096> : tensor<128xi32>
    %cst_13 = arith.constant dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_14 = arith.constant dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.broadcast %3 : tensor<1x128xi32> -> tensor<64x128xi32>
    %7 = tt.splat %arg7 : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg8 = %0 to %c160_i32 step %1  : i32 {
      %8 = arith.divsi %arg8, %c160_i32 : i32
      %9 = arith.divsi %arg8, %c32_i32 : i32
      %10 = arith.remsi %9, %c5_i32 : i32
      %11 = arith.remsi %arg8, %c32_i32 : i32
      %12 = arith.muli %8, %c2621440_i32 : i32
      %13 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
      %14 = arith.muli %10, %c524288_i32 : i32
      %15 = tt.addptr %13, %14 : !tt.ptr<bf16>, i32
      %16 = arith.muli %11, %c128_i32 : i32
      %17 = tt.splat %16 : i32 -> tensor<128xi32>
      %18 = arith.addi %17, %2 : tensor<128xi32>
      %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %20 = arith.muli %19, %cst_14 : tensor<128x1xi32>
      %21 = tt.splat %15 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %20 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %23 = tt.broadcast %22 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %24 = tt.addptr %23, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %25 = tt.addptr %arg3, %12 : !tt.ptr<bf16>, i32
      %26 = tt.addptr %25, %14 : !tt.ptr<bf16>, i32
      %27 = tt.splat %26 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %20 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %29 = tt.broadcast %28 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %30 = tt.addptr %29, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %31 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
      %32 = tt.addptr %31, %14 : !tt.ptr<bf16>, i32
      %33 = tt.splat %32 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %34 = tt.addptr %33, %20 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %35 = tt.broadcast %34 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %36 = tt.addptr %35, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %37 = arith.muli %8, %c20480_i32 : i32
      %38 = tt.addptr %arg4, %37 : !tt.ptr<f32>, i32
      %39 = arith.muli %10, %c4096_i32 : i32
      %40 = tt.addptr %38, %39 : !tt.ptr<f32>, i32
      %41 = tt.splat %40 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %42 = tt.addptr %41, %18 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %43 = tt.addptr %arg6, %37 : !tt.ptr<f32>, i32
      %44 = tt.addptr %43, %39 : !tt.ptr<f32>, i32
      %45 = tt.splat %44 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %46 = tt.addptr %45, %18 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %47 = arith.cmpi slt, %19, %cst_13 : tensor<128x1xi32>
      %48 = arith.cmpi slt, %18, %cst_12 : tensor<128xi32>
      %49 = tt.broadcast %47 : tensor<128x1xi1> -> tensor<128x128xi1>
      %50 = tt.load %24, %49, %cst_1 : tensor<128x128x!tt.ptr<bf16>>
      %51 = tt.load %30, %49, %cst_1 : tensor<128x128x!tt.ptr<bf16>>
      %52 = tt.load %46, %48, %cst_10 : tensor<128x!tt.ptr<f32>>
      %53 = tt.load %42, %48, %cst_10 : tensor<128x!tt.ptr<f32>>
      %54 = arith.muli %8, %c524288_i32 : i32
      %55 = tt.addptr %arg1, %54 : !tt.ptr<bf16>, i32
      %56 = arith.divsi %10, %c5_i32 : i32
      %57 = arith.muli %56, %c524288_i32 : i32
      %58 = tt.addptr %55, %57 : !tt.ptr<bf16>, i32
      %59 = tt.splat %58 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %60 = tt.addptr %arg2, %54 : !tt.ptr<bf16>, i32
      %61 = tt.addptr %60, %57 : !tt.ptr<bf16>, i32
      %62 = tt.splat %61 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %63 = arith.muli %19, %cst_13 : tensor<128x1xi32>
      %64 = tt.addptr %7, %63 : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %65 = tt.broadcast %64 : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
      %66 = tt.broadcast %47 : tensor<128x1xi1> -> tensor<128x64xi1>
      %67 = tt.expand_dims %52 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %68 = tt.broadcast %67 : tensor<128x1xf32> -> tensor<128x64xf32>
      %69 = tt.expand_dims %53 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %70 = tt.broadcast %69 : tensor<128x1xf32> -> tensor<128x64xf32>
      %71 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_11) -> (tensor<128x128xf32>)  : i32 {
        %73 = arith.muli %arg9, %c64_i32 : i32
        %74 = tt.splat %73 : i32 -> tensor<64xi32>
        %75 = arith.addi %74, %5 : tensor<64xi32>
        %76 = tt.expand_dims %75 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %77 = arith.muli %76, %cst_9 : tensor<64x1xi32>
        %78 = tt.addptr %59, %77 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %79 = tt.broadcast %78 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %80 = tt.addptr %79, %6 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %81 = tt.addptr %62, %77 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %82 = tt.broadcast %81 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %83 = tt.addptr %82, %6 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %84 = tt.expand_dims %75 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %85 = tt.broadcast %84 : tensor<1x64xi32> -> tensor<128x64xi32>
        %86 = tt.addptr %65, %85 : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
        %87 = arith.cmpi slt, %76, %cst_8 : tensor<64x1xi32>
        %88 = arith.cmpi slt, %84, %cst_7 : tensor<1x64xi32>
        %89 = tt.broadcast %88 : tensor<1x64xi1> -> tensor<128x64xi1>
        %90 = arith.andi %66, %89 : tensor<128x64xi1>
        %91 = tt.broadcast %87 : tensor<64x1xi1> -> tensor<64x128xi1>
        %92 = tt.load %80, %91, %cst_0 : tensor<64x128x!tt.ptr<bf16>>
        %93 = tt.load %83, %91, %cst_0 : tensor<64x128x!tt.ptr<bf16>>
        %94 = tt.bitcast %86 : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
        %95 = tt.load %94, %90, %cst {was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
        %96 = tt.trans %92 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %97 = tt.dot %50, %96, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %98 = arith.mulf %97, %cst_5 : tensor<128x64xf32>
        %99 = arith.sitofp %95 : tensor<128x64xi8> to tensor<128x64xf32>
        %100 = arith.subf %cst_4, %99 : tensor<128x64xf32>
        %101 = arith.mulf %100, %cst_3 : tensor<128x64xf32>
        %102 = arith.subf %98, %101 : tensor<128x64xf32>
        %103 = arith.subf %102, %68 : tensor<128x64xf32>
        %104 = math.exp %103 : tensor<128x64xf32>
        %105 = tt.trans %93 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %106 = tt.dot %51, %105, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %107 = arith.subf %106, %70 : tensor<128x64xf32>
        %108 = arith.mulf %104, %107 : tensor<128x64xf32>
        %109 = arith.truncf %108 : tensor<128x64xf32> to tensor<128x64xbf16>
        %110 = tt.dot %109, %92, %cst_11 {triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
        %111 = arith.mulf %110, %cst_2 : tensor<128x128xf32>
        %112 = arith.addf %arg10, %111 : tensor<128x128xf32>
        scf.yield %112 : tensor<128x128xf32>
      }
      %72 = arith.truncf %71 : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %36, %72, %49 : tensor<128x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_q_norm_dot_true.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %5 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %6 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %7 = tt.splat %arg7 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg8 = %0 to %c160_i32 step %1  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg8, %c160_i32 : i32
// CHECK-NEXT:       %9 = arith.divsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %10 = arith.remsi %9, %c5_i32 : i32
// CHECK-NEXT:       %11 = arith.remsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %12 = arith.muli %8, %c2621440_i32 : i32
// CHECK-NEXT:       %13 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %14 = arith.muli %10, %c524288_i32 : i32
// CHECK-NEXT:       %15 = tt.addptr %13, %14 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %16 = arith.muli %11, %c128_i32 : i32
// CHECK-NEXT:       %17 = tt.splat %16 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %18 = arith.addi %17, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %19 = tt.expand_dims %18 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %20 = arith.muli %19, %cst_14 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %21 = tt.splat %15 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %22 = tt.addptr %21, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %23 = tt.broadcast %22 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %24 = tt.addptr %23, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %25 = tt.addptr %arg3, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %26 = tt.addptr %25, %14 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %28 = tt.addptr %27, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %29 = tt.broadcast %28 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %30 = tt.addptr %29, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %31 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %32 = tt.addptr %31, %14 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %33 = tt.splat %32 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %34 = tt.addptr %33, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %35 = tt.broadcast %34 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %36 = tt.addptr %35, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %37 = arith.muli %8, %c20480_i32 : i32
// CHECK-NEXT:       %38 = tt.addptr %arg4, %37 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %39 = arith.muli %10, %c4096_i32 : i32
// CHECK-NEXT:       %40 = tt.addptr %38, %39 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %41 = tt.splat %40 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %42 = tt.addptr %41, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %43 = tt.addptr %arg6, %37 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %44 = tt.addptr %43, %39 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %45 = tt.splat %44 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %46 = tt.addptr %45, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %47 = arith.cmpi slt, %19, %cst_13 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %48 = arith.cmpi slt, %18, %cst_12 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %49 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %50 = tt.load %24, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %51 = tt.load %30, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %52 = tt.load %46, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %53 = tt.load %42, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %54 = arith.muli %8, %c524288_i32 : i32
// CHECK-NEXT:       %55 = tt.addptr %arg1, %54 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %56 = arith.divsi %10, %c5_i32 : i32
// CHECK-NEXT:       %57 = arith.muli %56, %c524288_i32 : i32
// CHECK-NEXT:       %58 = tt.addptr %55, %57 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %59 = tt.splat %58 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %60 = tt.addptr %arg2, %54 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %61 = tt.addptr %60, %57 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %62 = tt.splat %61 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %63 = arith.muli %19, %cst_13 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %64 = tt.addptr %7, %63 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %65 = tt.broadcast %64 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:       %66 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %67 = tt.expand_dims %52 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %68 = tt.broadcast %67 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %69 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %70 = tt.broadcast %69 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %71 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_11) -> (tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %73 = arith.muli %arg9, %c64_i32 : i32
// CHECK-NEXT:         %74 = tt.splat %73 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %75 = arith.addi %74, %5 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %76 = tt.expand_dims %75 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %77 = arith.muli %76, %cst_9 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %78 = tt.addptr %59, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %79 = tt.broadcast %78 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %80 = tt.addptr %79, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %81 = tt.addptr %62, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %82 = tt.broadcast %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %83 = tt.addptr %82, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %84 = tt.expand_dims %75 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:         %85 = tt.broadcast %84 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:         %86 = tt.addptr %65, %85 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:         %87 = arith.cmpi slt, %76, %cst_8 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %88 = arith.cmpi slt, %84, %cst_7 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:         %89 = tt.broadcast %88 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:         %90 = arith.andi %66, %89 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:         %91 = tt.broadcast %87 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:         %92 = tt.load %80, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %93 = tt.load %83, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %94 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %95 = tt.load %94, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %96 = tt.trans %92 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %97 = tt.dot %50, %96, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %98 = arith.mulf %97, %cst_5 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %99 = arith.sitofp %95 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:         %100 = arith.subf %cst_4, %99 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %101 = arith.mulf %100, %cst_3 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %102 = arith.subf %98, %101 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %103 = arith.subf %102, %68 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %104 = math.exp %103 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %105 = tt.trans %93 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %106 = tt.dot %51, %105, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %107 = arith.subf %106, %70 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %108 = arith.mulf %104, %107 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %109 = arith.truncf %108 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:         %110 = tt.dot %109, %92, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %111 = arith.mulf %110, %cst_2 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %112 = arith.addf %arg10, %111 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %112 : tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %72 = arith.truncf %71 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       tt.store %36, %72, %49 : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_2 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant dense<4096> : tensor<1x64xi32>
    %cst_8 = arith.constant dense<4096> : tensor<64x1xi32>
    %cst_9 = arith.constant dense<128> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_12 = arith.constant dense<4096> : tensor<128xi32>
    %cst_13 = arith.constant dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_14 = arith.constant dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.broadcast %3 : tensor<1x128xi32> -> tensor<64x128xi32>
    %7 = tt.splat %arg7 : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg8 = %0 to %c160_i32 step %1  : i32 {
      %8 = arith.divsi %arg8, %c160_i32 : i32
      %9 = arith.divsi %arg8, %c32_i32 : i32
      %10 = arith.remsi %9, %c5_i32 : i32
      %11 = arith.remsi %arg8, %c32_i32 : i32
      %12 = arith.muli %8, %c2621440_i32 : i32
      %13 = tt.addptr %arg0, %12 : !tt.ptr<bf16>, i32
      %14 = arith.muli %10, %c524288_i32 : i32
      %15 = tt.addptr %13, %14 : !tt.ptr<bf16>, i32
      %16 = arith.muli %11, %c128_i32 : i32
      %17 = tt.splat %16 : i32 -> tensor<128xi32>
      %18 = arith.addi %17, %2 : tensor<128xi32>
      %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %20 = arith.muli %19, %cst_14 : tensor<128x1xi32>
      %21 = tt.splat %15 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %20 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %23 = tt.broadcast %22 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %24 = tt.addptr %23, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %25 = tt.addptr %arg3, %12 : !tt.ptr<bf16>, i32
      %26 = tt.addptr %25, %14 : !tt.ptr<bf16>, i32
      %27 = tt.splat %26 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %20 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %29 = tt.broadcast %28 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %30 = tt.addptr %29, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %31 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
      %32 = tt.addptr %31, %14 : !tt.ptr<bf16>, i32
      %33 = tt.splat %32 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %34 = tt.addptr %33, %20 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %35 = tt.broadcast %34 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %36 = tt.addptr %35, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %37 = arith.muli %8, %c20480_i32 : i32
      %38 = tt.addptr %arg4, %37 : !tt.ptr<f32>, i32
      %39 = arith.muli %10, %c4096_i32 : i32
      %40 = tt.addptr %38, %39 : !tt.ptr<f32>, i32
      %41 = tt.splat %40 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %42 = tt.addptr %41, %18 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %43 = tt.addptr %arg6, %37 : !tt.ptr<f32>, i32
      %44 = tt.addptr %43, %39 : !tt.ptr<f32>, i32
      %45 = tt.splat %44 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %46 = tt.addptr %45, %18 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %47 = arith.cmpi slt, %19, %cst_13 : tensor<128x1xi32>
      %48 = arith.cmpi slt, %18, %cst_12 : tensor<128xi32>
      %49 = tt.broadcast %47 : tensor<128x1xi1> -> tensor<128x128xi1>
      %50 = tt.load %24, %49, %cst_1 : tensor<128x128x!tt.ptr<bf16>>
      %51 = tt.load %30, %49, %cst_1 : tensor<128x128x!tt.ptr<bf16>>
      %52 = tt.load %46, %48, %cst_10 : tensor<128x!tt.ptr<f32>>
      %53 = tt.load %42, %48, %cst_10 : tensor<128x!tt.ptr<f32>>
      %54 = arith.muli %8, %c524288_i32 : i32
      %55 = tt.addptr %arg1, %54 : !tt.ptr<bf16>, i32
      %56 = arith.divsi %10, %c5_i32 : i32
      %57 = arith.muli %56, %c524288_i32 : i32
      %58 = tt.addptr %55, %57 : !tt.ptr<bf16>, i32
      %59 = tt.splat %58 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %60 = tt.addptr %arg2, %54 : !tt.ptr<bf16>, i32
      %61 = tt.addptr %60, %57 : !tt.ptr<bf16>, i32
      %62 = tt.splat %61 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %63 = arith.muli %19, %cst_13 : tensor<128x1xi32>
      %64 = tt.addptr %7, %63 : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %65 = tt.broadcast %64 : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
      %66 = tt.broadcast %47 : tensor<128x1xi1> -> tensor<128x64xi1>
      %67 = tt.expand_dims %52 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %68 = tt.broadcast %67 : tensor<128x1xf32> -> tensor<128x64xf32>
      %69 = tt.expand_dims %53 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %70 = tt.broadcast %69 : tensor<128x1xf32> -> tensor<128x64xf32>
      %71 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_11) -> (tensor<128x128xf32>)  : i32 {
        %73 = arith.muli %arg9, %c64_i32 : i32
        %74 = tt.splat %73 : i32 -> tensor<64xi32>
        %75 = arith.addi %74, %5 : tensor<64xi32>
        %76 = tt.expand_dims %75 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %77 = arith.muli %76, %cst_9 : tensor<64x1xi32>
        %78 = tt.addptr %59, %77 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %79 = tt.broadcast %78 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %80 = tt.addptr %79, %6 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %81 = tt.addptr %62, %77 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %82 = tt.broadcast %81 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %83 = tt.addptr %82, %6 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %84 = tt.expand_dims %75 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %85 = tt.broadcast %84 : tensor<1x64xi32> -> tensor<128x64xi32>
        %86 = tt.addptr %65, %85 : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
        %87 = arith.cmpi slt, %76, %cst_8 : tensor<64x1xi32>
        %88 = arith.cmpi slt, %84, %cst_7 : tensor<1x64xi32>
        %89 = tt.broadcast %88 : tensor<1x64xi1> -> tensor<128x64xi1>
        %90 = arith.andi %66, %89 : tensor<128x64xi1>
        %91 = tt.broadcast %87 : tensor<64x1xi1> -> tensor<64x128xi1>
        %92 = tt.load %80, %91, %cst_0 : tensor<64x128x!tt.ptr<bf16>>
        %93 = tt.load %83, %91, %cst_0 : tensor<64x128x!tt.ptr<bf16>>
        %94 = tt.bitcast %86 : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
        %95 = tt.load %94, %90, %cst {was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
        %96 = tt.trans %92 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %97 = tt.dot %50, %96, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %98 = arith.mulf %97, %cst_5 : tensor<128x64xf32>
        %99 = arith.sitofp %95 : tensor<128x64xi8> to tensor<128x64xf32>
        %100 = arith.subf %cst_4, %99 : tensor<128x64xf32>
        %101 = arith.mulf %100, %cst_3 : tensor<128x64xf32>
        %102 = arith.subf %98, %101 : tensor<128x64xf32>
        %103 = arith.subf %102, %68 : tensor<128x64xf32>
        %104 = math.exp %103 : tensor<128x64xf32>
        %105 = tt.trans %93 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %106 = tt.dot %51, %105, %cst_6 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %107 = arith.subf %106, %70 : tensor<128x64xf32>
        %108 = arith.mulf %104, %107 : tensor<128x64xf32>
        %109 = arith.truncf %108 : tensor<128x64xf32> to tensor<128x64xbf16>
        %110 = tt.dot %109, %92, %cst_11 {triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
        %111 = arith.mulf %110, %cst_2 : tensor<128x128xf32>
        %112 = arith.addf %arg10, %111 : tensor<128x128xf32>
        scf.yield %112 : tensor<128x128xf32>
      }
      %72 = arith.truncf %71 : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %36, %72, %49 : tensor<128x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/sdpa_fwd/norm_dot_false.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<4096> : tensor<1x128xi32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<128xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %5 = tt.splat %arg5 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg6 = %0 to %c160_i32 step %1  : i32 {
// CHECK-NEXT:       %6 = arith.divsi %arg6, %c160_i32 : i32
// CHECK-NEXT:       %7 = arith.divsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %8 = arith.remsi %7, %c5_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %10 = arith.muli %6, %c2621440_i32 : i32
// CHECK-NEXT:       %11 = tt.addptr %arg0, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %12 = arith.muli %8, %c524288_i32 : i32
// CHECK-NEXT:       %13 = tt.addptr %11, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %14 = arith.muli %9, %c128_i32 : i32
// CHECK-NEXT:       %15 = tt.splat %14 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %16 = arith.addi %15, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %17 = tt.expand_dims %16 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %18 = arith.muli %17, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %19 = tt.splat %13 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %20 = tt.addptr %19, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %21 = tt.broadcast %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %22 = tt.addptr %21, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %24 = tt.addptr %23, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %26 = tt.addptr %25, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %27 = tt.broadcast %26 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %28 = tt.addptr %27, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %29 = arith.muli %6, %c20480_i32 : i32
// CHECK-NEXT:       %30 = tt.addptr %arg4, %29 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %31 = arith.muli %8, %c4096_i32 : i32
// CHECK-NEXT:       %32 = tt.addptr %30, %31 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %33 = tt.splat %32 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %34 = tt.addptr %33, %16 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %35 = arith.cmpi slt, %17, %cst_9 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %36 = arith.cmpi slt, %16, %cst_8 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %37 = tt.broadcast %35 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %38 = tt.load %22, %37, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %39 = arith.muli %6, %c524288_i32 : i32
// CHECK-NEXT:       %40 = tt.addptr %arg1, %39 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %41 = arith.divsi %8, %c5_i32 : i32
// CHECK-NEXT:       %42 = arith.muli %41, %c524288_i32 : i32
// CHECK-NEXT:       %43 = tt.addptr %40, %42 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %44 = tt.splat %43 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %45 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %46 = tt.addptr %45, %42 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %47 = tt.splat %46 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %48 = arith.muli %17, %cst_9 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %49 = tt.addptr %5, %48 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %50 = tt.broadcast %49 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
// CHECK-NEXT:       %51:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_7, %arg9 = %cst_5, %arg10 = %cst_6) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
// CHECK-NEXT:         %58 = arith.muli %arg7, %c128_i32 : i32
// CHECK-NEXT:         %59 = tt.splat %58 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %60 = arith.addi %59, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %61 = tt.expand_dims %60 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %62 = arith.muli %61, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %63 = tt.addptr %44, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %65 = tt.addptr %64, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %66 = tt.addptr %47, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %67 = tt.broadcast %66 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %68 = tt.addptr %67, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %69 = tt.expand_dims %60 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:         %70 = tt.broadcast %69 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:         %71 = tt.addptr %50, %70 {MetaUse} : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
// CHECK-NEXT:         %72 = arith.cmpi slt, %61, %cst_9 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %73 = arith.cmpi slt, %69, %cst_4 {MetaUse} : tensor<1x128xi32>
// CHECK-NEXT:         %74 = tt.broadcast %73 {MetaUse} : tensor<1x128xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %75 = arith.andi %37, %74 {MetaUse} : tensor<128x128xi1>
// CHECK-NEXT:         %76 = tt.bitcast %71 {MetaUse} : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %77 = tt.load %76, %75, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %78 = tt.broadcast %72 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %79 = tt.load %65, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %80 = tt.load %68, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %81 = tt.trans %79 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %82 = tt.dot %38, %81, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %83 = arith.mulf %82, %cst_3 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %84 = arith.sitofp %77 {DataUse} : tensor<128x128xi8> to tensor<128x128xf32>
// CHECK-NEXT:         %85 = arith.subf %cst_2, %84 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %86 = arith.mulf %85, %cst_1 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %87 = arith.subf %83, %86 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %88 = "tt.reduce"(%87) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %105 = arith.maximumf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %89 = arith.maximumf %arg9, %88 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %90 = tt.expand_dims %89 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.subf %87, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = math.exp %92 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %94 = arith.subf %arg9, %89 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %95 = math.exp %94 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %96 = arith.mulf %95, %arg10 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %97 = "tt.reduce"(%93) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %105 = arith.addf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %98 = arith.addf %96, %97 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %99 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %101 = arith.mulf %100, %arg8 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %102 = arith.truncf %93 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %103 = tt.dot %102, %80, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %104 = arith.addf %103, %101 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %104, %89, %98 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %52 = tt.expand_dims %51#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %53 = tt.broadcast %52 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %54 = arith.divf %51#0, %53 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %55 = math.log %51#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %56 = arith.addf %55, %51#1 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %57 = arith.truncf %54 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       tt.store %28, %57, %37 : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       tt.store %34, %56, %36 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<128x128xi8>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_1 = arith.constant dense<1.000000e+06> : tensor<128x128xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant dense<4096> : tensor<1x128xi32>
    %cst_5 = arith.constant dense<-1.000000e+06> : tensor<128xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_8 = arith.constant dense<4096> : tensor<128xi32>
    %cst_9 = arith.constant dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_10 = arith.constant dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.splat %arg5 : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg6 = %0 to %c160_i32 step %1  : i32 {
      %6 = arith.divsi %arg6, %c160_i32 : i32
      %7 = arith.divsi %arg6, %c32_i32 : i32
      %8 = arith.remsi %7, %c5_i32 : i32
      %9 = arith.remsi %arg6, %c32_i32 : i32
      %10 = arith.muli %6, %c2621440_i32 : i32
      %11 = tt.addptr %arg0, %10 : !tt.ptr<bf16>, i32
      %12 = arith.muli %8, %c524288_i32 : i32
      %13 = tt.addptr %11, %12 : !tt.ptr<bf16>, i32
      %14 = arith.muli %9, %c128_i32 : i32
      %15 = tt.splat %14 : i32 -> tensor<128xi32>
      %16 = arith.addi %15, %2 : tensor<128xi32>
      %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %18 = arith.muli %17, %cst_10 : tensor<128x1xi32>
      %19 = tt.splat %13 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %20 = tt.addptr %19, %18 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %21 = tt.broadcast %20 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
      %24 = tt.addptr %23, %12 : !tt.ptr<bf16>, i32
      %25 = tt.splat %24 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %26 = tt.addptr %25, %18 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %27 = tt.broadcast %26 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %29 = arith.muli %6, %c20480_i32 : i32
      %30 = tt.addptr %arg4, %29 : !tt.ptr<f32>, i32
      %31 = arith.muli %8, %c4096_i32 : i32
      %32 = tt.addptr %30, %31 : !tt.ptr<f32>, i32
      %33 = tt.splat %32 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %34 = tt.addptr %33, %16 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %35 = arith.cmpi slt, %17, %cst_9 : tensor<128x1xi32>
      %36 = arith.cmpi slt, %16, %cst_8 : tensor<128xi32>
      %37 = tt.broadcast %35 : tensor<128x1xi1> -> tensor<128x128xi1>
      %38 = tt.load %22, %37, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
      %39 = arith.muli %6, %c524288_i32 : i32
      %40 = tt.addptr %arg1, %39 : !tt.ptr<bf16>, i32
      %41 = arith.divsi %8, %c5_i32 : i32
      %42 = arith.muli %41, %c524288_i32 : i32
      %43 = tt.addptr %40, %42 : !tt.ptr<bf16>, i32
      %44 = tt.splat %43 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %45 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %45, %42 : !tt.ptr<bf16>, i32
      %47 = tt.splat %46 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %48 = arith.muli %17, %cst_9 : tensor<128x1xi32>
      %49 = tt.addptr %5, %48 : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %50 = tt.broadcast %49 : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
      %51:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_7, %arg9 = %cst_5, %arg10 = %cst_6) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
        %58 = arith.muli %arg7, %c128_i32 : i32
        %59 = tt.splat %58 : i32 -> tensor<128xi32>
        %60 = arith.addi %59, %2 : tensor<128xi32>
        %61 = tt.expand_dims %60 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %62 = arith.muli %61, %cst_10 : tensor<128x1xi32>
        %63 = tt.addptr %44, %62 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %64 = tt.broadcast %63 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %66 = tt.addptr %47, %62 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %67 = tt.broadcast %66 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %68 = tt.addptr %67, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %69 = tt.expand_dims %60 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %70 = tt.broadcast %69 : tensor<1x128xi32> -> tensor<128x128xi32>
        %71 = tt.addptr %50, %70 : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
        %72 = arith.cmpi slt, %61, %cst_9 : tensor<128x1xi32>
        %73 = arith.cmpi slt, %69, %cst_4 : tensor<1x128xi32>
        %74 = tt.broadcast %73 : tensor<1x128xi1> -> tensor<128x128xi1>
        %75 = arith.andi %37, %74 : tensor<128x128xi1>
        %76 = tt.bitcast %71 : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
        %77 = tt.load %76, %75, %cst {was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
        %78 = tt.broadcast %72 : tensor<128x1xi1> -> tensor<128x128xi1>
        %79 = tt.load %65, %78, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
        %80 = tt.load %68, %78, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
        %81 = tt.trans %79 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %82 = tt.dot %38, %81, %cst_7 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %83 = arith.mulf %82, %cst_3 : tensor<128x128xf32>
        %84 = arith.sitofp %77 : tensor<128x128xi8> to tensor<128x128xf32>
        %85 = arith.subf %cst_2, %84 : tensor<128x128xf32>
        %86 = arith.mulf %85, %cst_1 : tensor<128x128xf32>
        %87 = arith.subf %83, %86 : tensor<128x128xf32>
        %88 = "tt.reduce"(%87) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.maximumf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %89 = arith.maximumf %arg9, %88 : tensor<128xf32>
        %90 = tt.expand_dims %89 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.subf %87, %91 : tensor<128x128xf32>
        %93 = math.exp %92 : tensor<128x128xf32>
        %94 = arith.subf %arg9, %89 : tensor<128xf32>
        %95 = math.exp %94 : tensor<128xf32>
        %96 = arith.mulf %95, %arg10 : tensor<128xf32>
        %97 = "tt.reduce"(%93) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.addf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %98 = arith.addf %96, %97 : tensor<128xf32>
        %99 = tt.expand_dims %95 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %100 = tt.broadcast %99 : tensor<128x1xf32> -> tensor<128x128xf32>
        %101 = arith.mulf %100, %arg8 : tensor<128x128xf32>
        %102 = arith.truncf %93 : tensor<128x128xf32> to tensor<128x128xbf16>
        %103 = tt.dot %102, %80, %cst_7 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %104 = arith.addf %103, %101 {triton_cv12.add_from_dot} : tensor<128x128xf32>
        scf.yield %104, %89, %98 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
      }
      %52 = tt.expand_dims %51#2 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %53 = tt.broadcast %52 : tensor<128x1xf32> -> tensor<128x128xf32>
      %54 = arith.divf %51#0, %53 : tensor<128x128xf32>
      %55 = math.log %51#2 : tensor<128xf32>
      %56 = arith.addf %55, %51#1 : tensor<128xf32>
      %57 = arith.truncf %54 : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %28, %57, %37 : tensor<128x128x!tt.ptr<bf16>>
      tt.store %34, %56, %36 : tensor<128x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/sdpa_fwd/norm_dot_true.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<4096> : tensor<1x128xi32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<128xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %5 = tt.splat %arg5 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg6 = %0 to %c160_i32 step %1  : i32 {
// CHECK-NEXT:       %6 = arith.divsi %arg6, %c160_i32 : i32
// CHECK-NEXT:       %7 = arith.divsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %8 = arith.remsi %7, %c5_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %10 = arith.muli %6, %c2621440_i32 : i32
// CHECK-NEXT:       %11 = tt.addptr %arg0, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %12 = arith.muli %8, %c524288_i32 : i32
// CHECK-NEXT:       %13 = tt.addptr %11, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %14 = arith.muli %9, %c128_i32 : i32
// CHECK-NEXT:       %15 = tt.splat %14 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %16 = arith.addi %15, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %17 = tt.expand_dims %16 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %18 = arith.muli %17, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %19 = tt.splat %13 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %20 = tt.addptr %19, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %21 = tt.broadcast %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %22 = tt.addptr %21, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %24 = tt.addptr %23, %12 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %26 = tt.addptr %25, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %27 = tt.broadcast %26 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %28 = tt.addptr %27, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %29 = arith.muli %6, %c20480_i32 : i32
// CHECK-NEXT:       %30 = tt.addptr %arg4, %29 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %31 = arith.muli %8, %c4096_i32 : i32
// CHECK-NEXT:       %32 = tt.addptr %30, %31 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %33 = tt.splat %32 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %34 = tt.addptr %33, %16 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %35 = arith.cmpi slt, %17, %cst_9 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %36 = arith.cmpi slt, %16, %cst_8 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %37 = tt.broadcast %35 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %38 = tt.load %22, %37, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %39 = arith.muli %6, %c524288_i32 : i32
// CHECK-NEXT:       %40 = tt.addptr %arg1, %39 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %41 = arith.divsi %8, %c5_i32 : i32
// CHECK-NEXT:       %42 = arith.muli %41, %c524288_i32 : i32
// CHECK-NEXT:       %43 = tt.addptr %40, %42 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %44 = tt.splat %43 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %45 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %46 = tt.addptr %45, %42 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %47 = tt.splat %46 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %48 = arith.muli %17, %cst_9 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %49 = tt.addptr %5, %48 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %50 = tt.broadcast %49 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
// CHECK-NEXT:       %51:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_7, %arg9 = %cst_5, %arg10 = %cst_6) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
// CHECK-NEXT:         %58 = arith.muli %arg7, %c128_i32 : i32
// CHECK-NEXT:         %59 = tt.splat %58 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %60 = arith.addi %59, %2 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %61 = tt.expand_dims %60 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %62 = arith.muli %61, %cst_10 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %63 = tt.addptr %44, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %65 = tt.addptr %64, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %66 = tt.addptr %47, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %67 = tt.broadcast %66 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %68 = tt.addptr %67, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %69 = tt.expand_dims %60 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:         %70 = tt.broadcast %69 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:         %71 = tt.addptr %50, %70 {MetaUse} : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
// CHECK-NEXT:         %72 = arith.cmpi slt, %61, %cst_9 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %73 = arith.cmpi slt, %69, %cst_4 {MetaUse} : tensor<1x128xi32>
// CHECK-NEXT:         %74 = tt.broadcast %73 {MetaUse} : tensor<1x128xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %75 = arith.andi %37, %74 {MetaUse} : tensor<128x128xi1>
// CHECK-NEXT:         %76 = tt.bitcast %71 {MetaUse} : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %77 = tt.load %76, %75, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %78 = tt.broadcast %72 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %79 = tt.load %65, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %80 = tt.load %68, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %81 = tt.trans %79 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %82 = tt.dot %38, %81, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %83 = arith.mulf %82, %cst_3 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %84 = arith.sitofp %77 {DataUse} : tensor<128x128xi8> to tensor<128x128xf32>
// CHECK-NEXT:         %85 = arith.subf %cst_2, %84 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %86 = arith.mulf %85, %cst_1 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %87 = arith.subf %83, %86 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %88 = "tt.reduce"(%87) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %105 = arith.maximumf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %89 = arith.maximumf %arg9, %88 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %90 = tt.expand_dims %89 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.subf %87, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = math.exp %92 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %94 = arith.subf %arg9, %89 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %95 = math.exp %94 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %96 = arith.mulf %95, %arg10 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %97 = "tt.reduce"(%93) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %105 = arith.addf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %98 = arith.addf %96, %97 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %99 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %101 = arith.mulf %100, %arg8 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %102 = arith.truncf %93 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %103 = tt.dot %102, %80, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %104 = arith.addf %103, %101 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %104, %89, %98 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %52 = tt.expand_dims %51#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %53 = tt.broadcast %52 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %54 = arith.divf %51#0, %53 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %55 = math.log %51#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %56 = arith.addf %55, %51#1 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %57 = arith.truncf %54 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       tt.store %28, %57, %37 : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       tt.store %34, %56, %36 : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<128x128xi8>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_1 = arith.constant dense<1.000000e+06> : tensor<128x128xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_3 = arith.constant dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant dense<4096> : tensor<1x128xi32>
    %cst_5 = arith.constant dense<-1.000000e+06> : tensor<128xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_8 = arith.constant dense<4096> : tensor<128xi32>
    %cst_9 = arith.constant dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_10 = arith.constant dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.splat %arg5 : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg6 = %0 to %c160_i32 step %1  : i32 {
      %6 = arith.divsi %arg6, %c160_i32 : i32
      %7 = arith.divsi %arg6, %c32_i32 : i32
      %8 = arith.remsi %7, %c5_i32 : i32
      %9 = arith.remsi %arg6, %c32_i32 : i32
      %10 = arith.muli %6, %c2621440_i32 : i32
      %11 = tt.addptr %arg0, %10 : !tt.ptr<bf16>, i32
      %12 = arith.muli %8, %c524288_i32 : i32
      %13 = tt.addptr %11, %12 : !tt.ptr<bf16>, i32
      %14 = arith.muli %9, %c128_i32 : i32
      %15 = tt.splat %14 : i32 -> tensor<128xi32>
      %16 = arith.addi %15, %2 : tensor<128xi32>
      %17 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %18 = arith.muli %17, %cst_10 : tensor<128x1xi32>
      %19 = tt.splat %13 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %20 = tt.addptr %19, %18 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %21 = tt.broadcast %20 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
      %24 = tt.addptr %23, %12 : !tt.ptr<bf16>, i32
      %25 = tt.splat %24 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %26 = tt.addptr %25, %18 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %27 = tt.broadcast %26 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %29 = arith.muli %6, %c20480_i32 : i32
      %30 = tt.addptr %arg4, %29 : !tt.ptr<f32>, i32
      %31 = arith.muli %8, %c4096_i32 : i32
      %32 = tt.addptr %30, %31 : !tt.ptr<f32>, i32
      %33 = tt.splat %32 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %34 = tt.addptr %33, %16 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %35 = arith.cmpi slt, %17, %cst_9 : tensor<128x1xi32>
      %36 = arith.cmpi slt, %16, %cst_8 : tensor<128xi32>
      %37 = tt.broadcast %35 : tensor<128x1xi1> -> tensor<128x128xi1>
      %38 = tt.load %22, %37, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
      %39 = arith.muli %6, %c524288_i32 : i32
      %40 = tt.addptr %arg1, %39 : !tt.ptr<bf16>, i32
      %41 = arith.divsi %8, %c5_i32 : i32
      %42 = arith.muli %41, %c524288_i32 : i32
      %43 = tt.addptr %40, %42 : !tt.ptr<bf16>, i32
      %44 = tt.splat %43 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %45 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %45, %42 : !tt.ptr<bf16>, i32
      %47 = tt.splat %46 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %48 = arith.muli %17, %cst_9 : tensor<128x1xi32>
      %49 = tt.addptr %5, %48 : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %50 = tt.broadcast %49 : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
      %51:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_7, %arg9 = %cst_5, %arg10 = %cst_6) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
        %58 = arith.muli %arg7, %c128_i32 : i32
        %59 = tt.splat %58 : i32 -> tensor<128xi32>
        %60 = arith.addi %59, %2 : tensor<128xi32>
        %61 = tt.expand_dims %60 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %62 = arith.muli %61, %cst_10 : tensor<128x1xi32>
        %63 = tt.addptr %44, %62 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %64 = tt.broadcast %63 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %66 = tt.addptr %47, %62 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %67 = tt.broadcast %66 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %68 = tt.addptr %67, %4 : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %69 = tt.expand_dims %60 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %70 = tt.broadcast %69 : tensor<1x128xi32> -> tensor<128x128xi32>
        %71 = tt.addptr %50, %70 : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
        %72 = arith.cmpi slt, %61, %cst_9 : tensor<128x1xi32>
        %73 = arith.cmpi slt, %69, %cst_4 : tensor<1x128xi32>
        %74 = tt.broadcast %73 : tensor<1x128xi1> -> tensor<128x128xi1>
        %75 = arith.andi %37, %74 : tensor<128x128xi1>
        %76 = tt.bitcast %71 : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
        %77 = tt.load %76, %75, %cst {was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
        %78 = tt.broadcast %72 : tensor<128x1xi1> -> tensor<128x128xi1>
        %79 = tt.load %65, %78, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
        %80 = tt.load %68, %78, %cst_0 : tensor<128x128x!tt.ptr<bf16>>
        %81 = tt.trans %79 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %82 = tt.dot %38, %81, %cst_7 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %83 = arith.mulf %82, %cst_3 : tensor<128x128xf32>
        %84 = arith.sitofp %77 : tensor<128x128xi8> to tensor<128x128xf32>
        %85 = arith.subf %cst_2, %84 : tensor<128x128xf32>
        %86 = arith.mulf %85, %cst_1 : tensor<128x128xf32>
        %87 = arith.subf %83, %86 : tensor<128x128xf32>
        %88 = "tt.reduce"(%87) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.maximumf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %89 = arith.maximumf %arg9, %88 : tensor<128xf32>
        %90 = tt.expand_dims %89 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.subf %87, %91 : tensor<128x128xf32>
        %93 = math.exp %92 : tensor<128x128xf32>
        %94 = arith.subf %arg9, %89 : tensor<128xf32>
        %95 = math.exp %94 : tensor<128xf32>
        %96 = arith.mulf %95, %arg10 : tensor<128xf32>
        %97 = "tt.reduce"(%93) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.addf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) : (tensor<128x128xf32>) -> tensor<128xf32>
        %98 = arith.addf %96, %97 : tensor<128xf32>
        %99 = tt.expand_dims %95 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %100 = tt.broadcast %99 : tensor<128x1xf32> -> tensor<128x128xf32>
        %101 = arith.mulf %100, %arg8 : tensor<128x128xf32>
        %102 = arith.truncf %93 : tensor<128x128xf32> to tensor<128x128xbf16>
        %103 = tt.dot %102, %80, %cst_7 {triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %104 = arith.addf %103, %101 {triton_cv12.add_from_dot} : tensor<128x128xf32>
        scf.yield %104, %89, %98 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
      }
      %52 = tt.expand_dims %51#2 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %53 = tt.broadcast %52 : tensor<128x1xf32> -> tensor<128x128xf32>
      %54 = arith.divf %51#0, %53 : tensor<128x128xf32>
      %55 = math.log %51#2 : tensor<128xf32>
      %56 = arith.addf %55, %51#1 : tensor<128xf32>
      %57 = arith.truncf %54 : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %28, %57, %37 : tensor<128x128x!tt.ptr<bf16>>
      tt.store %34, %56, %36 : tensor<128x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/sdpa_infer/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_sdpa_infer_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<64xf32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0> : tensor<64x64xi8>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<64x64xf32>
// CHECK-NEXT:     %c262144_i32 = arith.constant 262144 : i32
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c320_i32 = arith.constant 320 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c524288_i64 = arith.constant 524288 : i64
// CHECK-NEXT:     %c2621440_i64 = arith.constant 2621440 : i64
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = tt.get_num_programs x : i32
// CHECK-NEXT:     %2 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %3 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %4 = arith.muli %3, %cst_1 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %5 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %6 = tt.broadcast %5 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %7 = tt.splat %arg6 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     scf.for %arg7 = %0 to %c320_i32 step %1  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg7, %c64_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg7, %c64_i32 : i32
// CHECK-NEXT:       %10 = arith.divsi %8, %c5_i32 : i32
// CHECK-NEXT:       %11 = arith.remsi %8, %c5_i32 : i32
// CHECK-NEXT:       %12 = arith.divsi %11, %c5_i32 : i32
// CHECK-NEXT:       %13 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %14 = arith.muli %13, %c2621440_i64 : i64
// CHECK-NEXT:       %15 = arith.extsi %11 : i32 to i64
// CHECK-NEXT:       %16 = arith.muli %15, %c524288_i64 : i64
// CHECK-NEXT:       %17 = arith.addi %14, %16 : i64
// CHECK-NEXT:       %18 = arith.muli %13, %c524288_i64 : i64
// CHECK-NEXT:       %19 = arith.extsi %12 : i32 to i64
// CHECK-NEXT:       %20 = arith.muli %19, %c524288_i64 : i64
// CHECK-NEXT:       %21 = arith.addi %18, %20 : i64
// CHECK-NEXT:       %22 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %23 = arith.muli %9, %c64_i32 : i32
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %22, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %25 = tt.addptr %arg1, %21 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %26 = tt.make_tensor_ptr %25, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %27 = tt.addptr %arg2, %21 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %28 = tt.make_tensor_ptr %27, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %29 = tt.addptr %arg5, %17 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %30 = tt.make_tensor_ptr %29, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %31 = tt.load %24 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %32 = arith.muli %9, %c262144_i32 : i32
// CHECK-NEXT:       %33 = tt.addptr %arg3, %32 : !tt.ptr<i1>, i32
// CHECK-NEXT:       %34:5 = scf.for %arg8 = %c0_i32 to %c4096_i32 step %c64_i32 iter_args(%arg9 = %cst, %arg10 = %cst_5, %arg11 = %cst_0, %arg12 = %28, %arg13 = %26) -> (tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>)  : i32 {
// CHECK-NEXT:         %39 = tt.addptr %33, %arg8 : !tt.ptr<i1>, i32
// CHECK-NEXT:         %40 = tt.splat %39 {MetaUse} : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
// CHECK-NEXT:         %41 = tt.addptr %40, %4 {MetaUse} : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
// CHECK-NEXT:         %42 = tt.broadcast %41 {MetaUse} : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
// CHECK-NEXT:         %43 = tt.addptr %42, %6 {MetaUse} : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
// CHECK-NEXT:         %44 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:         %45 = tt.trans %44 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %46 = tt.dot %31, %45, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %47 = arith.mulf %46, %7 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %48 = tt.bitcast %43 {MetaUse} : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:         %49 = tt.load %48 {DataUse, was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:         %50 = arith.cmpi ne, %49, %cst_3 {DataUse} : tensor<64x64xi8>
// CHECK-NEXT:         %51 = arith.select %50, %47, %cst_4 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:         %52 = "tt.reduce"(%51) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg14: f32, %arg15: f32):
// CHECK-NEXT:           %72 = arith.maximumf %arg14, %arg15 : f32
// CHECK-NEXT:           tt.reduce.return %72 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %53 = arith.maximumf %arg11, %52 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %54 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %55 = tt.broadcast %54 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:         %56 = arith.subf %51, %55 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %57 = math.exp %56 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %58 = arith.truncf %57 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:         %59 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:         %60 = "tt.reduce"(%57) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg14: f32, %arg15: f32):
// CHECK-NEXT:           %72 = arith.addf %arg14, %arg15 : f32
// CHECK-NEXT:           tt.reduce.return %72 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %61 = arith.subf %arg11, %53 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %62 = math.exp %61 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %63 = arith.mulf %arg9, %62 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %64 = arith.addf %63, %60 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %65 = tt.expand_dims %62 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %66 = tt.broadcast %65 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %67 = arith.mulf %arg10, %66 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %68 = tt.dot %58, %59, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %69 = arith.addf %68, %67 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:         annotation.mark %69 {hivm.tile_mix_cube_num = 2 : i32} : tensor<64x128xf32>
// CHECK-NEXT:         %70 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
// CHECK-NEXT:         %71 = tt.advance %arg13, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
// CHECK-NEXT:         scf.yield %64, %69, %53, %70, %71 : tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
// CHECK-NEXT:       %35 = tt.expand_dims %34#0 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %36 = tt.broadcast %35 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:       %37 = arith.divf %34#1, %36 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:       %38 = arith.truncf %37 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %30, %38 : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_sdpa_infer_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32>
    %c4096_i32 = arith.constant 4096 : i32
    %cst_1 = arith.constant dense<4096> : tensor<64x1xi32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_3 = arith.constant dense<0> : tensor<64x64xi8>
    %cst_4 = arith.constant dense<-1.000000e+06> : tensor<64x64xf32>
    %c262144_i32 = arith.constant 262144 : i32
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %c320_i32 = arith.constant 320 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c524288_i64 = arith.constant 524288 : i64
    %c2621440_i64 = arith.constant 2621440 : i64
    %c5_i32 = arith.constant 5 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<64x1xi32>
    %5 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %6 = tt.broadcast %5 : tensor<1x64xi32> -> tensor<64x64xi32>
    %7 = tt.splat %arg6 : f32 -> tensor<64x64xf32>
    scf.for %arg7 = %0 to %c320_i32 step %1  : i32 {
      %8 = arith.divsi %arg7, %c64_i32 : i32
      %9 = arith.remsi %arg7, %c64_i32 : i32
      %10 = arith.divsi %8, %c5_i32 : i32
      %11 = arith.remsi %8, %c5_i32 : i32
      %12 = arith.divsi %11, %c5_i32 : i32
      %13 = arith.extsi %10 : i32 to i64
      %14 = arith.muli %13, %c2621440_i64 : i64
      %15 = arith.extsi %11 : i32 to i64
      %16 = arith.muli %15, %c524288_i64 : i64
      %17 = arith.addi %14, %16 : i64
      %18 = arith.muli %13, %c524288_i64 : i64
      %19 = arith.extsi %12 : i32 to i64
      %20 = arith.muli %19, %c524288_i64 : i64
      %21 = arith.addi %18, %20 : i64
      %22 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i64
      %23 = arith.muli %9, %c64_i32 : i32
      %24 = tt.make_tensor_ptr %22, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %25 = tt.addptr %arg1, %21 : !tt.ptr<bf16>, i64
      %26 = tt.make_tensor_ptr %25, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %27 = tt.addptr %arg2, %21 : !tt.ptr<bf16>, i64
      %28 = tt.make_tensor_ptr %27, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %29 = tt.addptr %arg5, %17 : !tt.ptr<bf16>, i64
      %30 = tt.make_tensor_ptr %29, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %31 = tt.load %24 : !tt.ptr<tensor<64x128xbf16>>
      %32 = arith.muli %9, %c262144_i32 : i32
      %33 = tt.addptr %arg3, %32 : !tt.ptr<i1>, i32
      %34:5 = scf.for %arg8 = %c0_i32 to %c4096_i32 step %c64_i32 iter_args(%arg9 = %cst, %arg10 = %cst_5, %arg11 = %cst_0, %arg12 = %28, %arg13 = %26) -> (tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>)  : i32 {
        %39 = tt.addptr %33, %arg8 : !tt.ptr<i1>, i32
        %40 = tt.splat %39 : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
        %41 = tt.addptr %40, %4 : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
        %42 = tt.broadcast %41 : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
        %43 = tt.addptr %42, %6 : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
        %44 = tt.load %arg13 : !tt.ptr<tensor<64x128xbf16>>
        %45 = tt.trans %44 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %46 = tt.dot %31, %45, %cst_2 {triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
        %47 = arith.mulf %46, %7 : tensor<64x64xf32>
        %48 = tt.bitcast %43 : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
        %49 = tt.load %48 {was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
        %50 = arith.cmpi ne, %49, %cst_3 : tensor<64x64xi8>
        %51 = arith.select %50, %47, %cst_4 : tensor<64x64xi1>, tensor<64x64xf32>
        %52 = "tt.reduce"(%51) <{axis = 1 : i32}> ({
        ^bb0(%arg14: f32, %arg15: f32):
          %72 = arith.maximumf %arg14, %arg15 : f32
          tt.reduce.return %72 : f32
        }) : (tensor<64x64xf32>) -> tensor<64xf32>
        %53 = arith.maximumf %arg11, %52 : tensor<64xf32>
        %54 = tt.expand_dims %53 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %55 = tt.broadcast %54 : tensor<64x1xf32> -> tensor<64x64xf32>
        %56 = arith.subf %51, %55 : tensor<64x64xf32>
        %57 = math.exp %56 : tensor<64x64xf32>
        %58 = arith.truncf %57 : tensor<64x64xf32> to tensor<64x64xbf16>
        %59 = tt.load %arg12 : !tt.ptr<tensor<64x128xbf16>>
        %60 = "tt.reduce"(%57) <{axis = 1 : i32}> ({
        ^bb0(%arg14: f32, %arg15: f32):
          %72 = arith.addf %arg14, %arg15 : f32
          tt.reduce.return %72 : f32
        }) : (tensor<64x64xf32>) -> tensor<64xf32>
        %61 = arith.subf %arg11, %53 : tensor<64xf32>
        %62 = math.exp %61 : tensor<64xf32>
        %63 = arith.mulf %arg9, %62 : tensor<64xf32>
        %64 = arith.addf %63, %60 : tensor<64xf32>
        %65 = tt.expand_dims %62 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %66 = tt.broadcast %65 : tensor<64x1xf32> -> tensor<64x128xf32>
        %67 = arith.mulf %arg10, %66 : tensor<64x128xf32>
        %68 = tt.dot %58, %59, %cst_5 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
        %69 = arith.addf %68, %67 {triton_cv12.add_from_dot} : tensor<64x128xf32>
        annotation.mark %69 {hivm.tile_mix_cube_num = 2 : i32} : tensor<64x128xf32>
        %70 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
        %71 = tt.advance %arg13, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
        scf.yield %64, %69, %53, %70, %71 : tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
      %35 = tt.expand_dims %34#0 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %36 = tt.broadcast %35 : tensor<64x1xf32> -> tensor<64x128xf32>
      %37 = arith.divf %34#1, %36 : tensor<64x128xf32>
      %38 = arith.truncf %37 : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %30, %38 : !tt.ptr<tensor<64x128xbf16>>
    }
    tt.return
  }
}

// -----
// Source: gaoyou/0701/simplicial_kv1/norm_dot.mlir
// CHECK: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @simplicial_bwd_kv1_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: i32 {tt.divisibility = 16 : i32}, %arg43: i32 {tt.divisibility = 16 : i32}, %arg44: i32 {tt.divisibility = 16 : i32}, %arg45: i32 {tt.divisibility = 16 : i32}, %arg46: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<1.250000e-01> : tensor<64x32xbf16>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x64xbf16>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x32xbf16>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.250000e-01> : tensor<64x64xbf16>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x64xbf16>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x32xf32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %0 = tt.get_program_id x : i32
// CHECK-NEXT:     %1 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:     %2 = arith.addi %1, %c64_i32 : i32
// CHECK-NEXT:     %3 = tt.get_program_id y : i32
// CHECK-NEXT:     %4 = arith.divsi %3, %arg12 : i32
// CHECK-NEXT:     %5 = arith.remsi %3, %arg12 : i32
// CHECK-NEXT:     %6 = arith.muli %4, %arg15 : i32
// CHECK-NEXT:     %7 = arith.muli %5, %arg17 : i32
// CHECK-NEXT:     %8 = arith.addi %6, %7 : i32
// CHECK-NEXT:     %9 = tt.addptr %arg0, %8 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %10 = tt.addptr %arg1, %8 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %11 = tt.addptr %arg2, %8 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %12 = tt.addptr %arg3, %8 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %13 = tt.addptr %arg4, %8 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %14 = arith.muli %4, %arg30 : i32
// CHECK-NEXT:     %15 = arith.muli %5, %arg32 : i32
// CHECK-NEXT:     %16 = arith.addi %14, %15 : i32
// CHECK-NEXT:     %17 = tt.addptr %arg5, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %18 = arith.muli %4, %arg33 : i32
// CHECK-NEXT:     %19 = arith.muli %5, %arg34 : i32
// CHECK-NEXT:     %20 = arith.addi %18, %19 : i32
// CHECK-NEXT:     %21 = tt.addptr %arg6, %20 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %22 = arith.muli %4, %arg35 : i32
// CHECK-NEXT:     %23 = arith.muli %5, %arg36 : i32
// CHECK-NEXT:     %24 = arith.addi %22, %23 : i32
// CHECK-NEXT:     %25 = tt.addptr %arg7, %24 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %26 = arith.muli %4, %arg41 : i32
// CHECK-NEXT:     %27 = arith.muli %5, %arg43 : i32
// CHECK-NEXT:     %28 = arith.addi %26, %27 : i32
// CHECK-NEXT:     %29 = tt.addptr %arg8, %28 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %30 = arith.muli %4, %arg44 : i32
// CHECK-NEXT:     %31 = arith.muli %5, %arg46 : i32
// CHECK-NEXT:     %32 = arith.addi %30, %31 : i32
// CHECK-NEXT:     %33 = tt.addptr %arg9, %32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %34 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %35 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %36 = tt.splat %1 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %37 = tt.splat %1 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %38 = arith.addi %36, %34 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %39 = arith.addi %37, %35 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %40 = tt.expand_dims %38 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %41 = tt.expand_dims %39 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %42 = tt.splat %arg19 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %43 = arith.muli %40, %42 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %44 = tt.expand_dims %34 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %45 = tt.broadcast %43 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %46 = tt.broadcast %44 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %47 = arith.addi %45, %46 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:     %48 = tt.splat %arg11 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %49 = arith.cmpi slt, %40, %48 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %50 = tt.splat %10 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %51 = tt.addptr %50, %47 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:     %52 = tt.broadcast %49 {MetaUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %53 = tt.load %51, %52, %cst_4 {DataUse} : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %54 = tt.splat %arg25 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %55 = arith.muli %40, %54 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %56 = tt.broadcast %55 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %57 = arith.addi %56, %46 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:     %58 = tt.splat %12 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %59 = tt.addptr %58, %57 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:     %60 = tt.load %59, %52, %cst_4 {DataUse} : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %61 = arith.subi %1, %arg14 : i32
// CHECK-NEXT:     %62 = arith.maxsi %61, %c0_i32 : i32
// CHECK-NEXT:     %63 = arith.addi %2, %arg13 : i32
// CHECK-NEXT:     %64 = arith.minsi %arg11, %63 : i32
// CHECK-NEXT:     %65 = tt.splat %11 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:     %66 = tt.splat %13 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:     %67 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %68 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %69 = tt.splat %arg16 {MetaUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:     %70 = tt.expand_dims %34 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %71 = tt.broadcast %70 {MetaUse} : tensor<64x1xi32> -> tensor<64x32xi32>
// CHECK-NEXT:     %72 = tt.splat %arg11 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:     %73 = tt.splat %9 {MetaUse} : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
// CHECK-NEXT:     %74 = tt.splat %21 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:     %75 = tt.splat %25 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:     %76 = tt.splat %arg31 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:     %77 = tt.broadcast %44 {MetaUse} : tensor<1x64xi32> -> tensor<32x64xi32>
// CHECK-NEXT:     %78 = tt.splat %17 {MetaUse} : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %79 = tt.splat %arg13 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:     %80 = tt.broadcast %41 {DataUse} : tensor<64x1xi32> -> tensor<64x32xi32>
// CHECK-NEXT:     %81 = tt.splat %arg14 {DataUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %c0_i32, %alloca[] : memref<i32>
// CHECK-NEXT:     %alloca_7 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %c0_i32, %alloca_7[] : memref<i32>
// CHECK-NEXT:     %82:2 = scf.for %arg47 = %62 to %64 step %c1_i32 iter_args(%arg48 = %cst, %arg49 = %cst) -> (tensor<64x64xf32>, tensor<64x64xf32>)  : i32 {
// CHECK-NEXT:       %103 = arith.muli %arg47, %arg22 : i32
// CHECK-NEXT:       %104 = tt.splat %103 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %105 = arith.addi %104, %34 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %106 = tt.addptr %65, %105 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %107 = tt.load %106 {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %108 = tt.expand_dims %107 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:       %109 = arith.muli %arg47, %arg28 : i32
// CHECK-NEXT:       %110 = tt.splat %109 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %111 = arith.addi %110, %34 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %112 = tt.addptr %66, %111 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %113 = tt.load %112 {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %114 = tt.expand_dims %113 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:       %115 = tt.broadcast %108 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %116 = arith.mulf %53, %115 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:       %117 = tt.broadcast %114 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %118 = arith.mulf %60, %117 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:       %119 = arith.mulf %116, %cst_3 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:       %120 = arith.maxsi %1, %arg47 : i32
// CHECK-NEXT:       %121 = arith.addi %arg47, %arg14 : i32
// CHECK-NEXT:       %122 = arith.minsi %63, %121 : i32
// CHECK-NEXT:       %123 = arith.minsi %arg11, %122 : i32
// CHECK-NEXT:       %124 = tt.splat %arg47 {DataUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:       %125:2 = scf.for %arg50 = %120 to %123 step %c32_i32 iter_args(%arg51 = %arg48, %arg52 = %arg49) -> (tensor<64x64xf32>, tensor<64x64xf32>)  : i32 {
// CHECK-NEXT:         %126 = tt.splat %arg50 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %127 = tt.splat %arg50 {DataUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %128 = arith.addi %126, %67 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %129 = arith.addi %127, %68 {DataUse} : tensor<32xi32>
// CHECK-NEXT:         %130 = tt.expand_dims %128 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:         %131 = tt.expand_dims %129 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:         %132 = arith.muli %130, %69 {MetaUse} : tensor<1x32xi32>
// CHECK-NEXT:         %133 = tt.broadcast %132 {MetaUse} : tensor<1x32xi32> -> tensor<64x32xi32>
// CHECK-NEXT:         %134 = arith.addi %133, %71 {MetaUse} : tensor<64x32xi32>
// CHECK-NEXT:         %135 = arith.cmpi slt, %128, %72 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %136 = tt.expand_dims %135 {MetaUse, axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
// CHECK-NEXT:         %137 = tt.addptr %73, %134 {MetaUse} : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
// CHECK-NEXT:         %138 = tt.broadcast %136 {MetaUse} : tensor<1x32xi1> -> tensor<64x32xi1>
// CHECK-NEXT:         %139 = tt.load %137, %138, %cst_2 {DataUse} : tensor<64x32x!tt.ptr<bf16>>
// CHECK-NEXT:         %140 = tt.addptr %74, %128 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %141 = tt.load %140, %135, %cst_5 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %142 = tt.expand_dims %141 {DataUse, axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
// CHECK-NEXT:         %143 = tt.addptr %75, %128 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %144 = tt.load %143, %135, %cst_5 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %145 = tt.expand_dims %144 {DataUse, axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
// CHECK-NEXT:         %146 = tt.expand_dims %128 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %147 = arith.muli %146, %76 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %148 = tt.broadcast %147 {MetaUse} : tensor<32x1xi32> -> tensor<32x64xi32>
// CHECK-NEXT:         %149 = arith.addi %148, %77 {MetaUse} : tensor<32x64xi32>
// CHECK-NEXT:         %150 = tt.expand_dims %135 {MetaUse, axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
// CHECK-NEXT:         %151 = tt.addptr %78, %149 {MetaUse} : tensor<32x64x!tt.ptr<bf16>>, tensor<32x64xi32>
// CHECK-NEXT:         %152 = tt.broadcast %150 {MetaUse} : tensor<32x1xi1> -> tensor<32x64xi1>
// CHECK-NEXT:         %153 = tt.load %151, %152, %cst_1 {DataUse} : tensor<32x64x!tt.ptr<bf16>>
// CHECK-NEXT:         %154 = tt.dot %119, %139, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:         %155 = arith.subi %131, %79 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %156 = tt.broadcast %155 {DataUse} : tensor<1x32xi32> -> tensor<64x32xi32>
// CHECK-NEXT:         %157 = arith.cmpi slt, %156, %80 {DataUse} : tensor<64x32xi32>
// CHECK-NEXT:         %158 = tt.broadcast %131 {DataUse} : tensor<1x32xi32> -> tensor<64x32xi32>
// CHECK-NEXT:         %159 = arith.cmpi sle, %80, %158 {DataUse} : tensor<64x32xi32>
// CHECK-NEXT:         %160 = arith.andi %157, %159 {DataUse} : tensor<64x32xi1>
// CHECK-NEXT:         %161 = arith.subi %129, %81 {DataUse} : tensor<32xi32>
// CHECK-NEXT:         %162 = arith.cmpi slt, %161, %124 {DataUse} : tensor<32xi32>
// CHECK-NEXT:         %163 = arith.cmpi sle, %124, %129 {DataUse} : tensor<32xi32>
// CHECK-NEXT:         %164 = arith.andi %162, %163 {DataUse} : tensor<32xi1>
// CHECK-NEXT:         %165 = tt.expand_dims %164 {DataUse, axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
// CHECK-NEXT:         %166 = tt.broadcast %165 {DataUse} : tensor<1x32xi1> -> tensor<64x32xi1>
// CHECK-NEXT:         %167 = arith.andi %160, %166 {DataUse} : tensor<64x32xi1>
// CHECK-NEXT:         %168 = tt.reshape %167 allow_reorder {DataUse} : tensor<64x32xi1> -> tensor<2048xi1>
// CHECK-NEXT:         %169 = "tt.reduce"(%168) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg53: i1, %arg54: i1):
// CHECK-NEXT:           %173 = arith.addi %arg53, %arg54 : i1
// CHECK-NEXT:           tt.reduce.return %173 : i1
// CHECK-NEXT:         }) : (tensor<2048xi1>) -> i1
// CHECK-NEXT:         %170 = arith.extui %169 : i1 to i32
// CHECK-NEXT:         %171 = arith.cmpi sgt, %170, %c0_i32 : i32
// CHECK-NEXT:         %172:2 = scf.if %171 -> (tensor<64x64xf32>, tensor<64x64xf32>) {
// CHECK-NEXT:           %173 = tt.broadcast %142 {DataUse} : tensor<1x32xf32> -> tensor<64x32xf32>
// CHECK-NEXT:           %174 = arith.subf %154, %173 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:           %175 = math.exp %174 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:           %176 = arith.select %167, %175, %cst_6 {DataUse} : tensor<64x32xi1>, tensor<64x32xf32>
// CHECK-NEXT:           %177 = tt.broadcast %114 {DataUse} : tensor<1x64xbf16> -> tensor<32x64xbf16>
// CHECK-NEXT:           %178 = arith.mulf %153, %177 {DataUse} : tensor<32x64xbf16>
// CHECK-NEXT:           %179 = arith.truncf %176 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:           %180 = tt.dot %179, %178, %arg51 {DataUse, triton_cv12.normalized_dot} : tensor<64x32xbf16> * tensor<32x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:           %181 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:           %182 = arith.addi %181, %c1_i32 : i32
// CHECK-NEXT:           memref.store %182, %alloca[] : memref<i32>
// CHECK-NEXT:           %183 = tt.trans %153 {DataUse, order = array<i32: 1, 0>} : tensor<32x64xbf16> -> tensor<64x32xbf16>
// CHECK-NEXT:           %184 = tt.dot %118, %183, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:           %185 = arith.subf %cst_6, %176 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:           %186 = tt.broadcast %145 {DataUse} : tensor<1x32xf32> -> tensor<64x32xf32>
// CHECK-NEXT:           %187 = arith.mulf %185, %186 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:           %188 = math.fma %176, %184, %187 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:           %189 = arith.truncf %188 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:           %190 = arith.mulf %189, %cst_0 {DataUse} : tensor<64x32xbf16>
// CHECK-NEXT:           %191 = tt.trans %139 {DataUse, order = array<i32: 1, 0>} : tensor<64x32xbf16> -> tensor<32x64xbf16>
// CHECK-NEXT:           %192 = tt.broadcast %108 {DataUse} : tensor<1x64xbf16> -> tensor<32x64xbf16>
// CHECK-NEXT:           %193 = arith.mulf %191, %192 {DataUse} : tensor<32x64xbf16>
// CHECK-NEXT:           %194 = tt.dot %190, %193, %arg52 {DataUse, triton_cv12.normalized_dot} : tensor<64x32xbf16> * tensor<32x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:           %195 = memref.load %alloca_7[] : memref<i32>
// CHECK-NEXT:           %196 = arith.addi %195, %c1_i32 : i32
// CHECK-NEXT:           memref.store %196, %alloca_7[] : memref<i32>
// CHECK-NEXT:           scf.yield %180, %194 : tensor<64x64xf32>, tensor<64x64xf32>
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %arg51, %arg52 : tensor<64x64xf32>, tensor<64x64xf32>
// CHECK-NEXT:         } {DataUse, hivm.matmul_limited_in_cube}
// CHECK-NEXT:         scf.yield %172#0, %172#1 : tensor<64x64xf32>, tensor<64x64xf32>
// CHECK-NEXT:       } {DataUse, tt.num_stages = 1 : i32}
// CHECK-NEXT:       scf.yield %125#0, %125#1 : tensor<64x64xf32>, tensor<64x64xf32>
// CHECK-NEXT:     } {DataUse, hivm.matmul_limited_in_cube}
// CHECK-NEXT:     %83 = memref.load %alloca_7[] : memref<i32>
// CHECK-NEXT:     %84 = arith.cmpi ne, %83, %c0_i32 : i32
// CHECK-NEXT:     %85 = scf.if %84 -> (tensor<64x64xf32>) {
// CHECK-NEXT:       scf.yield %82#1 : tensor<64x64xf32>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %cst : tensor<64x64xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %86 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %87 = arith.cmpi ne, %86, %c0_i32 : i32
// CHECK-NEXT:     %88 = scf.if %87 -> (tensor<64x64xf32>) {
// CHECK-NEXT:       scf.yield %82#0 : tensor<64x64xf32>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %cst : tensor<64x64xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %89 = tt.splat %arg45 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %90 = arith.muli %40, %89 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %91 = tt.broadcast %90 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %92 = arith.addi %91, %46 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:     %93 = tt.splat %arg42 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:     %94 = arith.muli %40, %93 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %95 = tt.broadcast %94 {MetaUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %96 = arith.addi %95, %46 {MetaUse} : tensor<64x64xi32>
// CHECK-NEXT:     %97 = tt.splat %33 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %98 = tt.addptr %97, %92 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:     %99 = arith.truncf %88 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     tt.store %98, %99, %52 : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %100 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     %101 = tt.addptr %100, %96 {MetaUse} : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
// CHECK-NEXT:     %102 = arith.truncf %85 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     tt.store %101, %102, %52 : tensor<64x64x!tt.ptr<bf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @simplicial_bwd_kv1_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32 {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: i32 {tt.divisibility = 16 : i32}, %arg41: i32 {tt.divisibility = 16 : i32}, %arg42: i32 {tt.divisibility = 16 : i32}, %arg43: i32 {tt.divisibility = 16 : i32}, %arg44: i32 {tt.divisibility = 16 : i32}, %arg45: i32 {tt.divisibility = 16 : i32}, %arg46: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_0 = arith.constant dense<1.250000e-01> : tensor<64x32xbf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x32xbf16>
    %c32_i32 = arith.constant 32 : i32
    %cst_3 = arith.constant dense<1.250000e-01> : tensor<64x64xbf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64x64xbf16>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = arith.addi %1, %c64_i32 : i32
    %3 = tt.get_program_id y : i32
    %4 = arith.divsi %3, %arg12 : i32
    %5 = arith.remsi %3, %arg12 : i32
    %6 = arith.muli %4, %arg15 : i32
    %7 = arith.muli %5, %arg17 : i32
    %8 = arith.addi %6, %7 : i32
    %9 = tt.addptr %arg0, %8 : !tt.ptr<bf16>, i32
    %10 = tt.addptr %arg1, %8 : !tt.ptr<bf16>, i32
    %11 = tt.addptr %arg2, %8 : !tt.ptr<bf16>, i32
    %12 = tt.addptr %arg3, %8 : !tt.ptr<bf16>, i32
    %13 = tt.addptr %arg4, %8 : !tt.ptr<bf16>, i32
    %14 = arith.muli %4, %arg30 : i32
    %15 = arith.muli %5, %arg32 : i32
    %16 = arith.addi %14, %15 : i32
    %17 = tt.addptr %arg5, %16 : !tt.ptr<bf16>, i32
    %18 = arith.muli %4, %arg33 : i32
    %19 = arith.muli %5, %arg34 : i32
    %20 = arith.addi %18, %19 : i32
    %21 = tt.addptr %arg6, %20 : !tt.ptr<f32>, i32
    %22 = arith.muli %4, %arg35 : i32
    %23 = arith.muli %5, %arg36 : i32
    %24 = arith.addi %22, %23 : i32
    %25 = tt.addptr %arg7, %24 : !tt.ptr<f32>, i32
    %26 = arith.muli %4, %arg41 : i32
    %27 = arith.muli %5, %arg43 : i32
    %28 = arith.addi %26, %27 : i32
    %29 = tt.addptr %arg8, %28 : !tt.ptr<bf16>, i32
    %30 = arith.muli %4, %arg44 : i32
    %31 = arith.muli %5, %arg46 : i32
    %32 = arith.addi %30, %31 : i32
    %33 = tt.addptr %arg9, %32 : !tt.ptr<bf16>, i32
    %34 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %35 = tt.splat %1 : i32 -> tensor<64xi32>
    %36 = arith.addi %35, %34 : tensor<64xi32>
    %37 = tt.expand_dims %36 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %38 = tt.splat %arg19 : i32 -> tensor<64x1xi32>
    %39 = arith.muli %37, %38 : tensor<64x1xi32>
    %40 = tt.expand_dims %34 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %41 = tt.broadcast %39 : tensor<64x1xi32> -> tensor<64x64xi32>
    %42 = tt.broadcast %40 : tensor<1x64xi32> -> tensor<64x64xi32>
    %43 = arith.addi %41, %42 : tensor<64x64xi32>
    %44 = tt.splat %arg11 : i32 -> tensor<64x1xi32>
    %45 = arith.cmpi slt, %37, %44 : tensor<64x1xi32>
    %46 = tt.splat %10 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %47 = tt.addptr %46, %43 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %48 = tt.broadcast %45 : tensor<64x1xi1> -> tensor<64x64xi1>
    %49 = tt.load %47, %48, %cst_4 : tensor<64x64x!tt.ptr<bf16>>
    %50 = tt.splat %arg25 : i32 -> tensor<64x1xi32>
    %51 = arith.muli %37, %50 : tensor<64x1xi32>
    %52 = tt.broadcast %51 : tensor<64x1xi32> -> tensor<64x64xi32>
    %53 = arith.addi %52, %42 : tensor<64x64xi32>
    %54 = tt.splat %12 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %55 = tt.addptr %54, %53 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %56 = tt.load %55, %48, %cst_4 : tensor<64x64x!tt.ptr<bf16>>
    %57 = arith.subi %1, %arg14 : i32
    %58 = arith.maxsi %57, %c0_i32 : i32
    %59 = arith.addi %2, %arg13 : i32
    %60 = arith.minsi %arg11, %59 : i32
    %61 = tt.splat %11 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %62 = tt.splat %13 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %63 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %64 = tt.splat %arg16 : i32 -> tensor<1x32xi32>
    %65 = tt.expand_dims %34 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %66 = tt.broadcast %65 : tensor<64x1xi32> -> tensor<64x32xi32>
    %67 = tt.splat %arg11 : i32 -> tensor<32xi32>
    %68 = tt.splat %9 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %69 = tt.splat %21 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %70 = tt.splat %25 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %71 = tt.splat %arg31 : i32 -> tensor<32x1xi32>
    %72 = tt.broadcast %40 : tensor<1x64xi32> -> tensor<32x64xi32>
    %73 = tt.splat %17 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>>
    %74 = tt.splat %arg13 : i32 -> tensor<1x32xi32>
    %75 = tt.broadcast %37 : tensor<64x1xi32> -> tensor<64x32xi32>
    %76 = tt.splat %arg14 : i32 -> tensor<32xi32>
    %alloca = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca[] : memref<i32>
    %alloca_7 = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca_7[] : memref<i32>
    %77:2 = scf.for %arg47 = %58 to %60 step %c1_i32 iter_args(%arg48 = %cst, %arg49 = %cst) -> (tensor<64x64xf32>, tensor<64x64xf32>)  : i32 {
      %98 = arith.muli %arg47, %arg22 : i32
      %99 = tt.splat %98 : i32 -> tensor<64xi32>
      %100 = arith.addi %99, %34 : tensor<64xi32>
      %101 = tt.addptr %61, %100 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %102 = tt.load %101 : tensor<64x!tt.ptr<bf16>>
      %103 = tt.expand_dims %102 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
      %104 = arith.muli %arg47, %arg28 : i32
      %105 = tt.splat %104 : i32 -> tensor<64xi32>
      %106 = arith.addi %105, %34 : tensor<64xi32>
      %107 = tt.addptr %62, %106 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %108 = tt.load %107 : tensor<64x!tt.ptr<bf16>>
      %109 = tt.expand_dims %108 {axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
      %110 = tt.broadcast %103 : tensor<1x64xbf16> -> tensor<64x64xbf16>
      %111 = arith.mulf %49, %110 : tensor<64x64xbf16>
      %112 = tt.broadcast %109 : tensor<1x64xbf16> -> tensor<64x64xbf16>
      %113 = arith.mulf %56, %112 : tensor<64x64xbf16>
      %114 = arith.mulf %111, %cst_3 : tensor<64x64xbf16>
      %115 = arith.maxsi %1, %arg47 : i32
      %116 = arith.addi %arg47, %arg14 : i32
      %117 = arith.minsi %59, %116 : i32
      %118 = arith.minsi %arg11, %117 : i32
      %119 = tt.splat %arg47 : i32 -> tensor<32xi32>
      %120:2 = scf.for %arg50 = %115 to %118 step %c32_i32 iter_args(%arg51 = %arg48, %arg52 = %arg49) -> (tensor<64x64xf32>, tensor<64x64xf32>)  : i32 {
        %121 = tt.splat %arg50 : i32 -> tensor<32xi32>
        %122 = arith.addi %121, %63 : tensor<32xi32>
        %123 = tt.expand_dims %122 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
        %124 = arith.muli %123, %64 : tensor<1x32xi32>
        %125 = tt.broadcast %124 : tensor<1x32xi32> -> tensor<64x32xi32>
        %126 = arith.addi %125, %66 : tensor<64x32xi32>
        %127 = arith.cmpi slt, %122, %67 : tensor<32xi32>
        %128 = tt.expand_dims %127 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
        %129 = tt.addptr %68, %126 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
        %130 = tt.broadcast %128 : tensor<1x32xi1> -> tensor<64x32xi1>
        %131 = tt.load %129, %130, %cst_2 : tensor<64x32x!tt.ptr<bf16>>
        %132 = tt.addptr %69, %122 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %133 = tt.load %132, %127, %cst_5 : tensor<32x!tt.ptr<f32>>
        %134 = tt.expand_dims %133 {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
        %135 = tt.addptr %70, %122 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %136 = tt.load %135, %127, %cst_5 : tensor<32x!tt.ptr<f32>>
        %137 = tt.expand_dims %136 {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
        %138 = tt.expand_dims %122 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %139 = arith.muli %138, %71 : tensor<32x1xi32>
        %140 = tt.broadcast %139 : tensor<32x1xi32> -> tensor<32x64xi32>
        %141 = arith.addi %140, %72 : tensor<32x64xi32>
        %142 = tt.expand_dims %127 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
        %143 = tt.addptr %73, %141 : tensor<32x64x!tt.ptr<bf16>>, tensor<32x64xi32>
        %144 = tt.broadcast %142 : tensor<32x1xi1> -> tensor<32x64xi1>
        %145 = tt.load %143, %144, %cst_1 : tensor<32x64x!tt.ptr<bf16>>
        %146 = tt.dot %114, %131, %cst_6 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
        %147 = arith.subi %123, %74 : tensor<1x32xi32>
        %148 = tt.broadcast %147 : tensor<1x32xi32> -> tensor<64x32xi32>
        %149 = arith.cmpi slt, %148, %75 : tensor<64x32xi32>
        %150 = tt.broadcast %123 : tensor<1x32xi32> -> tensor<64x32xi32>
        %151 = arith.cmpi sle, %75, %150 : tensor<64x32xi32>
        %152 = arith.andi %149, %151 : tensor<64x32xi1>
        %153 = arith.subi %122, %76 : tensor<32xi32>
        %154 = arith.cmpi slt, %153, %119 : tensor<32xi32>
        %155 = arith.cmpi sle, %119, %122 : tensor<32xi32>
        %156 = arith.andi %154, %155 : tensor<32xi1>
        %157 = tt.expand_dims %156 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
        %158 = tt.broadcast %157 : tensor<1x32xi1> -> tensor<64x32xi1>
        %159 = arith.andi %152, %158 : tensor<64x32xi1>
        %160 = tt.reshape %159 allow_reorder : tensor<64x32xi1> -> tensor<2048xi1>
        %161 = "tt.reduce"(%160) <{axis = 0 : i32}> ({
        ^bb0(%arg53: i1, %arg54: i1):
          %165 = arith.addi %arg53, %arg54 : i1
          tt.reduce.return %165 : i1
        }) : (tensor<2048xi1>) -> i1
        %162 = arith.extui %161 : i1 to i32
        %163 = arith.cmpi sgt, %162, %c0_i32 : i32
        %164:2 = scf.if %163 -> (tensor<64x64xf32>, tensor<64x64xf32>) {
          %165 = tt.broadcast %134 : tensor<1x32xf32> -> tensor<64x32xf32>
          %166 = arith.subf %146, %165 : tensor<64x32xf32>
          %167 = math.exp %166 : tensor<64x32xf32>
          %168 = arith.select %159, %167, %cst_6 : tensor<64x32xi1>, tensor<64x32xf32>
          %169 = tt.broadcast %109 : tensor<1x64xbf16> -> tensor<32x64xbf16>
          %170 = arith.mulf %145, %169 : tensor<32x64xbf16>
          %171 = arith.truncf %168 : tensor<64x32xf32> to tensor<64x32xbf16>
          %172 = tt.dot %171, %170, %arg51 {triton_cv12.normalized_dot} : tensor<64x32xbf16> * tensor<32x64xbf16> -> tensor<64x64xf32>
          %173 = memref.load %alloca[] : memref<i32>
          %174 = arith.addi %173, %c1_i32 : i32
          memref.store %174, %alloca[] : memref<i32>
          %175 = tt.trans %145 {order = array<i32: 1, 0>} : tensor<32x64xbf16> -> tensor<64x32xbf16>
          %176 = tt.dot %113, %175, %cst_6 {triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
          %177 = arith.subf %cst_6, %168 : tensor<64x32xf32>
          %178 = tt.broadcast %137 : tensor<1x32xf32> -> tensor<64x32xf32>
          %179 = arith.mulf %177, %178 : tensor<64x32xf32>
          %180 = math.fma %168, %176, %179 : tensor<64x32xf32>
          %181 = arith.truncf %180 : tensor<64x32xf32> to tensor<64x32xbf16>
          %182 = arith.mulf %181, %cst_0 : tensor<64x32xbf16>
          %183 = tt.trans %131 {order = array<i32: 1, 0>} : tensor<64x32xbf16> -> tensor<32x64xbf16>
          %184 = tt.broadcast %103 : tensor<1x64xbf16> -> tensor<32x64xbf16>
          %185 = arith.mulf %183, %184 : tensor<32x64xbf16>
          %186 = tt.dot %182, %185, %arg52 {triton_cv12.normalized_dot} : tensor<64x32xbf16> * tensor<32x64xbf16> -> tensor<64x64xf32>
          %187 = memref.load %alloca_7[] : memref<i32>
          %188 = arith.addi %187, %c1_i32 : i32
          memref.store %188, %alloca_7[] : memref<i32>
          scf.yield %172, %186 : tensor<64x64xf32>, tensor<64x64xf32>
        } else {
          scf.yield %arg51, %arg52 : tensor<64x64xf32>, tensor<64x64xf32>
        } {hivm.matmul_limited_in_cube}
        scf.yield %164#0, %164#1 : tensor<64x64xf32>, tensor<64x64xf32>
      } {tt.num_stages = 1 : i32}
      scf.yield %120#0, %120#1 : tensor<64x64xf32>, tensor<64x64xf32>
    } {hivm.matmul_limited_in_cube}
    %78 = memref.load %alloca_7[] : memref<i32>
    %79 = arith.cmpi ne, %78, %c0_i32 : i32
    %80 = scf.if %79 -> (tensor<64x64xf32>) {
      scf.yield %77#1 : tensor<64x64xf32>
    } else {
      scf.yield %cst : tensor<64x64xf32>
    }
    %81 = memref.load %alloca[] : memref<i32>
    %82 = arith.cmpi ne, %81, %c0_i32 : i32
    %83 = scf.if %82 -> (tensor<64x64xf32>) {
      scf.yield %77#0 : tensor<64x64xf32>
    } else {
      scf.yield %cst : tensor<64x64xf32>
    }
    %84 = tt.splat %arg45 : i32 -> tensor<64x1xi32>
    %85 = arith.muli %37, %84 : tensor<64x1xi32>
    %86 = tt.broadcast %85 : tensor<64x1xi32> -> tensor<64x64xi32>
    %87 = arith.addi %86, %42 : tensor<64x64xi32>
    %88 = tt.splat %arg42 : i32 -> tensor<64x1xi32>
    %89 = arith.muli %37, %88 : tensor<64x1xi32>
    %90 = tt.broadcast %89 : tensor<64x1xi32> -> tensor<64x64xi32>
    %91 = arith.addi %90, %42 : tensor<64x64xi32>
    %92 = tt.splat %33 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %93 = tt.addptr %92, %87 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %94 = arith.truncf %83 : tensor<64x64xf32> to tensor<64x64xbf16>
    tt.store %93, %94, %48 : tensor<64x64x!tt.ptr<bf16>>
    %95 = tt.splat %29 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
    %96 = tt.addptr %95, %91 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %97 = arith.truncf %80 : tensor<64x64xf32> to tensor<64x64xbf16>
    tt.store %96, %97, %48 : tensor<64x64x!tt.ptr<bf16>>
    tt.return
  }
}

// -----
// Source: gaoyou/0701/solve_tril/norm_dot.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 8)>
// CHECK-NEXT: module {
// CHECK-NEXT:   module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:     tt.func public @merge_16x16_to_64x64_inverse_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
// CHECK-NEXT:       %c50_i32 = arith.constant 50 : i32
// CHECK-NEXT:       %c34_i32 = arith.constant 34 : i32
// CHECK-NEXT:       %c18_i32 = arith.constant 18 : i32
// CHECK-NEXT:       %c2048_i32 = arith.constant 2048 : i32
// CHECK-NEXT:       %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:       %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:       %cst = arith.constant {MetaUse} dense<48> : tensor<16xi32>
// CHECK-NEXT:       %cst_0 = arith.constant {MetaUse} dense<32> : tensor<16xi32>
// CHECK-NEXT:       %cst_1 = arith.constant {MetaUse} dense<16> : tensor<16xi32>
// CHECK-NEXT:       %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16xf32>
// CHECK-NEXT:       %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16x16xf32>
// CHECK-NEXT:       %c48_i32 = arith.constant 48 : i32
// CHECK-NEXT:       %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:       %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:       %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:       %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:       %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:       %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:       %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:       %0 = tt.get_program_id x : i32
// CHECK-NEXT:       %1 = tt.get_program_id y : i32
// CHECK-NEXT:       %2 = arith.divsi %1, %c32_i32 : i32
// CHECK-NEXT:       %3 = arith.remsi %1, %c32_i32 : i32
// CHECK-NEXT:       %4 = arith.muli %2, %arg2 : i32
// CHECK-NEXT:       %5 = tt.make_range {MetaUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT:       %6 = tt.make_range {DataUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT:       %7 = tt.expand_dims %6 {DataUse, axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT:       %8 = tt.expand_dims %6 {DataUse, axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT:       %9 = tt.broadcast %7 {DataUse} : tensor<16x1xi32> -> tensor<16x16xi32>
// CHECK-NEXT:       %10 = tt.broadcast %8 {DataUse} : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK-NEXT:       %11 = arith.cmpi sgt, %9, %10 {DataUse} : tensor<16x16xi32>
// CHECK-NEXT:       %12 = arith.cmpi eq, %9, %10 {DataUse} : tensor<16x16xi32>
// CHECK-NEXT:       %13 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:       %14 = arith.addi %13, %3 : i32
// CHECK-NEXT:       %15 = arith.muli %14, %c64_i32 : i32
// CHECK-NEXT:       %16 = tt.addptr %arg0, %15 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %17 = tt.addptr %arg1, %15 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %18 = arith.muli %0, %c64_i32 : i32
// CHECK-NEXT:       %19 = arith.extsi %arg2 : i32 to i64
// CHECK-NEXT:       %20 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%18, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %21 = arith.addi %18, %c16_i32 : i32
// CHECK-NEXT:       %22 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %23 = arith.addi %18, %c32_i32 : i32
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %25 = arith.addi %18, %c48_i32 : i32
// CHECK-NEXT:       %26 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c48_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %27 = tt.load %20 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %28 = tt.load %22 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %29 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %30 = tt.load %26 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %31 = arith.select %11, %27, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       %32 = arith.subf %cst_3, %31 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %33 = arith.select %11, %28, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       %34 = arith.subf %cst_3, %33 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %35 = arith.select %11, %29, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       %36 = arith.subf %cst_3, %35 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %37 = arith.select %11, %30, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       %38 = arith.subf %cst_3, %37 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %39 = arith.subi %arg2, %18 : i32
// CHECK-NEXT:       %40 = arith.minsi %39, %c16_i32 : i32
// CHECK-NEXT:       %41 = scf.for %arg3 = %c2_i32 to %40 step %c1_i32 iter_args(%arg4 = %32) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:         %97 = arith.addi %18, %arg3 : i32
// CHECK-NEXT:         %98 = arith.muli %97, %c2048_i32 : i32
// CHECK-NEXT:         %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %102 = tt.load %101 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %103 = arith.subf %cst_2, %102 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %104 = tt.splat %arg3 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %105 = arith.cmpi slt, %6, %104 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %106 = arith.select %105, %103, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %107 = tt.expand_dims %106 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %108 = tt.broadcast %107 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %109 = arith.mulf %108, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:         %110 = "tt.reduce"(%109) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %118 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %118 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %111 = arith.addf %106, %110 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %112 = arith.cmpi eq, %6, %104 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %113 = tt.expand_dims %112 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %114 = tt.expand_dims %111 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:         %115 = tt.broadcast %114 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %116 = tt.broadcast %113 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:         %117 = arith.select %116, %115, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:         scf.yield %117 : tensor<16x16xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %42 = arith.minsi %39, %c32_i32 : i32
// CHECK-NEXT:       %43 = scf.for %arg3 = %c18_i32 to %42 step %c1_i32 iter_args(%arg4 = %34) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:         %97 = arith.addi %18, %arg3 : i32
// CHECK-NEXT:         %98 = arith.muli %97, %c2048_i32 : i32
// CHECK-NEXT:         %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %102 = tt.addptr %101, %cst_1 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %103 = tt.load %102 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %104 = arith.subf %cst_2, %103 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %105 = arith.subi %arg3, %c16_i32 : i32
// CHECK-NEXT:         %106 = tt.splat %105 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %107 = arith.cmpi slt, %6, %106 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %108 = arith.select %107, %104, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %109 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %110 = tt.broadcast %109 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %111 = arith.mulf %110, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:         %112 = "tt.reduce"(%111) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %120 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %120 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %113 = arith.addf %108, %112 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %114 = arith.cmpi eq, %6, %106 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %116 = tt.expand_dims %113 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:         %117 = tt.broadcast %116 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %118 = tt.broadcast %115 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:         %119 = arith.select %118, %117, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:         scf.yield %119 : tensor<16x16xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %44 = arith.minsi %39, %c48_i32 : i32
// CHECK-NEXT:       %45 = scf.for %arg3 = %c34_i32 to %44 step %c1_i32 iter_args(%arg4 = %36) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:         %97 = arith.addi %18, %arg3 : i32
// CHECK-NEXT:         %98 = arith.muli %97, %c2048_i32 : i32
// CHECK-NEXT:         %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %102 = tt.addptr %101, %cst_0 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %103 = tt.load %102 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %104 = arith.subf %cst_2, %103 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %105 = arith.subi %arg3, %c32_i32 : i32
// CHECK-NEXT:         %106 = tt.splat %105 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %107 = arith.cmpi slt, %6, %106 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %108 = arith.select %107, %104, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %109 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %110 = tt.broadcast %109 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %111 = arith.mulf %110, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:         %112 = "tt.reduce"(%111) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %120 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %120 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %113 = arith.addf %108, %112 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %114 = arith.cmpi eq, %6, %106 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %116 = tt.expand_dims %113 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:         %117 = tt.broadcast %116 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %118 = tt.broadcast %115 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:         %119 = arith.select %118, %117, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:         scf.yield %119 : tensor<16x16xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %46 = arith.minsi %39, %c64_i32 : i32
// CHECK-NEXT:       %47 = scf.for %arg3 = %c50_i32 to %46 step %c1_i32 iter_args(%arg4 = %38) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:         %97 = arith.addi %18, %arg3 : i32
// CHECK-NEXT:         %98 = arith.muli %97, %c2048_i32 : i32
// CHECK-NEXT:         %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %102 = tt.addptr %101, %cst {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %103 = tt.load %102 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %104 = arith.subf %cst_2, %103 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %105 = arith.subi %arg3, %c48_i32 : i32
// CHECK-NEXT:         %106 = tt.splat %105 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %107 = arith.cmpi slt, %6, %106 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %108 = arith.select %107, %104, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %109 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %110 = tt.broadcast %109 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %111 = arith.mulf %110, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:         %112 = "tt.reduce"(%111) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %120 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %120 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:         %113 = arith.addf %108, %112 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %114 = arith.cmpi eq, %6, %106 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %116 = tt.expand_dims %113 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:         %117 = tt.broadcast %116 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:         %118 = tt.broadcast %115 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:         %119 = arith.select %118, %117, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:         scf.yield %119 : tensor<16x16xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %48 = arith.uitofp %12 {DataUse} : tensor<16x16xi1> to tensor<16x16xf32>
// CHECK-NEXT:       %49 = arith.addf %41, %48 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %50 = arith.addf %43, %48 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %51 = arith.addf %45, %48 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %52 = arith.addf %47, %48 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %53 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %54 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %55 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %56 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %57 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %58 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %59 = tt.load %53 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %60 = tt.load %54 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %61 = tt.load %55 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %62 = tt.load %56 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %63 = tt.load %57 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %64 = tt.load %58 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %65 = tt.dot %50, %59, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %66 = tt.dot %65, %49, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %67 = arith.subf %cst_3, %66 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %68 = tt.dot %51, %61, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %69 = tt.dot %68, %50, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %70 = arith.subf %cst_3, %69 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %71 = tt.dot %52, %64, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %72 = tt.dot %71, %51, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %73 = arith.subf %cst_3, %72 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %74 = tt.dot %60, %49, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %75 = tt.dot %61, %67, %74 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %76 = tt.dot %51, %75, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %77 = arith.subf %cst_3, %76 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %78 = tt.dot %63, %50, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %79 = tt.dot %64, %70, %78 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %80 = tt.dot %52, %79, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %81 = arith.subf %cst_3, %80 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %82 = tt.dot %62, %49, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %83 = tt.dot %63, %67, %82 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %84 = tt.dot %64, %77, %83 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %85 = tt.dot %52, %84, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %86 = arith.subf %cst_3, %85 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %87 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%18, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %88 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %89 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %90 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c48_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %91 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %92 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %93 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %94 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %95 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %96 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %87, %49 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %88, %50 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %89, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %90, %52 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %91, %67 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %92, %77 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %93, %70 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %94, %86 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %95, %81 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.store %96, %73 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       tt.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.disable_auto_tile_and_bind_subblock} {
// CHECK-NEXT:     tt.func public @merge_16x16_to_64x64_inverse_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
// CHECK-NEXT:       %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<8x16xf32>
// CHECK-NEXT:       %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16x8xf32>
// CHECK-NEXT:       %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:       %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:       %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:       %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:       %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:       %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:       %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:       %c48_i32 = arith.constant 48 : i32
// CHECK-NEXT:       %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16x16xf32>
// CHECK-NEXT:       %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16xf32>
// CHECK-NEXT:       %cst_3 = arith.constant {MetaUse} dense<16> : tensor<16xi32>
// CHECK-NEXT:       %cst_4 = arith.constant {MetaUse} dense<32> : tensor<16xi32>
// CHECK-NEXT:       %cst_5 = arith.constant {MetaUse} dense<48> : tensor<16xi32>
// CHECK-NEXT:       %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:       %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:       %c2048_i32 = arith.constant 2048 : i32
// CHECK-NEXT:       %c18_i32 = arith.constant 18 : i32
// CHECK-NEXT:       %c34_i32 = arith.constant 34 : i32
// CHECK-NEXT:       %c50_i32 = arith.constant 50 : i32
// CHECK-NEXT:       %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:       %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:       %2 = affine.apply #map()[%1]
// CHECK-NEXT:       %3 = arith.index_cast %2 : index to i32
// CHECK-NEXT:       %4 = tt.get_program_id x : i32
// CHECK-NEXT:       %5 = tt.get_program_id y : i32
// CHECK-NEXT:       %6 = arith.divsi %5, %c32_i32 : i32
// CHECK-NEXT:       %7 = arith.remsi %5, %c32_i32 : i32
// CHECK-NEXT:       %8 = arith.muli %6, %arg2 : i32
// CHECK-NEXT:       %9 = tt.make_range {MetaUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT:       %10 = tt.make_range {DataUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT:       %11 = tt.expand_dims %10 {DataUse, axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %10[%2] [8] [1] {DataUse, should_kept_slice} : tensor<16xi32> to tensor<8xi32>
// CHECK-NEXT:       %12 = tt.expand_dims %extracted_slice {DataUse, axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
// CHECK-NEXT:       %13 = tt.broadcast %11 {DataUse} : tensor<16x1xi32> -> tensor<16x8xi32>
// CHECK-NEXT:       %14 = tt.broadcast %12 {DataUse} : tensor<1x8xi32> -> tensor<16x8xi32>
// CHECK-NEXT:       %15 = arith.cmpi sgt, %13, %14 {DataUse} : tensor<16x8xi32>
// CHECK-NEXT:       %16 = arith.cmpi eq, %13, %14 {DataUse} : tensor<16x8xi32>
// CHECK-NEXT:       %17 = arith.muli %8, %c32_i32 : i32
// CHECK-NEXT:       %18 = arith.addi %17, %7 : i32
// CHECK-NEXT:       %19 = arith.muli %18, %c64_i32 : i32
// CHECK-NEXT:       %20 = tt.addptr %arg0, %19 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %21 = tt.addptr %arg1, %19 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %22 = arith.muli %4, %c64_i32 : i32
// CHECK-NEXT:       %23 = arith.extsi %arg2 : i32 to i64
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%22, %3] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
// CHECK-NEXT:       %25 = arith.addi %22, %c16_i32 : i32
// CHECK-NEXT:       %26 = arith.addi %3, %c16_i32 : i32
// CHECK-NEXT:       %27 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%25, %26] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
// CHECK-NEXT:       %28 = arith.addi %22, %c32_i32 : i32
// CHECK-NEXT:       %29 = arith.addi %3, %c32_i32 : i32
// CHECK-NEXT:       %30 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%28, %29] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
// CHECK-NEXT:       %31 = arith.addi %22, %c48_i32 : i32
// CHECK-NEXT:       %32 = arith.addi %3, %c48_i32 : i32
// CHECK-NEXT:       %33 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%31, %32] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
// CHECK-NEXT:       %34 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       %35 = tt.load %27 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       %36 = tt.load %30 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       %37 = tt.load %33 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       %38 = arith.select %15, %34, %cst_0 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:       %39 = arith.subf %cst_0, %38 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %40 = arith.select %15, %35, %cst_0 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:       %41 = arith.subf %cst_0, %40 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %42 = arith.select %15, %36, %cst_0 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:       %43 = arith.subf %cst_0, %42 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %44 = arith.select %15, %37, %cst_0 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:       %45 = arith.subf %cst_0, %44 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %46 = arith.subi %arg2, %22 : i32
// CHECK-NEXT:       %47 = arith.minsi %46, %c16_i32 : i32
// CHECK-NEXT:       %48 = scf.for %arg3 = %c2_i32 to %47 step %c1_i32 iter_args(%arg4 = %39) -> (tensor<16x8xf32>)  : i32 {
// CHECK-NEXT:         %157 = arith.addi %22, %arg3 : i32
// CHECK-NEXT:         %158 = arith.muli %157, %c2048_i32 : i32
// CHECK-NEXT:         %159 = tt.addptr %20, %158 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %160 = tt.splat %159 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %161 = tt.addptr %160, %9 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %162 = tt.load %161 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %163 = arith.subf %cst_2, %162 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %164 = tt.splat %arg3 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %165 = arith.cmpi slt, %10, %164 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %166 = arith.select %165, %163, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %167 = tt.expand_dims %166 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %168 = tt.broadcast %167 {DataUse} : tensor<16x1xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %169 = arith.mulf %168, %arg4 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:         %170 = "tt.reduce"(%169) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %178 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %178 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x8xf32>) -> tensor<8xf32>
// CHECK-NEXT:         %extracted_slice_54 = tensor.extract_slice %166[%2] [8] [1] {DataUse, should_kept_slice} : tensor<16xf32> to tensor<8xf32>
// CHECK-NEXT:         %171 = arith.addf %extracted_slice_54, %170 {DataUse} : tensor<8xf32>
// CHECK-NEXT:         %172 = arith.cmpi eq, %10, %164 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %173 = tt.expand_dims %172 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %174 = tt.expand_dims %171 {DataUse, axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
// CHECK-NEXT:         %175 = tt.broadcast %174 {DataUse} : tensor<1x8xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %176 = tt.broadcast %173 {DataUse} : tensor<16x1xi1> -> tensor<16x8xi1>
// CHECK-NEXT:         %177 = arith.select %176, %175, %arg4 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:         scf.yield %177 : tensor<16x8xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %49 = arith.minsi %46, %c32_i32 : i32
// CHECK-NEXT:       %50 = scf.for %arg3 = %c18_i32 to %49 step %c1_i32 iter_args(%arg4 = %41) -> (tensor<16x8xf32>)  : i32 {
// CHECK-NEXT:         %157 = arith.addi %22, %arg3 : i32
// CHECK-NEXT:         %158 = arith.muli %157, %c2048_i32 : i32
// CHECK-NEXT:         %159 = tt.addptr %20, %158 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %160 = tt.splat %159 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %161 = tt.addptr %160, %9 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %162 = tt.addptr %161, %cst_3 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %163 = tt.load %162 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %164 = arith.subf %cst_2, %163 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %165 = arith.subi %arg3, %c16_i32 : i32
// CHECK-NEXT:         %166 = tt.splat %165 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %167 = arith.cmpi slt, %10, %166 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %168 = arith.select %167, %164, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %169 = tt.expand_dims %168 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %170 = tt.broadcast %169 {DataUse} : tensor<16x1xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %171 = arith.mulf %170, %arg4 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:         %172 = "tt.reduce"(%171) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %180 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %180 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x8xf32>) -> tensor<8xf32>
// CHECK-NEXT:         %extracted_slice_54 = tensor.extract_slice %168[%2] [8] [1] {DataUse, should_kept_slice} : tensor<16xf32> to tensor<8xf32>
// CHECK-NEXT:         %173 = arith.addf %extracted_slice_54, %172 {DataUse} : tensor<8xf32>
// CHECK-NEXT:         %174 = arith.cmpi eq, %10, %166 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %175 = tt.expand_dims %174 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %176 = tt.expand_dims %173 {DataUse, axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
// CHECK-NEXT:         %177 = tt.broadcast %176 {DataUse} : tensor<1x8xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %178 = tt.broadcast %175 {DataUse} : tensor<16x1xi1> -> tensor<16x8xi1>
// CHECK-NEXT:         %179 = arith.select %178, %177, %arg4 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:         scf.yield %179 : tensor<16x8xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %51 = arith.minsi %46, %c48_i32 : i32
// CHECK-NEXT:       %52 = scf.for %arg3 = %c34_i32 to %51 step %c1_i32 iter_args(%arg4 = %43) -> (tensor<16x8xf32>)  : i32 {
// CHECK-NEXT:         %157 = arith.addi %22, %arg3 : i32
// CHECK-NEXT:         %158 = arith.muli %157, %c2048_i32 : i32
// CHECK-NEXT:         %159 = tt.addptr %20, %158 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %160 = tt.splat %159 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %161 = tt.addptr %160, %9 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %162 = tt.addptr %161, %cst_4 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %163 = tt.load %162 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %164 = arith.subf %cst_2, %163 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %165 = arith.subi %arg3, %c32_i32 : i32
// CHECK-NEXT:         %166 = tt.splat %165 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %167 = arith.cmpi slt, %10, %166 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %168 = arith.select %167, %164, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %169 = tt.expand_dims %168 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %170 = tt.broadcast %169 {DataUse} : tensor<16x1xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %171 = arith.mulf %170, %arg4 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:         %172 = "tt.reduce"(%171) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %180 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %180 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x8xf32>) -> tensor<8xf32>
// CHECK-NEXT:         %extracted_slice_54 = tensor.extract_slice %168[%2] [8] [1] {DataUse, should_kept_slice} : tensor<16xf32> to tensor<8xf32>
// CHECK-NEXT:         %173 = arith.addf %extracted_slice_54, %172 {DataUse} : tensor<8xf32>
// CHECK-NEXT:         %174 = arith.cmpi eq, %10, %166 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %175 = tt.expand_dims %174 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %176 = tt.expand_dims %173 {DataUse, axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
// CHECK-NEXT:         %177 = tt.broadcast %176 {DataUse} : tensor<1x8xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %178 = tt.broadcast %175 {DataUse} : tensor<16x1xi1> -> tensor<16x8xi1>
// CHECK-NEXT:         %179 = arith.select %178, %177, %arg4 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:         scf.yield %179 : tensor<16x8xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %53 = arith.minsi %46, %c64_i32 : i32
// CHECK-NEXT:       %54 = scf.for %arg3 = %c50_i32 to %53 step %c1_i32 iter_args(%arg4 = %45) -> (tensor<16x8xf32>)  : i32 {
// CHECK-NEXT:         %157 = arith.addi %22, %arg3 : i32
// CHECK-NEXT:         %158 = arith.muli %157, %c2048_i32 : i32
// CHECK-NEXT:         %159 = tt.addptr %20, %158 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %160 = tt.splat %159 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %161 = tt.addptr %160, %9 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %162 = tt.addptr %161, %cst_5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:         %163 = tt.load %162 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %164 = arith.subf %cst_2, %163 {DataUse} : tensor<16xf32>
// CHECK-NEXT:         %165 = arith.subi %arg3, %c48_i32 : i32
// CHECK-NEXT:         %166 = tt.splat %165 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:         %167 = arith.cmpi slt, %10, %166 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %168 = arith.select %167, %164, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:         %169 = tt.expand_dims %168 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:         %170 = tt.broadcast %169 {DataUse} : tensor<16x1xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %171 = arith.mulf %170, %arg4 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:         %172 = "tt.reduce"(%171) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:           %180 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:           tt.reduce.return %180 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<16x8xf32>) -> tensor<8xf32>
// CHECK-NEXT:         %extracted_slice_54 = tensor.extract_slice %168[%2] [8] [1] {DataUse, should_kept_slice} : tensor<16xf32> to tensor<8xf32>
// CHECK-NEXT:         %173 = arith.addf %extracted_slice_54, %172 {DataUse} : tensor<8xf32>
// CHECK-NEXT:         %174 = arith.cmpi eq, %10, %166 {DataUse} : tensor<16xi32>
// CHECK-NEXT:         %175 = tt.expand_dims %174 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:         %176 = tt.expand_dims %173 {DataUse, axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
// CHECK-NEXT:         %177 = tt.broadcast %176 {DataUse} : tensor<1x8xf32> -> tensor<16x8xf32>
// CHECK-NEXT:         %178 = tt.broadcast %175 {DataUse} : tensor<16x1xi1> -> tensor<16x8xi1>
// CHECK-NEXT:         %179 = arith.select %178, %177, %arg4 {DataUse} : tensor<16x8xi1>, tensor<16x8xf32>
// CHECK-NEXT:         scf.yield %179 : tensor<16x8xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %55 = arith.uitofp %16 {DataUse} : tensor<16x8xi1> to tensor<16x8xf32>
// CHECK-NEXT:       %56 = arith.addf %48, %55 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %57 = arith.addf %50, %55 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %58 = arith.addf %52, %55 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %59 = arith.addf %54, %55 {DataUse} : tensor<16x8xf32>
// CHECK-NEXT:       %60 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %61 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %62 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%28, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %63 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%31, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %64 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%31, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %65 = tt.make_tensor_ptr %20, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%31, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:       %66 = tt.load %60 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %67 = tt.load %61 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %68 = tt.load %62 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %69 = tt.load %63 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %70 = tt.load %64 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %71 = tt.load %65 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:       %72 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview = memref.subview %72[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %57 in writable %subview : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %72, %alloc : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %73 = bufferization.to_tensor %alloc restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %74 = tt.dot %73, %66, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %75 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %74 in writable %75 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_6 = memref.subview %75[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_7 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_6, %alloc_7 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %76 = bufferization.to_tensor %alloc_7 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %77 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_8 = memref.subview %77[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %76 in writable %subview_8 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_9 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %77, %alloc_9 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %78 = bufferization.to_tensor %alloc_9 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %79 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_10 = memref.subview %79[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %56 in writable %subview_10 : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_11 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %79, %alloc_11 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %80 = bufferization.to_tensor %alloc_11 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %81 = tt.dot %78, %80, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %82 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %81 in writable %82 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_12 = memref.subview %82[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_13 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_12, %alloc_13 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %83 = bufferization.to_tensor %alloc_13 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %84 = arith.subf %cst, %83 {DataUse} : tensor<8x16xf32>
// CHECK-NEXT:       %85 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_14 = memref.subview %85[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %58 in writable %subview_14 : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_15 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %85, %alloc_15 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %86 = bufferization.to_tensor %alloc_15 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %87 = tt.dot %86, %68, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %88 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %87 in writable %88 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_16 = memref.subview %88[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_17 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_16, %alloc_17 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %89 = bufferization.to_tensor %alloc_17 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %90 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_18 = memref.subview %90[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %89 in writable %subview_18 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_19 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %90, %alloc_19 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %91 = bufferization.to_tensor %alloc_19 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %92 = tt.dot %91, %73, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %93 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %92 in writable %93 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_20 = memref.subview %93[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_21 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_20, %alloc_21 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %94 = bufferization.to_tensor %alloc_21 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %95 = arith.subf %cst, %94 {DataUse} : tensor<8x16xf32>
// CHECK-NEXT:       %96 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_22 = memref.subview %96[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %59 in writable %subview_22 : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_23 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %96, %alloc_23 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %97 = bufferization.to_tensor %alloc_23 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %98 = tt.dot %97, %71, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %99 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %98 in writable %99 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_24 = memref.subview %99[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_25 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_24, %alloc_25 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %100 = bufferization.to_tensor %alloc_25 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %101 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_26 = memref.subview %101[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %100 in writable %subview_26 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_27 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %101, %alloc_27 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %102 = bufferization.to_tensor %alloc_27 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %103 = tt.dot %102, %86, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %104 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %103 in writable %104 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_28 = memref.subview %104[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_29 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_28, %alloc_29 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %105 = bufferization.to_tensor %alloc_29 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %106 = arith.subf %cst, %105 {DataUse} : tensor<8x16xf32>
// CHECK-NEXT:       %107 = tt.dot %67, %80, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %108 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_30 = memref.subview %108[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %84 in writable %subview_30 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_31 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %108, %alloc_31 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %109 = bufferization.to_tensor %alloc_31 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %110 = tt.dot %68, %109, %107 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %111 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %110 in writable %111 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_32 = memref.subview %111[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_33 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_32, %alloc_33 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %112 = bufferization.to_tensor %alloc_33 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %113 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_34 = memref.subview %113[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %112 in writable %subview_34 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_35 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %113, %alloc_35 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %114 = bufferization.to_tensor %alloc_35 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %115 = tt.dot %86, %114, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %116 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %115 in writable %116 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_36 = memref.subview %116[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_37 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_36, %alloc_37 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %117 = bufferization.to_tensor %alloc_37 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %118 = arith.subf %cst, %117 {DataUse} : tensor<8x16xf32>
// CHECK-NEXT:       %119 = tt.dot %70, %73, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %120 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_38 = memref.subview %120[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %95 in writable %subview_38 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_39 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %120, %alloc_39 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %121 = bufferization.to_tensor %alloc_39 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %122 = tt.dot %71, %121, %119 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %123 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %122 in writable %123 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_40 = memref.subview %123[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_41 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_40, %alloc_41 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %124 = bufferization.to_tensor %alloc_41 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %125 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_42 = memref.subview %125[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %124 in writable %subview_42 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_43 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %125, %alloc_43 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %126 = bufferization.to_tensor %alloc_43 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %127 = tt.dot %97, %126, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %128 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %127 in writable %128 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_44 = memref.subview %128[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_45 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_44, %alloc_45 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %129 = bufferization.to_tensor %alloc_45 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %130 = arith.subf %cst, %129 {DataUse} : tensor<8x16xf32>
// CHECK-NEXT:       %131 = tt.dot %69, %80, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %132 = tt.dot %70, %109, %131 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %133 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_46 = memref.subview %133[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %118 in writable %subview_46 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_47 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %133, %alloc_47 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %134 = bufferization.to_tensor %alloc_47 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %135 = tt.dot %71, %134, %132 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %136 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %135 in writable %136 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_48 = memref.subview %136[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_49 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_48, %alloc_49 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %137 = bufferization.to_tensor %alloc_49 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %138 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       %subview_50 = memref.subview %138[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       bufferization.materialize_in_destination %137 in writable %subview_50 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
// CHECK-NEXT:       %alloc_51 = memref.alloc() : memref<16x16xf32>
// CHECK-NEXT:       memref.copy %138, %alloc_51 : memref<16x16xf32> to memref<16x16xf32>
// CHECK-NEXT:       %139 = bufferization.to_tensor %alloc_51 restrict writable {DataUse} : memref<16x16xf32>
// CHECK-NEXT:       %140 = tt.dot %97, %139, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %141 = memref_ext.alloc_workspace() : memref<16x16xf32>
// CHECK-NEXT:       bufferization.materialize_in_destination %140 in writable %141 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
// CHECK-NEXT:       %subview_52 = memref.subview %141[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
// CHECK-NEXT:       %alloc_53 = memref.alloc() : memref<8x16xf32>
// CHECK-NEXT:       memref.copy %subview_52, %alloc_53 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
// CHECK-NEXT:       %142 = bufferization.to_tensor %alloc_53 restrict writable {DataUse} : memref<8x16xf32>
// CHECK-NEXT:       %143 = arith.subf %cst, %142 {DataUse} : tensor<8x16xf32>
// CHECK-NEXT:       %144 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%22, %3] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:       %145 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%25, %26] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:       %146 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%28, %29] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:       %147 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%31, %32] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:       %148 = arith.addi %25, %3 : i32
// CHECK-NEXT:       %149 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%148, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:       %150 = arith.addi %28, %3 : i32
// CHECK-NEXT:       %151 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%150, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:       %152 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%150, %c16_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:       %153 = arith.addi %31, %3 : i32
// CHECK-NEXT:       %154 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%153, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:       %155 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%153, %c16_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:       %156 = tt.make_tensor_ptr %21, [%23, %c64_i64], [%c2048_i64, %c1_i64], [%153, %c32_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:       tt.store %144, %56 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       tt.store %145, %57 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       tt.store %146, %58 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       tt.store %147, %59 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:       tt.store %149, %84 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:       tt.store %151, %118 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:       tt.store %152, %95 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:       tt.store %154, %143 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:       tt.store %155, %130 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:       tt.store %156, %106 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:       tt.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

#map = affine_map<()[s0] -> (s0 * 8)>
module {
  module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
    tt.func public @merge_16x16_to_64x64_inverse_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
      %c50_i32 = arith.constant 50 : i32
      %c34_i32 = arith.constant 34 : i32
      %c18_i32 = arith.constant 18 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %cst = arith.constant dense<48> : tensor<16xi32>
      %cst_0 = arith.constant dense<32> : tensor<16xi32>
      %cst_1 = arith.constant dense<16> : tensor<16xi32>
      %cst_2 = arith.constant dense<0.000000e+00> : tensor<16xf32>
      %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
      %c48_i32 = arith.constant 48 : i32
      %c16_i32 = arith.constant 16 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1_i64 = arith.constant 1 : i64
      %c2048_i64 = arith.constant 2048 : i64
      %c64_i64 = arith.constant 64 : i64
      %c64_i32 = arith.constant 64 : i32
      %c32_i32 = arith.constant 32 : i32
      %0 = tt.get_program_id x : i32
      %1 = tt.get_program_id y : i32
      %2 = arith.divsi %1, %c32_i32 : i32
      %3 = arith.remsi %1, %c32_i32 : i32
      %4 = arith.muli %2, %arg2 : i32
      %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %7 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      %8 = tt.broadcast %6 : tensor<16x1xi32> -> tensor<16x16xi32>
      %9 = tt.broadcast %7 : tensor<1x16xi32> -> tensor<16x16xi32>
      %10 = arith.cmpi sgt, %8, %9 : tensor<16x16xi32>
      %11 = arith.cmpi eq, %8, %9 : tensor<16x16xi32>
      %12 = arith.muli %4, %c32_i32 : i32
      %13 = arith.addi %12, %3 : i32
      %14 = arith.muli %13, %c64_i32 : i32
      %15 = tt.addptr %arg0, %14 : !tt.ptr<f32>, i32
      %16 = tt.addptr %arg1, %14 : !tt.ptr<f32>, i32
      %17 = arith.muli %0, %c64_i32 : i32
      %18 = arith.extsi %arg2 : i32 to i64
      %19 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %20 = arith.addi %17, %c16_i32 : i32
      %21 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%20, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %22 = arith.addi %17, %c32_i32 : i32
      %23 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%22, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %24 = arith.addi %17, %c48_i32 : i32
      %25 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c48_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %26 = tt.load %19 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %27 = tt.load %21 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %28 = tt.load %23 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %29 = tt.load %25 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %30 = arith.select %10, %26, %cst_3 : tensor<16x16xi1>, tensor<16x16xf32>
      %31 = arith.subf %cst_3, %30 : tensor<16x16xf32>
      %32 = arith.select %10, %27, %cst_3 : tensor<16x16xi1>, tensor<16x16xf32>
      %33 = arith.subf %cst_3, %32 : tensor<16x16xf32>
      %34 = arith.select %10, %28, %cst_3 : tensor<16x16xi1>, tensor<16x16xf32>
      %35 = arith.subf %cst_3, %34 : tensor<16x16xf32>
      %36 = arith.select %10, %29, %cst_3 : tensor<16x16xi1>, tensor<16x16xf32>
      %37 = arith.subf %cst_3, %36 : tensor<16x16xf32>
      %38 = arith.subi %arg2, %17 : i32
      %39 = arith.minsi %38, %c16_i32 : i32
      %40 = scf.for %arg3 = %c2_i32 to %39 step %c1_i32 iter_args(%arg4 = %31) -> (tensor<16x16xf32>)  : i32 {
        %96 = arith.addi %17, %arg3 : i32
        %97 = arith.muli %96, %c2048_i32 : i32
        %98 = tt.addptr %15, %97 : !tt.ptr<f32>, i32
        %99 = tt.splat %98 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %100 = tt.addptr %99, %5 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %101 = tt.load %100 : tensor<16x!tt.ptr<f32>>
        %102 = arith.subf %cst_2, %101 : tensor<16xf32>
        %103 = tt.splat %arg3 : i32 -> tensor<16xi32>
        %104 = arith.cmpi slt, %5, %103 : tensor<16xi32>
        %105 = arith.select %104, %102, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %106 = tt.expand_dims %105 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %107 = tt.broadcast %106 : tensor<16x1xf32> -> tensor<16x16xf32>
        %108 = arith.mulf %107, %arg4 : tensor<16x16xf32>
        %109 = "tt.reduce"(%108) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %117 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %117 : f32
        }) : (tensor<16x16xf32>) -> tensor<16xf32>
        %110 = arith.addf %105, %109 : tensor<16xf32>
        %111 = arith.cmpi eq, %5, %103 : tensor<16xi32>
        %112 = tt.expand_dims %111 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %113 = tt.expand_dims %110 {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
        %114 = tt.broadcast %113 : tensor<1x16xf32> -> tensor<16x16xf32>
        %115 = tt.broadcast %112 : tensor<16x1xi1> -> tensor<16x16xi1>
        %116 = arith.select %115, %114, %arg4 : tensor<16x16xi1>, tensor<16x16xf32>
        scf.yield %116 : tensor<16x16xf32>
      }
      %41 = arith.minsi %38, %c32_i32 : i32
      %42 = scf.for %arg3 = %c18_i32 to %41 step %c1_i32 iter_args(%arg4 = %33) -> (tensor<16x16xf32>)  : i32 {
        %96 = arith.addi %17, %arg3 : i32
        %97 = arith.muli %96, %c2048_i32 : i32
        %98 = tt.addptr %15, %97 : !tt.ptr<f32>, i32
        %99 = tt.splat %98 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %100 = tt.addptr %99, %5 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %101 = tt.addptr %100, %cst_1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %102 = tt.load %101 : tensor<16x!tt.ptr<f32>>
        %103 = arith.subf %cst_2, %102 : tensor<16xf32>
        %104 = arith.subi %arg3, %c16_i32 : i32
        %105 = tt.splat %104 : i32 -> tensor<16xi32>
        %106 = arith.cmpi slt, %5, %105 : tensor<16xi32>
        %107 = arith.select %106, %103, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %108 = tt.expand_dims %107 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %109 = tt.broadcast %108 : tensor<16x1xf32> -> tensor<16x16xf32>
        %110 = arith.mulf %109, %arg4 : tensor<16x16xf32>
        %111 = "tt.reduce"(%110) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %119 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %119 : f32
        }) : (tensor<16x16xf32>) -> tensor<16xf32>
        %112 = arith.addf %107, %111 : tensor<16xf32>
        %113 = arith.cmpi eq, %5, %105 : tensor<16xi32>
        %114 = tt.expand_dims %113 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %115 = tt.expand_dims %112 {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
        %116 = tt.broadcast %115 : tensor<1x16xf32> -> tensor<16x16xf32>
        %117 = tt.broadcast %114 : tensor<16x1xi1> -> tensor<16x16xi1>
        %118 = arith.select %117, %116, %arg4 : tensor<16x16xi1>, tensor<16x16xf32>
        scf.yield %118 : tensor<16x16xf32>
      }
      %43 = arith.minsi %38, %c48_i32 : i32
      %44 = scf.for %arg3 = %c34_i32 to %43 step %c1_i32 iter_args(%arg4 = %35) -> (tensor<16x16xf32>)  : i32 {
        %96 = arith.addi %17, %arg3 : i32
        %97 = arith.muli %96, %c2048_i32 : i32
        %98 = tt.addptr %15, %97 : !tt.ptr<f32>, i32
        %99 = tt.splat %98 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %100 = tt.addptr %99, %5 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %101 = tt.addptr %100, %cst_0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %102 = tt.load %101 : tensor<16x!tt.ptr<f32>>
        %103 = arith.subf %cst_2, %102 : tensor<16xf32>
        %104 = arith.subi %arg3, %c32_i32 : i32
        %105 = tt.splat %104 : i32 -> tensor<16xi32>
        %106 = arith.cmpi slt, %5, %105 : tensor<16xi32>
        %107 = arith.select %106, %103, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %108 = tt.expand_dims %107 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %109 = tt.broadcast %108 : tensor<16x1xf32> -> tensor<16x16xf32>
        %110 = arith.mulf %109, %arg4 : tensor<16x16xf32>
        %111 = "tt.reduce"(%110) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %119 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %119 : f32
        }) : (tensor<16x16xf32>) -> tensor<16xf32>
        %112 = arith.addf %107, %111 : tensor<16xf32>
        %113 = arith.cmpi eq, %5, %105 : tensor<16xi32>
        %114 = tt.expand_dims %113 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %115 = tt.expand_dims %112 {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
        %116 = tt.broadcast %115 : tensor<1x16xf32> -> tensor<16x16xf32>
        %117 = tt.broadcast %114 : tensor<16x1xi1> -> tensor<16x16xi1>
        %118 = arith.select %117, %116, %arg4 : tensor<16x16xi1>, tensor<16x16xf32>
        scf.yield %118 : tensor<16x16xf32>
      }
      %45 = arith.minsi %38, %c64_i32 : i32
      %46 = scf.for %arg3 = %c50_i32 to %45 step %c1_i32 iter_args(%arg4 = %37) -> (tensor<16x16xf32>)  : i32 {
        %96 = arith.addi %17, %arg3 : i32
        %97 = arith.muli %96, %c2048_i32 : i32
        %98 = tt.addptr %15, %97 : !tt.ptr<f32>, i32
        %99 = tt.splat %98 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %100 = tt.addptr %99, %5 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %101 = tt.addptr %100, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %102 = tt.load %101 : tensor<16x!tt.ptr<f32>>
        %103 = arith.subf %cst_2, %102 : tensor<16xf32>
        %104 = arith.subi %arg3, %c48_i32 : i32
        %105 = tt.splat %104 : i32 -> tensor<16xi32>
        %106 = arith.cmpi slt, %5, %105 : tensor<16xi32>
        %107 = arith.select %106, %103, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %108 = tt.expand_dims %107 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %109 = tt.broadcast %108 : tensor<16x1xf32> -> tensor<16x16xf32>
        %110 = arith.mulf %109, %arg4 : tensor<16x16xf32>
        %111 = "tt.reduce"(%110) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %119 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %119 : f32
        }) : (tensor<16x16xf32>) -> tensor<16xf32>
        %112 = arith.addf %107, %111 : tensor<16xf32>
        %113 = arith.cmpi eq, %5, %105 : tensor<16xi32>
        %114 = tt.expand_dims %113 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %115 = tt.expand_dims %112 {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
        %116 = tt.broadcast %115 : tensor<1x16xf32> -> tensor<16x16xf32>
        %117 = tt.broadcast %114 : tensor<16x1xi1> -> tensor<16x16xi1>
        %118 = arith.select %117, %116, %arg4 : tensor<16x16xi1>, tensor<16x16xf32>
        scf.yield %118 : tensor<16x16xf32>
      }
      %47 = arith.uitofp %11 : tensor<16x16xi1> to tensor<16x16xf32>
      %48 = arith.addf %40, %47 : tensor<16x16xf32>
      %49 = arith.addf %42, %47 : tensor<16x16xf32>
      %50 = arith.addf %44, %47 : tensor<16x16xf32>
      %51 = arith.addf %46, %47 : tensor<16x16xf32>
      %52 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %53 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %54 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%22, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %55 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %56 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %57 = tt.make_tensor_ptr %15, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %58 = tt.load %52 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %59 = tt.load %53 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %60 = tt.load %54 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %61 = tt.load %55 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %62 = tt.load %56 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %63 = tt.load %57 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %64 = tt.dot %49, %58, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %65 = tt.dot %64, %48, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %66 = arith.subf %cst_3, %65 : tensor<16x16xf32>
      %67 = tt.dot %50, %60, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %68 = tt.dot %67, %49, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %69 = arith.subf %cst_3, %68 : tensor<16x16xf32>
      %70 = tt.dot %51, %63, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %71 = tt.dot %70, %50, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %72 = arith.subf %cst_3, %71 : tensor<16x16xf32>
      %73 = tt.dot %59, %48, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %74 = tt.dot %60, %66, %73 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %75 = tt.dot %50, %74, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %76 = arith.subf %cst_3, %75 : tensor<16x16xf32>
      %77 = tt.dot %62, %49, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %78 = tt.dot %63, %69, %77 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %79 = tt.dot %51, %78, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %80 = arith.subf %cst_3, %79 : tensor<16x16xf32>
      %81 = tt.dot %61, %48, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %82 = tt.dot %62, %66, %81 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %83 = tt.dot %63, %76, %82 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %84 = tt.dot %51, %83, %cst_3 {triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %85 = arith.subf %cst_3, %84 : tensor<16x16xf32>
      %86 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %87 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%20, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %88 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%22, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %89 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c48_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %90 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %91 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%22, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %92 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%22, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %93 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %94 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %95 = tt.make_tensor_ptr %16, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      tt.store %86, %48 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %87, %49 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %88, %50 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %89, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %90, %66 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %91, %76 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %92, %69 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %93, %85 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %94, %80 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.store %95, %72 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
      tt.return
    }
  }
  module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.disable_auto_tile_and_bind_subblock} {
    tt.func public @merge_16x16_to_64x64_inverse_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
      %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
      %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x8xf32>
      %c32_i32 = arith.constant 32 : i32
      %c64_i32 = arith.constant 64 : i32
      %c64_i64 = arith.constant 64 : i64
      %c2048_i64 = arith.constant 2048 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %c48_i32 = arith.constant 48 : i32
      %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
      %cst_2 = arith.constant dense<0.000000e+00> : tensor<16xf32>
      %cst_3 = arith.constant dense<16> : tensor<16xi32>
      %cst_4 = arith.constant dense<32> : tensor<16xi32>
      %cst_5 = arith.constant dense<48> : tensor<16xi32>
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2048_i32 = arith.constant 2048 : i32
      %c18_i32 = arith.constant 18 : i32
      %c34_i32 = arith.constant 34 : i32
      %c50_i32 = arith.constant 50 : i32
      %0 = hivm.hir.get_sub_block_idx -> i64
      %1 = arith.index_cast %0 : i64 to index
      %2 = affine.apply #map()[%1]
      %3 = arith.index_cast %2 : index to i32
      %4 = tt.get_program_id x : i32
      %5 = tt.get_program_id y : i32
      %6 = arith.divsi %5, %c32_i32 : i32
      %7 = arith.remsi %5, %c32_i32 : i32
      %8 = arith.muli %6, %arg2 : i32
      %9 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %extracted_slice = tensor.extract_slice %9[%2] [8] [1] {should_kept_slice} : tensor<16xi32> to tensor<8xi32>
      %11 = tt.expand_dims %extracted_slice {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
      %12 = tt.broadcast %10 : tensor<16x1xi32> -> tensor<16x8xi32>
      %13 = tt.broadcast %11 : tensor<1x8xi32> -> tensor<16x8xi32>
      %14 = arith.cmpi sgt, %12, %13 : tensor<16x8xi32>
      %15 = arith.cmpi eq, %12, %13 : tensor<16x8xi32>
      %16 = arith.muli %8, %c32_i32 : i32
      %17 = arith.addi %16, %7 : i32
      %18 = arith.muli %17, %c64_i32 : i32
      %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
      %20 = tt.addptr %arg1, %18 : !tt.ptr<f32>, i32
      %21 = arith.muli %4, %c64_i32 : i32
      %22 = arith.extsi %arg2 : i32 to i64
      %23 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%21, %3] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
      %24 = arith.addi %21, %c16_i32 : i32
      %25 = arith.addi %3, %c16_i32 : i32
      %26 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%24, %25] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
      %27 = arith.addi %21, %c32_i32 : i32
      %28 = arith.addi %3, %c32_i32 : i32
      %29 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%27, %28] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
      %30 = arith.addi %21, %c48_i32 : i32
      %31 = arith.addi %3, %c48_i32 : i32
      %32 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%30, %31] {order = array<i32: 1, 0>} : <tensor<16x8xf32>>
      %33 = tt.load %23 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
      %34 = tt.load %26 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
      %35 = tt.load %29 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
      %36 = tt.load %32 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x8xf32>>
      %37 = arith.select %14, %33, %cst_0 : tensor<16x8xi1>, tensor<16x8xf32>
      %38 = arith.subf %cst_0, %37 : tensor<16x8xf32>
      %39 = arith.select %14, %34, %cst_0 : tensor<16x8xi1>, tensor<16x8xf32>
      %40 = arith.subf %cst_0, %39 : tensor<16x8xf32>
      %41 = arith.select %14, %35, %cst_0 : tensor<16x8xi1>, tensor<16x8xf32>
      %42 = arith.subf %cst_0, %41 : tensor<16x8xf32>
      %43 = arith.select %14, %36, %cst_0 : tensor<16x8xi1>, tensor<16x8xf32>
      %44 = arith.subf %cst_0, %43 : tensor<16x8xf32>
      %45 = arith.subi %arg2, %21 : i32
      %46 = arith.minsi %45, %c16_i32 : i32
      %47 = scf.for %arg3 = %c2_i32 to %46 step %c1_i32 iter_args(%arg4 = %38) -> (tensor<16x8xf32>)  : i32 {
        %156 = arith.addi %21, %arg3 : i32
        %157 = arith.muli %156, %c2048_i32 : i32
        %158 = tt.addptr %19, %157 : !tt.ptr<f32>, i32
        %159 = tt.splat %158 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %160 = tt.addptr %159, %9 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %161 = tt.load %160 : tensor<16x!tt.ptr<f32>>
        %162 = arith.subf %cst_2, %161 : tensor<16xf32>
        %163 = tt.splat %arg3 : i32 -> tensor<16xi32>
        %164 = arith.cmpi slt, %9, %163 : tensor<16xi32>
        %165 = arith.select %164, %162, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %166 = tt.expand_dims %165 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %167 = tt.broadcast %166 : tensor<16x1xf32> -> tensor<16x8xf32>
        %168 = arith.mulf %167, %arg4 : tensor<16x8xf32>
        %169 = "tt.reduce"(%168) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %177 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %177 : f32
        }) : (tensor<16x8xf32>) -> tensor<8xf32>
        %extracted_slice_54 = tensor.extract_slice %165[%2] [8] [1] {should_kept_slice} : tensor<16xf32> to tensor<8xf32>
        %170 = arith.addf %extracted_slice_54, %169 : tensor<8xf32>
        %171 = arith.cmpi eq, %9, %163 : tensor<16xi32>
        %172 = tt.expand_dims %171 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %173 = tt.expand_dims %170 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
        %174 = tt.broadcast %173 : tensor<1x8xf32> -> tensor<16x8xf32>
        %175 = tt.broadcast %172 : tensor<16x1xi1> -> tensor<16x8xi1>
        %176 = arith.select %175, %174, %arg4 : tensor<16x8xi1>, tensor<16x8xf32>
        scf.yield %176 : tensor<16x8xf32>
      }
      %48 = arith.minsi %45, %c32_i32 : i32
      %49 = scf.for %arg3 = %c18_i32 to %48 step %c1_i32 iter_args(%arg4 = %40) -> (tensor<16x8xf32>)  : i32 {
        %156 = arith.addi %21, %arg3 : i32
        %157 = arith.muli %156, %c2048_i32 : i32
        %158 = tt.addptr %19, %157 : !tt.ptr<f32>, i32
        %159 = tt.splat %158 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %160 = tt.addptr %159, %9 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %161 = tt.addptr %160, %cst_3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %162 = tt.load %161 : tensor<16x!tt.ptr<f32>>
        %163 = arith.subf %cst_2, %162 : tensor<16xf32>
        %164 = arith.subi %arg3, %c16_i32 : i32
        %165 = tt.splat %164 : i32 -> tensor<16xi32>
        %166 = arith.cmpi slt, %9, %165 : tensor<16xi32>
        %167 = arith.select %166, %163, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %168 = tt.expand_dims %167 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %169 = tt.broadcast %168 : tensor<16x1xf32> -> tensor<16x8xf32>
        %170 = arith.mulf %169, %arg4 : tensor<16x8xf32>
        %171 = "tt.reduce"(%170) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %179 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %179 : f32
        }) : (tensor<16x8xf32>) -> tensor<8xf32>
        %extracted_slice_54 = tensor.extract_slice %167[%2] [8] [1] {should_kept_slice} : tensor<16xf32> to tensor<8xf32>
        %172 = arith.addf %extracted_slice_54, %171 : tensor<8xf32>
        %173 = arith.cmpi eq, %9, %165 : tensor<16xi32>
        %174 = tt.expand_dims %173 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %175 = tt.expand_dims %172 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
        %176 = tt.broadcast %175 : tensor<1x8xf32> -> tensor<16x8xf32>
        %177 = tt.broadcast %174 : tensor<16x1xi1> -> tensor<16x8xi1>
        %178 = arith.select %177, %176, %arg4 : tensor<16x8xi1>, tensor<16x8xf32>
        scf.yield %178 : tensor<16x8xf32>
      }
      %50 = arith.minsi %45, %c48_i32 : i32
      %51 = scf.for %arg3 = %c34_i32 to %50 step %c1_i32 iter_args(%arg4 = %42) -> (tensor<16x8xf32>)  : i32 {
        %156 = arith.addi %21, %arg3 : i32
        %157 = arith.muli %156, %c2048_i32 : i32
        %158 = tt.addptr %19, %157 : !tt.ptr<f32>, i32
        %159 = tt.splat %158 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %160 = tt.addptr %159, %9 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %161 = tt.addptr %160, %cst_4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %162 = tt.load %161 : tensor<16x!tt.ptr<f32>>
        %163 = arith.subf %cst_2, %162 : tensor<16xf32>
        %164 = arith.subi %arg3, %c32_i32 : i32
        %165 = tt.splat %164 : i32 -> tensor<16xi32>
        %166 = arith.cmpi slt, %9, %165 : tensor<16xi32>
        %167 = arith.select %166, %163, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %168 = tt.expand_dims %167 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %169 = tt.broadcast %168 : tensor<16x1xf32> -> tensor<16x8xf32>
        %170 = arith.mulf %169, %arg4 : tensor<16x8xf32>
        %171 = "tt.reduce"(%170) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %179 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %179 : f32
        }) : (tensor<16x8xf32>) -> tensor<8xf32>
        %extracted_slice_54 = tensor.extract_slice %167[%2] [8] [1] {should_kept_slice} : tensor<16xf32> to tensor<8xf32>
        %172 = arith.addf %extracted_slice_54, %171 : tensor<8xf32>
        %173 = arith.cmpi eq, %9, %165 : tensor<16xi32>
        %174 = tt.expand_dims %173 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %175 = tt.expand_dims %172 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
        %176 = tt.broadcast %175 : tensor<1x8xf32> -> tensor<16x8xf32>
        %177 = tt.broadcast %174 : tensor<16x1xi1> -> tensor<16x8xi1>
        %178 = arith.select %177, %176, %arg4 : tensor<16x8xi1>, tensor<16x8xf32>
        scf.yield %178 : tensor<16x8xf32>
      }
      %52 = arith.minsi %45, %c64_i32 : i32
      %53 = scf.for %arg3 = %c50_i32 to %52 step %c1_i32 iter_args(%arg4 = %44) -> (tensor<16x8xf32>)  : i32 {
        %156 = arith.addi %21, %arg3 : i32
        %157 = arith.muli %156, %c2048_i32 : i32
        %158 = tt.addptr %19, %157 : !tt.ptr<f32>, i32
        %159 = tt.splat %158 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %160 = tt.addptr %159, %9 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %161 = tt.addptr %160, %cst_5 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %162 = tt.load %161 : tensor<16x!tt.ptr<f32>>
        %163 = arith.subf %cst_2, %162 : tensor<16xf32>
        %164 = arith.subi %arg3, %c48_i32 : i32
        %165 = tt.splat %164 : i32 -> tensor<16xi32>
        %166 = arith.cmpi slt, %9, %165 : tensor<16xi32>
        %167 = arith.select %166, %163, %cst_2 : tensor<16xi1>, tensor<16xf32>
        %168 = tt.expand_dims %167 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %169 = tt.broadcast %168 : tensor<16x1xf32> -> tensor<16x8xf32>
        %170 = arith.mulf %169, %arg4 : tensor<16x8xf32>
        %171 = "tt.reduce"(%170) <{axis = 0 : i32}> ({
        ^bb0(%arg5: f32, %arg6: f32):
          %179 = arith.addf %arg5, %arg6 : f32
          tt.reduce.return %179 : f32
        }) : (tensor<16x8xf32>) -> tensor<8xf32>
        %extracted_slice_54 = tensor.extract_slice %167[%2] [8] [1] {should_kept_slice} : tensor<16xf32> to tensor<8xf32>
        %172 = arith.addf %extracted_slice_54, %171 : tensor<8xf32>
        %173 = arith.cmpi eq, %9, %165 : tensor<16xi32>
        %174 = tt.expand_dims %173 {axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
        %175 = tt.expand_dims %172 {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
        %176 = tt.broadcast %175 : tensor<1x8xf32> -> tensor<16x8xf32>
        %177 = tt.broadcast %174 : tensor<16x1xi1> -> tensor<16x8xi1>
        %178 = arith.select %177, %176, %arg4 : tensor<16x8xi1>, tensor<16x8xf32>
        scf.yield %178 : tensor<16x8xf32>
      }
      %54 = arith.uitofp %15 : tensor<16x8xi1> to tensor<16x8xf32>
      %55 = arith.addf %47, %54 : tensor<16x8xf32>
      %56 = arith.addf %49, %54 : tensor<16x8xf32>
      %57 = arith.addf %51, %54 : tensor<16x8xf32>
      %58 = arith.addf %53, %54 : tensor<16x8xf32>
      %59 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%24, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %60 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %61 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%27, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %62 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%30, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %63 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%30, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %64 = tt.make_tensor_ptr %19, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%30, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %65 = tt.load %59 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %66 = tt.load %60 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %67 = tt.load %61 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %68 = tt.load %62 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %69 = tt.load %63 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %70 = tt.load %64 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %71 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview = memref.subview %71[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %56 in writable %subview : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc = memref.alloc() : memref<16x16xf32>
      memref.copy %71, %alloc : memref<16x16xf32> to memref<16x16xf32>
      %72 = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
      %73 = tt.dot %72, %65, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %74 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %73 in writable %74 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_6 = memref.subview %74[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_7 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_6, %alloc_7 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %75 = bufferization.to_tensor %alloc_7 restrict writable : memref<8x16xf32>
      %76 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_8 = memref.subview %76[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %75 in writable %subview_8 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_9 = memref.alloc() : memref<16x16xf32>
      memref.copy %76, %alloc_9 : memref<16x16xf32> to memref<16x16xf32>
      %77 = bufferization.to_tensor %alloc_9 restrict writable : memref<16x16xf32>
      %78 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_10 = memref.subview %78[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %55 in writable %subview_10 : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_11 = memref.alloc() : memref<16x16xf32>
      memref.copy %78, %alloc_11 : memref<16x16xf32> to memref<16x16xf32>
      %79 = bufferization.to_tensor %alloc_11 restrict writable : memref<16x16xf32>
      %80 = tt.dot %77, %79, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %81 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %80 in writable %81 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_12 = memref.subview %81[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_13 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_12, %alloc_13 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %82 = bufferization.to_tensor %alloc_13 restrict writable : memref<8x16xf32>
      %83 = arith.subf %cst, %82 : tensor<8x16xf32>
      %84 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_14 = memref.subview %84[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %57 in writable %subview_14 : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_15 = memref.alloc() : memref<16x16xf32>
      memref.copy %84, %alloc_15 : memref<16x16xf32> to memref<16x16xf32>
      %85 = bufferization.to_tensor %alloc_15 restrict writable : memref<16x16xf32>
      %86 = tt.dot %85, %67, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %87 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %86 in writable %87 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_16 = memref.subview %87[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_17 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_16, %alloc_17 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %88 = bufferization.to_tensor %alloc_17 restrict writable : memref<8x16xf32>
      %89 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_18 = memref.subview %89[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %88 in writable %subview_18 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_19 = memref.alloc() : memref<16x16xf32>
      memref.copy %89, %alloc_19 : memref<16x16xf32> to memref<16x16xf32>
      %90 = bufferization.to_tensor %alloc_19 restrict writable : memref<16x16xf32>
      %91 = tt.dot %90, %72, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %92 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %91 in writable %92 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_20 = memref.subview %92[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_21 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_20, %alloc_21 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %93 = bufferization.to_tensor %alloc_21 restrict writable : memref<8x16xf32>
      %94 = arith.subf %cst, %93 : tensor<8x16xf32>
      %95 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_22 = memref.subview %95[0, %2] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %58 in writable %subview_22 : (tensor<16x8xf32>, memref<16x8xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_23 = memref.alloc() : memref<16x16xf32>
      memref.copy %95, %alloc_23 : memref<16x16xf32> to memref<16x16xf32>
      %96 = bufferization.to_tensor %alloc_23 restrict writable : memref<16x16xf32>
      %97 = tt.dot %96, %70, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %98 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %97 in writable %98 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_24 = memref.subview %98[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_25 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_24, %alloc_25 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %99 = bufferization.to_tensor %alloc_25 restrict writable : memref<8x16xf32>
      %100 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_26 = memref.subview %100[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %99 in writable %subview_26 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_27 = memref.alloc() : memref<16x16xf32>
      memref.copy %100, %alloc_27 : memref<16x16xf32> to memref<16x16xf32>
      %101 = bufferization.to_tensor %alloc_27 restrict writable : memref<16x16xf32>
      %102 = tt.dot %101, %85, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %103 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %102 in writable %103 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_28 = memref.subview %103[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_29 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_28, %alloc_29 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %104 = bufferization.to_tensor %alloc_29 restrict writable : memref<8x16xf32>
      %105 = arith.subf %cst, %104 : tensor<8x16xf32>
      %106 = tt.dot %66, %79, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %107 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_30 = memref.subview %107[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %83 in writable %subview_30 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_31 = memref.alloc() : memref<16x16xf32>
      memref.copy %107, %alloc_31 : memref<16x16xf32> to memref<16x16xf32>
      %108 = bufferization.to_tensor %alloc_31 restrict writable : memref<16x16xf32>
      %109 = tt.dot %67, %108, %106 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %110 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %109 in writable %110 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_32 = memref.subview %110[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_33 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_32, %alloc_33 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %111 = bufferization.to_tensor %alloc_33 restrict writable : memref<8x16xf32>
      %112 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_34 = memref.subview %112[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %111 in writable %subview_34 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_35 = memref.alloc() : memref<16x16xf32>
      memref.copy %112, %alloc_35 : memref<16x16xf32> to memref<16x16xf32>
      %113 = bufferization.to_tensor %alloc_35 restrict writable : memref<16x16xf32>
      %114 = tt.dot %85, %113, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %115 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %114 in writable %115 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_36 = memref.subview %115[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_37 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_36, %alloc_37 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %116 = bufferization.to_tensor %alloc_37 restrict writable : memref<8x16xf32>
      %117 = arith.subf %cst, %116 : tensor<8x16xf32>
      %118 = tt.dot %69, %72, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %119 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_38 = memref.subview %119[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %94 in writable %subview_38 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_39 = memref.alloc() : memref<16x16xf32>
      memref.copy %119, %alloc_39 : memref<16x16xf32> to memref<16x16xf32>
      %120 = bufferization.to_tensor %alloc_39 restrict writable : memref<16x16xf32>
      %121 = tt.dot %70, %120, %118 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %122 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %121 in writable %122 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_40 = memref.subview %122[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_41 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_40, %alloc_41 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %123 = bufferization.to_tensor %alloc_41 restrict writable : memref<8x16xf32>
      %124 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_42 = memref.subview %124[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %123 in writable %subview_42 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_43 = memref.alloc() : memref<16x16xf32>
      memref.copy %124, %alloc_43 : memref<16x16xf32> to memref<16x16xf32>
      %125 = bufferization.to_tensor %alloc_43 restrict writable : memref<16x16xf32>
      %126 = tt.dot %96, %125, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %127 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %126 in writable %127 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_44 = memref.subview %127[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_45 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_44, %alloc_45 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %128 = bufferization.to_tensor %alloc_45 restrict writable : memref<8x16xf32>
      %129 = arith.subf %cst, %128 : tensor<8x16xf32>
      %130 = tt.dot %68, %79, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %131 = tt.dot %69, %108, %130 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %132 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_46 = memref.subview %132[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %117 in writable %subview_46 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_47 = memref.alloc() : memref<16x16xf32>
      memref.copy %132, %alloc_47 : memref<16x16xf32> to memref<16x16xf32>
      %133 = bufferization.to_tensor %alloc_47 restrict writable : memref<16x16xf32>
      %134 = tt.dot %70, %133, %131 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %135 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %134 in writable %135 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_48 = memref.subview %135[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_49 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_48, %alloc_49 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %136 = bufferization.to_tensor %alloc_49 restrict writable : memref<8x16xf32>
      %137 = memref_ext.alloc_workspace() : memref<16x16xf32>
      %subview_50 = memref.subview %137[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      bufferization.materialize_in_destination %136 in writable %subview_50 : (tensor<8x16xf32>, memref<8x16xf32, strided<[16, 1], offset: ?>>) -> ()
      %alloc_51 = memref.alloc() : memref<16x16xf32>
      memref.copy %137, %alloc_51 : memref<16x16xf32> to memref<16x16xf32>
      %138 = bufferization.to_tensor %alloc_51 restrict writable : memref<16x16xf32>
      %139 = tt.dot %96, %138, %cst_1 {tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %140 = memref_ext.alloc_workspace() : memref<16x16xf32>
      bufferization.materialize_in_destination %139 in writable %140 : (tensor<16x16xf32>, memref<16x16xf32>) -> ()
      %subview_52 = memref.subview %140[%2, 0] [8, 16] [1, 1] : memref<16x16xf32> to memref<8x16xf32, strided<[16, 1], offset: ?>>
      %alloc_53 = memref.alloc() : memref<8x16xf32>
      memref.copy %subview_52, %alloc_53 : memref<8x16xf32, strided<[16, 1], offset: ?>> to memref<8x16xf32>
      %141 = bufferization.to_tensor %alloc_53 restrict writable : memref<8x16xf32>
      %142 = arith.subf %cst, %141 : tensor<8x16xf32>
      %143 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%21, %3] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
      %144 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%24, %25] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
      %145 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%27, %28] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
      %146 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%30, %31] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
      %147 = arith.addi %24, %3 : i32
      %148 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%147, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
      %149 = arith.addi %27, %3 : i32
      %150 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%149, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
      %151 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%149, %c16_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
      %152 = arith.addi %30, %3 : i32
      %153 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%152, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
      %154 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%152, %c16_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
      %155 = tt.make_tensor_ptr %20, [%22, %c64_i64], [%c2048_i64, %c1_i64], [%152, %c32_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
      tt.store %143, %55 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
      tt.store %144, %56 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
      tt.store %145, %57 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
      tt.store %146, %58 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
      tt.store %148, %83 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
      tt.store %150, %117 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
      tt.store %151, %94 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
      tt.store %153, %142 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
      tt.store %154, %129 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
      tt.store %155, %105 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
      tt.return
    }
  }
}
