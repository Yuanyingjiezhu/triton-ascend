// NOTE: Snapshot regression coverage for kernels collected under gaoyou/0701.
// NOTE: Regenerate deliberately when --start-slice output changes.
// RUN: triton-opt --start-slice -split-input-file %s | FileCheck %s --match-full-lines

// -----
// Source: gaoyou/0701/bwd_dqkwg/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 8192)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 4096)>
// CHECK-NEXT: #map2 = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_bwd_kernel_dqkwg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: f32, %arg11: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xbf16>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_3 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map1()[%1]
// CHECK-NEXT:     %4 = affine.apply #map2()[%1]
// CHECK-NEXT:     %5 = affine.apply #map2()[%1]
// CHECK-NEXT:     %6 = affine.apply #map2()[%1]
// CHECK-NEXT:     %7 = arith.index_cast %6 : index to i32
// CHECK-NEXT:     %8 = affine.apply #map2()[%1]
// CHECK-NEXT:     %9 = arith.index_cast %8 : index to i32
// CHECK-NEXT:     %10 = affine.apply #map2()[%1]
// CHECK-NEXT:     %11 = arith.index_cast %10 : index to i32
// CHECK-NEXT:     %12 = tt.get_program_id x : i32
// CHECK-NEXT:     %13 = tt.get_program_id y : i32
// CHECK-NEXT:     %14 = tt.get_program_id z : i32
// CHECK-NEXT:     %15 = arith.divsi %14, %c32_i32 : i32
// CHECK-NEXT:     %16 = arith.remsi %14, %c32_i32 : i32
// CHECK-NEXT:     %17 = arith.addi %arg11, %c63_i32 : i32
// CHECK-NEXT:     %18 = arith.divsi %17, %c64_i32 : i32
// CHECK-NEXT:     %19 = arith.muli %15, %18 : i32
// CHECK-NEXT:     %20 = arith.addi %19, %13 : i32
// CHECK-NEXT:     %21 = arith.muli %15, %arg11 : i32
// CHECK-NEXT:     %22 = arith.muli %21, %c32_i32 : i32
// CHECK-NEXT:     %23 = arith.addi %22, %16 : i32
// CHECK-NEXT:     %24 = arith.muli %23, %c128_i32 : i32
// CHECK-NEXT:     %25 = tt.addptr %arg2, %24 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %26 = tt.addptr %arg5, %24 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %27 = arith.muli %20, %c32_i32 : i32
// CHECK-NEXT:     %28 = arith.addi %27, %16 : i32
// CHECK-NEXT:     %29 = arith.extsi %28 : i32 to i64
// CHECK-NEXT:     %30 = arith.muli %29, %c16384_i64 : i64
// CHECK-NEXT:     %31 = tt.addptr %arg4, %30 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %32 = tt.addptr %arg6, %30 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %33 = tt.addptr %arg0, %24 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %34 = tt.addptr %arg1, %24 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %35 = tt.addptr %arg7, %24 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %36 = tt.addptr %arg8, %24 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %37 = arith.muli %12, %arg11 : i32
// CHECK-NEXT:     %38 = arith.muli %37, %c32_i32 : i32
// CHECK-NEXT:     %39 = tt.addptr %arg9, %38 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %40 = arith.muli %13, %c64_i32 : i32
// CHECK-NEXT:     %41 = arith.extsi %arg11 : i32 to i64
// CHECK-NEXT:     %42 = tt.make_tensor_ptr %25, [%41, %c128_i64], [%c4096_i64, %c1_i64], [%40, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %43 = tt.make_tensor_ptr %26, [%41, %c128_i64], [%c4096_i64, %c1_i64], [%40, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %44 = arith.muli %12, %c128_i32 : i32
// CHECK-NEXT:     %45 = tt.make_tensor_ptr %31, [%c128_i64, %c128_i64], [%c1_i64, %c128_i64], [%c0_i32, %44] {order = array<i32: 0, 1>} : <tensor<128x128xbf16>>
// CHECK-NEXT:     %46 = tt.make_tensor_ptr %32, [%c128_i64, %c128_i64], [%c1_i64, %c128_i64], [%c0_i32, %44] {order = array<i32: 0, 1>} : <tensor<128x128xbf16>>
// CHECK-NEXT:     %47 = tt.load %42 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %48 = tt.load %43 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %49 = tt.load %45 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:     %50 = tt.load %46 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:     %51 = arith.mulf %49, %50 {DataUse} : tensor<128x128xbf16>
// CHECK-NEXT:     %52 = arith.extf %51 {DataUse} : tensor<128x128xbf16> to tensor<128x128xf32>
// CHECK-NEXT:     %53 = tt.reshape %52 allow_reorder {DataUse} : tensor<128x128xf32> -> tensor<16384xf32>
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %53[%2] [8192] [1] {to_be_bubbled_slice} : tensor<16384xf32> to tensor<8192xf32>
// CHECK-NEXT:     %54 = "tt.reduce"(%extracted_slice) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %156 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %156 : f32
// CHECK-NEXT:     }) {tiled_op} : (tensor<8192xf32>) -> f32
// CHECK-NEXT:     %55 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:     %inserted = tensor.insert %54 into %55[%1] {vv_communication} : tensor<2xf32>
// CHECK-NEXT:     %56 = "tt.reduce"(%inserted) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %156 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %156 : f32
// CHECK-NEXT:     }) {tiled_op} : (tensor<2xf32>) -> f32
// CHECK-NEXT:     %57 = arith.addf %56, %cst_3 : f32
// CHECK-NEXT:     %58 = tt.trans %47 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:     %59 = tt.dot %48, %58, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %60 = tt.dot %48, %49, %cst_0 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %61 = tt.dot %47, %50, %cst_0 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     gpu.barrier
// CHECK-NEXT:     %62 = tt.make_tensor_ptr %33, [%41, %c128_i64], [%c4096_i64, %c1_i64], [%40, %44] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %63 = tt.make_tensor_ptr %34, [%41, %c128_i64], [%c4096_i64, %c1_i64], [%40, %44] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %64 = tt.load %62 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %65 = tt.load %63 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %66 = arith.addi %40, %7 : i32
// CHECK-NEXT:     %67 = tt.make_tensor_ptr %35, [%41, %c128_i64], [%c4096_i64, %c1_i64], [%66, %44] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x128xbf16>>
// CHECK-NEXT:     %68 = arith.addi %40, %9 : i32
// CHECK-NEXT:     %69 = tt.make_tensor_ptr %36, [%41, %c128_i64], [%c4096_i64, %c1_i64], [%68, %44] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x128xbf16>>
// CHECK-NEXT:     %70 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %71 = tt.splat %40 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %72 = arith.addi %71, %70 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %73 = tt.splat %arg11 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %74 = arith.cmpi slt, %72, %73 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %75 = tt.expand_dims %72 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %76 = tt.expand_dims %72 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %77 = tt.broadcast %75 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %78 = tt.broadcast %76 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %79 = arith.cmpi sge, %77, %78 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %80 = tt.expand_dims %74 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %81 = tt.expand_dims %74 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %82 = tt.broadcast %80 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %83 = tt.broadcast %81 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %84 = arith.andi %82, %83 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %85 = arith.andi %79, %84 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %86 = tt.addptr %arg3, %23 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %87 = tt.addptr %39, %23 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %88 = tt.make_tensor_ptr %86, [%41], [%c32_i64], [%40] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %89 = tt.load %88 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %90 = arith.addi %40, %c64_i32 : i32
// CHECK-NEXT:     %91 = arith.minsi %90, %arg11 : i32
// CHECK-NEXT:     %92 = arith.subi %91, %c1_i32 : i32
// CHECK-NEXT:     %93 = arith.muli %92, %c32_i32 : i32
// CHECK-NEXT:     %94 = tt.addptr %86, %93 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %95 = tt.load %94 : !tt.ptr<bf16>
// CHECK-NEXT:     %96 = arith.extf %95 : bf16 to f32
// CHECK-NEXT:     %97 = math.exp %96 : f32
// CHECK-NEXT:     %98 = arith.mulf %57, %97 : f32
// CHECK-NEXT:     %99 = arith.extf %89 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %100 = math.exp %99 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %101 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %102 = tt.broadcast %101 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %103 = arith.mulf %60, %102 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %104 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %105 = arith.mulf %103, %104 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %106 = arith.subf %cst_1, %89 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:     %107 = tt.splat %95 {DataUse} : bf16 -> tensor<64xbf16>
// CHECK-NEXT:     %108 = arith.addf %106, %107 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:     %109 = arith.extf %108 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %110 = math.exp %109 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %111 = arith.select %74, %110, %cst_2 {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:     %112 = tt.expand_dims %111 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %113 = tt.broadcast %112 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %114 = arith.mulf %61, %113 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %115 = arith.extf %65 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
// CHECK-NEXT:     %116 = arith.mulf %114, %115 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %117 = tt.reshape %116 allow_reorder {DataUse} : tensor<64x128xf32> -> tensor<8192xf32>
// CHECK-NEXT:     %extracted_slice_4 = tensor.extract_slice %117[%3] [4096] [1] {to_be_bubbled_slice} : tensor<8192xf32> to tensor<4096xf32>
// CHECK-NEXT:     %118 = "tt.reduce"(%extracted_slice_4) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %156 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %156 : f32
// CHECK-NEXT:     }) {tiled_op} : (tensor<4096xf32>) -> f32
// CHECK-NEXT:     %119 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:     %inserted_5 = tensor.insert %118 into %119[%1] {vv_communication} : tensor<2xf32>
// CHECK-NEXT:     %120 = "tt.reduce"(%inserted_5) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %156 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %156 : f32
// CHECK-NEXT:     }) {tiled_op} : (tensor<2xf32>) -> f32
// CHECK-NEXT:     %121 = arith.addf %98, %120 : f32
// CHECK-NEXT:     %122 = tt.expand_dims %89 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %123 = tt.expand_dims %89 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:     %124 = tt.broadcast %122 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %125 = tt.broadcast %123 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %126 = arith.subf %124, %125 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:     %127 = arith.extf %126 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %128 = math.exp %127 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %129 = arith.mulf %59, %128 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %130 = arith.select %85, %129, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %131 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     %132 = arith.mulf %130, %131 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %133 = arith.truncf %132 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     %extracted_slice_6 = tensor.extract_slice %133[%4, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:     %134 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:     %inserted_slice = tensor.insert_slice %extracted_slice_6 into %134[%4, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:     %135 = tt.dot %inserted_slice, %65, %cst_0 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %136 = arith.addf %135, %105 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:     %extracted_slice_7 = tensor.extract_slice %133[%5, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:     %137 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:     %inserted_slice_8 = tensor.insert_slice %extracted_slice_7 into %137[%5, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:     %138 = tt.trans %inserted_slice_8 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %139 = tt.dot %138, %64, %cst_0 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %140 = arith.addf %139, %114 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:     %141 = arith.extf %64 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
// CHECK-NEXT:     %142 = arith.mulf %136, %141 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %143 = "tt.reduce"(%142) <{axis = 1 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %156 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %156 : f32
// CHECK-NEXT:     }) {DataUse} : (tensor<64x128xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %144 = arith.mulf %140, %115 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %145 = "tt.reduce"(%144) <{axis = 1 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %156 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %156 : f32
// CHECK-NEXT:     }) {DataUse} : (tensor<64x128xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %146 = arith.subf %143, %145 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %147 = arith.addi %40, %11 : i32
// CHECK-NEXT:     %148 = tt.make_tensor_ptr %87, [%41], [%c32_i64], [%147] {order = array<i32: 0>, tiled_op} : <tensor<32xf32>>
// CHECK-NEXT:     %149 = tt.splat %92 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %150 = arith.cmpi slt, %72, %149 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %151 = tt.splat %121 {DataUse} : f32 -> tensor<64xf32>
// CHECK-NEXT:     %152 = arith.addf %146, %151 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %153 = arith.select %150, %146, %152 {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:     %154 = arith.truncf %136 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     %extracted_slice_9 = tensor.extract_slice %154[%6, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:     tt.store %67, %extracted_slice_9 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x128xbf16>>
// CHECK-NEXT:     %155 = arith.truncf %140 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     %extracted_slice_10 = tensor.extract_slice %155[%8, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:     tt.store %69, %extracted_slice_10 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x128xbf16>>
// CHECK-NEXT:     %extracted_slice_11 = tensor.extract_slice %153[%10] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK-NEXT:     tt.store %148, %extracted_slice_11 {boundaryCheck = array<i32: 0>, tiled_op} : !tt.ptr<tensor<32xf32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_bwd_kernel_dqkwg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: f32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xbf16>
    %c32_i64 = arith.constant 32 : i64
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
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
    %35 = tt.load %30 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %36 = tt.load %31 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %37 = tt.load %33 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
    %38 = tt.load %34 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
    %39 = arith.mulf %37, %38 {DataUse} : tensor<128x128xbf16>
    %40 = arith.extf %39 {DataUse} : tensor<128x128xbf16> to tensor<128x128xf32>
    %41 = tt.reshape %40 allow_reorder {DataUse} : tensor<128x128xf32> -> tensor<16384xf32>
    %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) : (tensor<16384xf32>) -> f32
    %43 = arith.addf %42, %cst : f32
    %44 = tt.trans %35 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
    %45 = tt.dot %36, %44, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
    %46 = tt.dot %36, %37, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
    %47 = tt.dot %35, %38, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
    gpu.barrier
    %48 = tt.make_tensor_ptr %21, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %49 = tt.make_tensor_ptr %22, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %50 = tt.load %48 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %51 = tt.load %49 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %52 = tt.make_tensor_ptr %23, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %53 = tt.make_tensor_ptr %24, [%29, %c128_i64], [%c4096_i64, %c1_i64], [%28, %32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %54 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %55 = tt.splat %28 {DataUse} : i32 -> tensor<64xi32>
    %56 = arith.addi %55, %54 {DataUse} : tensor<64xi32>
    %57 = tt.splat %arg11 {DataUse} : i32 -> tensor<64xi32>
    %58 = arith.cmpi slt, %56, %57 {DataUse} : tensor<64xi32>
    %59 = tt.expand_dims %56 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %60 = tt.expand_dims %56 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %61 = tt.broadcast %59 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
    %62 = tt.broadcast %60 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %63 = arith.cmpi sge, %61, %62 {DataUse} : tensor<64x64xi32>
    %64 = tt.expand_dims %58 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %65 = tt.expand_dims %58 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %66 = tt.broadcast %64 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
    %67 = tt.broadcast %65 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
    %68 = arith.andi %66, %67 {DataUse} : tensor<64x64xi1>
    %69 = arith.andi %63, %68 {DataUse} : tensor<64x64xi1>
    %70 = tt.addptr %arg3, %11 : !tt.ptr<bf16>, i32
    %71 = tt.addptr %27, %11 : !tt.ptr<f32>, i32
    %72 = tt.make_tensor_ptr %70, [%29], [%c32_i64], [%28] {order = array<i32: 0>} : <tensor<64xbf16>>
    %73 = tt.load %72 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %74 = arith.addi %28, %c64_i32 : i32
    %75 = arith.minsi %74, %arg11 : i32
    %76 = arith.subi %75, %c1_i32 : i32
    %77 = arith.muli %76, %c32_i32 : i32
    %78 = tt.addptr %70, %77 : !tt.ptr<bf16>, i32
    %79 = tt.load %78 : !tt.ptr<bf16>
    %80 = arith.extf %79 : bf16 to f32
    %81 = math.exp %80 : f32
    %82 = arith.mulf %43, %81 : f32
    %83 = arith.extf %73 {DataUse} : tensor<64xbf16> to tensor<64xf32>
    %84 = math.exp %83 {DataUse} : tensor<64xf32>
    %85 = tt.expand_dims %84 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %86 = tt.broadcast %85 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
    %87 = arith.mulf %46, %86 {DataUse} : tensor<64x128xf32>
    %88 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x128xf32>
    %89 = arith.mulf %87, %88 {DataUse} : tensor<64x128xf32>
    %90 = arith.subf %cst_1, %73 {DataUse} : tensor<64xbf16>
    %91 = tt.splat %79 {DataUse} : bf16 -> tensor<64xbf16>
    %92 = arith.addf %90, %91 {DataUse} : tensor<64xbf16>
    %93 = arith.extf %92 {DataUse} : tensor<64xbf16> to tensor<64xf32>
    %94 = math.exp %93 {DataUse} : tensor<64xf32>
    %95 = arith.select %58, %94, %cst_0 {DataUse} : tensor<64xi1>, tensor<64xf32>
    %96 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %97 = tt.broadcast %96 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
    %98 = arith.mulf %47, %97 {DataUse} : tensor<64x128xf32>
    %99 = arith.extf %51 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
    %100 = arith.mulf %98, %99 {DataUse} : tensor<64x128xf32>
    %101 = tt.reshape %100 allow_reorder {DataUse} : tensor<64x128xf32> -> tensor<8192xf32>
    %102 = "tt.reduce"(%101) <{axis = 0 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) : (tensor<8192xf32>) -> f32
    %103 = arith.addf %82, %102 : f32
    %104 = tt.expand_dims %73 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %105 = tt.expand_dims %73 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
    %106 = tt.broadcast %104 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %107 = tt.broadcast %105 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
    %108 = arith.subf %106, %107 {DataUse} : tensor<64x64xbf16>
    %109 = arith.extf %108 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
    %110 = math.exp %109 {DataUse} : tensor<64x64xf32>
    %111 = arith.mulf %45, %110 {DataUse} : tensor<64x64xf32>
    %112 = arith.select %69, %111, %cst_3 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
    %113 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x64xf32>
    %114 = arith.mulf %112, %113 {DataUse} : tensor<64x64xf32>
    %115 = arith.truncf %114 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
    %116 = tt.dot %115, %51, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %117 = arith.addf %116, %89 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
    %118 = tt.trans %115 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
    %119 = tt.dot %118, %50, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %120 = arith.addf %119, %98 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
    %121 = arith.extf %50 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
    %122 = arith.mulf %117, %121 {DataUse} : tensor<64x128xf32>
    %123 = "tt.reduce"(%122) <{axis = 1 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) {DataUse} : (tensor<64x128xf32>) -> tensor<64xf32>
    %124 = arith.mulf %120, %99 {DataUse} : tensor<64x128xf32>
    %125 = "tt.reduce"(%124) <{axis = 1 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %135 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %135 : f32
    }) {DataUse} : (tensor<64x128xf32>) -> tensor<64xf32>
    %126 = arith.subf %123, %125 {DataUse} : tensor<64xf32>
    %127 = tt.make_tensor_ptr %71, [%29], [%c32_i64], [%28] {order = array<i32: 0>} : <tensor<64xf32>>
    %128 = tt.splat %76 {DataUse} : i32 -> tensor<64xi32>
    %129 = arith.cmpi slt, %56, %128 {DataUse} : tensor<64xi32>
    %130 = tt.splat %103 {DataUse} : f32 -> tensor<64xf32>
    %131 = arith.addf %126, %130 {DataUse} : tensor<64xf32>
    %132 = arith.select %129, %126, %131 {DataUse} : tensor<64xi1>, tensor<64xf32>
    %133 = arith.truncf %117 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %52, %133 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    %134 = arith.truncf %120 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %53, %134 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    tt.store %127, %132 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xf32>>
    tt.return
  }
}


// -----
// Source: gaoyou/0701/chunk_o/fwd_h_sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_fwd_kernel_h(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<32> : tensor<64xi32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xbf16>
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = arith.index_cast %3 : index to i32
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = tt.get_program_id x : i32
// CHECK-NEXT:     %7 = tt.get_program_id y : i32
// CHECK-NEXT:     %8 = tt.get_program_id z : i32
// CHECK-NEXT:     %9 = arith.divsi %8, %c32_i32 : i32
// CHECK-NEXT:     %10 = arith.remsi %8, %c32_i32 : i32
// CHECK-NEXT:     %11 = arith.muli %9, %arg4 : i32
// CHECK-NEXT:     %12 = arith.addi %arg4, %c63_i32 : i32
// CHECK-NEXT:     %13 = arith.divsi %12, %c64_i32 : i32
// CHECK-NEXT:     %14 = arith.muli %9, %13 : i32
// CHECK-NEXT:     %15 = arith.muli %11, %c32_i32 : i32
// CHECK-NEXT:     %16 = arith.addi %15, %10 : i32
// CHECK-NEXT:     %17 = arith.muli %16, %c128_i32 : i32
// CHECK-NEXT:     %18 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %19 = arith.muli %6, %c64_i32 : i32
// CHECK-NEXT:     %20 = arith.extsi %arg4 : i32 to i64
// CHECK-NEXT:     %21 = tt.addptr %arg1, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %22 = arith.muli %7, %c64_i32 : i32
// CHECK-NEXT:     %23 = tt.addptr %arg3, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %24 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %25 = tt.splat %23 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:     %26 = tt.splat %10 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %27 = tt.splat %arg4 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %28 = scf.for %arg5 = %c0_i32 to %13 step %c1_i32 iter_args(%arg6 = %cst_0) -> (tensor<64x64xf32>)  : i32 {
// CHECK-NEXT:       %29 = arith.muli %arg5, %c64_i32 : i32
// CHECK-NEXT:       %30 = tt.make_tensor_ptr %18, [%c128_i64, %20], [%c1_i64, %c4096_i64], [%19, %29] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %31 = tt.make_tensor_ptr %21, [%20, %c128_i64], [%c4096_i64, %c1_i64], [%29, %22] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %32 = arith.addi %14, %arg5 : i32
// CHECK-NEXT:       %33 = arith.muli %32, %c32_i32 : i32
// CHECK-NEXT:       %34 = arith.addi %33, %10 : i32
// CHECK-NEXT:       %35 = arith.extsi %34 : i32 to i64
// CHECK-NEXT:       %36 = arith.muli %35, %c16384_i64 : i64
// CHECK-NEXT:       %37 = tt.addptr %arg2, %36 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %38 = arith.addi %19, %4 : i32
// CHECK-NEXT:       %39 = tt.make_tensor_ptr %37, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%38, %22] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:       %40 = arith.truncf %arg6 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %40[%3, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       tt.store %39, %extracted_slice {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:       %41 = tt.load %30 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %42 = tt.load %31 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %43 = arith.addi %arg5, %c1_i32 : i32
// CHECK-NEXT:       %44 = arith.muli %43, %c64_i32 : i32
// CHECK-NEXT:       %45 = arith.minsi %44, %arg4 : i32
// CHECK-NEXT:       %46 = arith.subi %45, %c1_i32 : i32
// CHECK-NEXT:       %47 = arith.muli %46, %c32_i32 : i32
// CHECK-NEXT:       %48 = tt.addptr %23, %47 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %49 = tt.addptr %48, %10 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %50 = tt.load %49 : !tt.ptr<bf16>
// CHECK-NEXT:       %51 = tt.splat %29 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %52 = arith.addi %51, %24 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %53 = arith.muli %52, %cst {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %54 = tt.addptr %25, %53 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %55 = tt.addptr %54, %26 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %56 = arith.cmpi slt, %52, %27 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %57 = tt.load %55, %56, %cst_1 {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %58 = arith.extf %50 : bf16 to f32
// CHECK-NEXT:       %59 = math.exp %58 : f32
// CHECK-NEXT:       %60 = tt.splat %59 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:       %61 = arith.mulf %arg6, %60 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %62 = tt.splat %50 {DataUse} : bf16 -> tensor<64xbf16>
// CHECK-NEXT:       %63 = arith.subf %62, %57 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:       %64 = arith.extf %63 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:       %65 = math.exp %64 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %66 = tt.expand_dims %65 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %67 = arith.extf %42 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %68 = tt.broadcast %66 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %69 = arith.mulf %67, %68 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %70 = arith.truncf %69 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %extracted_slice_2 = tensor.extract_slice %70[%5, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       %71 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:       %inserted_slice = tensor.insert_slice %extracted_slice_2 into %71[%5, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:       %72 = tt.dot %41, %inserted_slice, %cst_0 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %73 = arith.addf %72, %61 {DataUse, triton_cv12.add_from_dot} : tensor<64x64xf32>
// CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %73[%2, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xf32> to tensor<32x64xf32>
// CHECK-NEXT:       %74 = tensor.empty() : tensor<64x64xf32>
// CHECK-NEXT:       %inserted_slice_4 = tensor.insert_slice %extracted_slice_3 into %74[%2, 0] [32, 64] [1, 1] {to_be_eliminated_slice} : tensor<32x64xf32> into tensor<64x64xf32>
// CHECK-NEXT:       scf.yield %inserted_slice_4 : tensor<64x64xf32>
// CHECK-NEXT:     } {Undefined, tiled_op}
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_fwd_kernel_h(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xbf16>
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
    %cst_1 = arith.constant {MetaUse} dense<32> : tensor<64xi32>
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
    %18 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %19 = tt.splat %17 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
    %20 = tt.splat %4 {MetaUse} : i32 -> tensor<64xi32>
    %21 = tt.splat %arg4 {MetaUse} : i32 -> tensor<64xi32>
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
      %33 = arith.truncf %arg6 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
      tt.store %32, %33 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
      %34 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %35 = tt.load %25 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %36 = arith.addi %arg5, %c1_i32 : i32
      %37 = arith.muli %36, %c64_i32 : i32
      %38 = arith.minsi %37, %arg4 : i32
      %39 = arith.subi %38, %c1_i32 : i32
      %40 = arith.muli %39, %c32_i32 : i32
      %41 = tt.addptr %17, %40 : !tt.ptr<bf16>, i32
      %42 = tt.addptr %41, %4 : !tt.ptr<bf16>, i32
      %43 = tt.load %42 : !tt.ptr<bf16>
      %44 = tt.splat %23 {MetaUse} : i32 -> tensor<64xi32>
      %45 = arith.addi %44, %18 {MetaUse} : tensor<64xi32>
      %46 = arith.muli %45, %cst_1 {MetaUse} : tensor<64xi32>
      %47 = tt.addptr %19, %46 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %48 = tt.addptr %47, %20 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %49 = arith.cmpi slt, %45, %21 {MetaUse} : tensor<64xi32>
      %50 = tt.load %48, %49, %cst {DataUse} : tensor<64x!tt.ptr<bf16>>
      %51 = arith.extf %43 : bf16 to f32
      %52 = math.exp %51 : f32
      %53 = tt.splat %52 {DataUse} : f32 -> tensor<64x64xf32>
      %54 = arith.mulf %arg6, %53 {DataUse} : tensor<64x64xf32>
      %55 = tt.splat %43 {DataUse} : bf16 -> tensor<64xbf16>
      %56 = arith.subf %55, %50 {DataUse} : tensor<64xbf16>
      %57 = arith.extf %56 {DataUse} : tensor<64xbf16> to tensor<64xf32>
      %58 = math.exp %57 {DataUse} : tensor<64xf32>
      %59 = tt.expand_dims %58 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %60 = arith.extf %35 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
      %61 = tt.broadcast %59 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
      %62 = arith.mulf %60, %61 {DataUse} : tensor<64x64xf32>
      %63 = arith.truncf %62 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
      %64 = tt.dot %34, %63, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %65 = arith.addf %64, %54 {DataUse, triton_cv12.add_from_dot} : tensor<64x64xf32>
      scf.yield %65 : tensor<64x64xf32>
    } {Undefined}
    tt.return
  }
}


// -----
// Source: gaoyou/0701/chunk_o/fwd_o_sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_fwd_kernel_o(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = arith.index_cast %3 : index to i32
// CHECK-NEXT:     %5 = tt.get_program_id x : i32
// CHECK-NEXT:     %6 = tt.get_program_id y : i32
// CHECK-NEXT:     %7 = tt.get_program_id z : i32
// CHECK-NEXT:     %8 = arith.divsi %7, %c32_i32 : i32
// CHECK-NEXT:     %9 = arith.remsi %7, %c32_i32 : i32
// CHECK-NEXT:     %10 = arith.addi %arg7, %c63_i32 : i32
// CHECK-NEXT:     %11 = arith.divsi %10, %c64_i32 : i32
// CHECK-NEXT:     %12 = arith.muli %8, %11 : i32
// CHECK-NEXT:     %13 = arith.addi %12, %6 : i32
// CHECK-NEXT:     %14 = arith.muli %8, %arg7 : i32
// CHECK-NEXT:     %15 = arith.muli %14, %c32_i32 : i32
// CHECK-NEXT:     %16 = arith.addi %15, %9 : i32
// CHECK-NEXT:     %17 = arith.muli %16, %c128_i32 : i32
// CHECK-NEXT:     %18 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %19 = tt.addptr %arg1, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %20 = tt.addptr %arg2, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %21 = tt.addptr %arg5, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %22 = arith.muli %13, %c32_i32 : i32
// CHECK-NEXT:     %23 = arith.addi %22, %9 : i32
// CHECK-NEXT:     %24 = arith.extsi %23 : i32 to i64
// CHECK-NEXT:     %25 = arith.muli %24, %c16384_i64 : i64
// CHECK-NEXT:     %26 = tt.addptr %arg3, %25 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %27 = arith.muli %6, %c64_i32 : i32
// CHECK-NEXT:     %28 = arith.extsi %arg7 : i32 to i64
// CHECK-NEXT:     %29 = tt.make_tensor_ptr %18, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %30 = tt.make_tensor_ptr %19, [%c128_i64, %28], [%c1_i64, %c4096_i64], [%c0_i32, %27] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:     %31 = arith.muli %5, %c128_i32 : i32
// CHECK-NEXT:     %32 = tt.make_tensor_ptr %26, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %31] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:     %33 = tt.load %29 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %34 = tt.load %30 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:     %35 = tt.load %32 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:     %36 = tt.dot %33, %35, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %37 = tt.dot %33, %34, %cst_0 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %38 = tt.addptr %arg4, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %39 = tt.make_tensor_ptr %38, [%28], [%c32_i64], [%27] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %40 = tt.load %39 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %41 = arith.extf %40 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %42 = math.exp %41 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %43 = tt.expand_dims %42 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %44 = tt.broadcast %43 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %45 = arith.mulf %36, %44 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %46 = tt.expand_dims %40 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %47 = tt.expand_dims %40 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:     %48 = tt.broadcast %46 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %49 = tt.broadcast %47 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %50 = arith.subf %48, %49 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:     %51 = arith.extf %50 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %52 = math.exp %51 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %53 = arith.mulf %37, %52 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %54 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %55 = tt.splat %27 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %56 = arith.addi %55, %54 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %57 = tt.splat %arg7 {DataUse} : i32 -> tensor<64xi32>
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
// CHECK-NEXT:     %70 = arith.select %69, %53, %cst_0 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %71 = tt.make_tensor_ptr %20, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %31] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %72 = arith.addi %27, %4 : i32
// CHECK-NEXT:     %73 = tt.make_tensor_ptr %21, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%72, %31] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x128xbf16>>
// CHECK-NEXT:     %74 = tt.load %71 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %75 = tt.splat %arg6 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %76 = arith.mulf %45, %75 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %77 = arith.truncf %70 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %77[%2, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:     %78 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:     %inserted_slice = tensor.insert_slice %extracted_slice into %78[%2, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:     %79 = tt.dot %inserted_slice, %74, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %80 = arith.mulf %79, %75 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %81 = arith.addf %76, %80 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %82 = arith.truncf %81 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     %extracted_slice_1 = tensor.extract_slice %82[%3, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:     tt.store %73, %extracted_slice_1 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x128xbf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_fwd_kernel_o(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
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
    %28 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %29 = tt.load %25 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
    %30 = tt.load %27 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x128xbf16>>
    %31 = tt.dot %28, %30, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
    %32 = tt.dot %28, %29, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
    %33 = tt.addptr %arg4, %11 : !tt.ptr<bf16>, i32
    %34 = tt.make_tensor_ptr %33, [%23], [%c32_i64], [%22] {order = array<i32: 0>} : <tensor<64xbf16>>
    %35 = tt.load %34 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %36 = arith.extf %35 {DataUse} : tensor<64xbf16> to tensor<64xf32>
    %37 = math.exp %36 {DataUse} : tensor<64xf32>
    %38 = tt.expand_dims %37 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %39 = tt.broadcast %38 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
    %40 = arith.mulf %31, %39 {DataUse} : tensor<64x128xf32>
    %41 = tt.expand_dims %35 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %42 = tt.expand_dims %35 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
    %43 = tt.broadcast %41 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %44 = tt.broadcast %42 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
    %45 = arith.subf %43, %44 {DataUse} : tensor<64x64xbf16>
    %46 = arith.extf %45 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
    %47 = math.exp %46 {DataUse} : tensor<64x64xf32>
    %48 = arith.mulf %32, %47 {DataUse} : tensor<64x64xf32>
    %49 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %50 = tt.splat %22 {DataUse} : i32 -> tensor<64xi32>
    %51 = arith.addi %50, %49 {DataUse} : tensor<64xi32>
    %52 = tt.splat %arg7 {DataUse} : i32 -> tensor<64xi32>
    %53 = arith.cmpi slt, %51, %52 {DataUse} : tensor<64xi32>
    %54 = tt.expand_dims %51 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %55 = tt.expand_dims %51 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %56 = tt.broadcast %54 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
    %57 = tt.broadcast %55 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %58 = arith.cmpi sge, %56, %57 {DataUse} : tensor<64x64xi32>
    %59 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %60 = tt.expand_dims %53 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %61 = tt.broadcast %59 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
    %62 = tt.broadcast %60 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
    %63 = arith.andi %61, %62 {DataUse} : tensor<64x64xi1>
    %64 = arith.andi %58, %63 {DataUse} : tensor<64x64xi1>
    %65 = arith.select %64, %48, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
    %66 = tt.make_tensor_ptr %15, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %26] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %67 = tt.make_tensor_ptr %16, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%22, %26] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %68 = tt.load %66 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %69 = tt.splat %arg6 {DataUse} : f32 -> tensor<64x128xf32>
    %70 = arith.mulf %40, %69 {DataUse} : tensor<64x128xf32>
    %71 = arith.truncf %65 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
    %72 = tt.dot %71, %68, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %73 = arith.mulf %72, %69 {DataUse} : tensor<64x128xf32>
    %74 = arith.addf %70, %73 {DataUse} : tensor<64x128xf32>
    %75 = arith.truncf %74 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %67, %75 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    tt.return
  }
}


// -----
// Source: gaoyou/0701/delta_rule_bwd_dhu/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x32xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x32xf32>
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c524288_i64 = arith.constant 524288 : i64
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = arith.index_cast %3 : index to i32
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map1()[%1]
// CHECK-NEXT:     %7 = arith.index_cast %6 : index to i32
// CHECK-NEXT:     %8 = affine.apply #map1()[%1]
// CHECK-NEXT:     %9 = affine.apply #map1()[%1]
// CHECK-NEXT:     %10 = affine.apply #map1()[%1]
// CHECK-NEXT:     %11 = tt.get_program_id x : i32
// CHECK-NEXT:     %12 = tt.get_program_id y : i32
// CHECK-NEXT:     %13 = arith.divsi %12, %c32_i32 : i32
// CHECK-NEXT:     %14 = arith.remsi %12, %c32_i32 : i32
// CHECK-NEXT:     %15 = arith.muli %13, %arg9 : i32
// CHECK-NEXT:     %16 = arith.addi %arg9, %c63_i32 : i32
// CHECK-NEXT:     %17 = arith.divsi %16, %c64_i32 : i32
// CHECK-NEXT:     %18 = arith.muli %13, %17 : i32
// CHECK-NEXT:     %19 = arith.muli %15, %c32_i32 : i32
// CHECK-NEXT:     %20 = arith.addi %19, %14 : i32
// CHECK-NEXT:     %21 = arith.extsi %20 : i32 to i64
// CHECK-NEXT:     %22 = arith.muli %21, %c128_i64 : i64
// CHECK-NEXT:     %23 = tt.addptr %arg0, %22 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %24 = tt.addptr %arg1, %22 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %25 = tt.addptr %arg2, %22 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %26 = tt.addptr %arg4, %22 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %27 = tt.addptr %arg6, %22 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %28 = tt.addptr %arg7, %22 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %29 = arith.muli %18, %c32_i32 : i32
// CHECK-NEXT:     %30 = arith.addi %29, %14 : i32
// CHECK-NEXT:     %31 = arith.extsi %30 : i32 to i64
// CHECK-NEXT:     %32 = arith.muli %31, %c16384_i64 : i64
// CHECK-NEXT:     %33 = arith.subi %17, %c1_i32 : i32
// CHECK-NEXT:     %34 = arith.muli %11, %c32_i32 : i32
// CHECK-NEXT:     %35 = tt.addptr %arg3, %19 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %36 = tt.addptr %35, %14 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %37 = arith.extsi %arg9 : i32 to i64
// CHECK-NEXT:     %38 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %39 = tt.splat %arg9 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %40 = tt.splat %arg8 {DataUse} : f32 -> tensor<128x32xf32>
// CHECK-NEXT:     %41 = scf.for %arg10 = %c-1_i32 to %33 step %c1_i32 iter_args(%arg11 = %cst_1) -> (tensor<128x32xf32>)  : i32 {
// CHECK-NEXT:       %42 = arith.subi %33, %arg10 : i32
// CHECK-NEXT:       %43 = arith.addi %42, %c-1_i32 : i32
// CHECK-NEXT:       %44 = arith.extsi %43 : i32 to i64
// CHECK-NEXT:       %45 = arith.muli %44, %c524288_i64 : i64
// CHECK-NEXT:       %46 = arith.addi %32, %45 : i64
// CHECK-NEXT:       %47 = tt.addptr %arg5, %46 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %48 = tt.make_tensor_ptr %47, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%4, %34] {order = array<i32: 1, 0>, tiled_op} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %49 = arith.truncf %arg11 {DataUse} : tensor<128x32xf32> to tensor<128x32xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %49[%3, 0] [64, 32] [1, 1] {to_be_bubbled_slice} : tensor<128x32xbf16> to tensor<64x32xbf16>
// CHECK-NEXT:       tt.store %48, %extracted_slice {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %50 = arith.muli %42, %c64_i32 : i32
// CHECK-NEXT:       %51 = arith.minsi %50, %arg9 : i32
// CHECK-NEXT:       %52 = arith.subi %51, %c1_i32 : i32
// CHECK-NEXT:       %53 = arith.addi %15, %52 : i32
// CHECK-NEXT:       %54 = arith.muli %53, %c32_i32 : i32
// CHECK-NEXT:       %55 = tt.addptr %arg3, %54 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %56 = tt.addptr %55, %14 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %57 = tt.load %56 : !tt.ptr<f32>
// CHECK-NEXT:       %58 = arith.muli %43, %c64_i32 : i32
// CHECK-NEXT:       %59 = tt.make_tensor_ptr %36, [%37], [%c32_i64], [%58] {order = array<i32: 0>} : <tensor<64xf32>>
// CHECK-NEXT:       %60 = tt.load %59 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
// CHECK-NEXT:       %61 = math.exp %57 : f32
// CHECK-NEXT:       %62 = math.exp %60 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %63 = tt.make_tensor_ptr %27, [%37, %c128_i64], [%c4096_i64, %c1_i64], [%58, %34] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %64 = arith.addi %58, %7 : i32
// CHECK-NEXT:       %65 = tt.make_tensor_ptr %28, [%37, %c128_i64], [%c4096_i64, %c1_i64], [%64, %34] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x32xbf16>>
// CHECK-NEXT:       %66 = tt.make_tensor_ptr %26, [%37, %c128_i64], [%c4096_i64, %c1_i64], [%58, %34] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %67 = tt.load %66 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %68 = tt.make_tensor_ptr %24, [%37, %c128_i64], [%c4096_i64, %c1_i64], [%58, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %69 = tt.load %68 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %extracted_slice_2 = tensor.extract_slice %49[%5, 0] [64, 32] [1, 1] {to_be_bubbled_slice} : tensor<128x32xbf16> to tensor<64x32xbf16>
// CHECK-NEXT:       %70 = tensor.empty() : tensor<128x32xbf16>
// CHECK-NEXT:       %inserted_slice = tensor.insert_slice %extracted_slice_2 into %70[%5, 0] [64, 32] [1, 1] {cv_communication_slice} : tensor<64x32xbf16> into tensor<128x32xbf16>
// CHECK-NEXT:       %71 = tt.dot %69, %inserted_slice, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %72 = tt.splat %58 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %73 = arith.addi %72, %38 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %74 = arith.cmpi slt, %73, %39 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %75 = tt.splat %57 {DataUse} : f32 -> tensor<64xf32>
// CHECK-NEXT:       %76 = arith.subf %75, %60 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %77 = math.exp %76 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %78 = arith.select %74, %77, %cst_0 {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:       %79 = tt.expand_dims %78 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %80 = tt.broadcast %79 {DataUse} : tensor<64x1xf32> -> tensor<64x32xf32>
// CHECK-NEXT:       %81 = arith.mulf %71, %80 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %82 = tt.load %63 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %83 = arith.extf %82 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
// CHECK-NEXT:       %84 = arith.addf %81, %83 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %85 = arith.truncf %84 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %85[%6, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       tt.store %65, %extracted_slice_3 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x32xbf16>>
// CHECK-NEXT:       %86 = tt.make_tensor_ptr %25, [%c128_i64, %37], [%c1_i64, %c4096_i64], [%c0_i32, %58] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:       %87 = tt.make_tensor_ptr %23, [%c128_i64, %37], [%c1_i64, %c4096_i64], [%c0_i32, %58] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:       %88 = tt.load %86 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:       %89 = tt.load %87 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:       %90 = tt.splat %61 {DataUse} : f32 -> tensor<128x32xf32>
// CHECK-NEXT:       %91 = arith.mulf %arg11, %90 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       %92 = tt.expand_dims %62 {DataUse, axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
// CHECK-NEXT:       %93 = arith.extf %89 {DataUse} : tensor<128x64xbf16> to tensor<128x64xf32>
// CHECK-NEXT:       %94 = tt.broadcast %92 {DataUse} : tensor<1x64xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %95 = arith.mulf %93, %94 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:       %96 = arith.extf %67 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
// CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %95[0, %9] [128, 32] [1, 1] {to_be_bubbled_slice} : tensor<128x64xf32> to tensor<128x32xf32>
// CHECK-NEXT:       %97 = tensor.empty() : tensor<128x64xf32>
// CHECK-NEXT:       %inserted_slice_5 = tensor.insert_slice %extracted_slice_4 into %97[0, %9] [128, 32] [1, 1] {cv_communication_slice} : tensor<128x32xf32> into tensor<128x64xf32>
// CHECK-NEXT:       %extracted_slice_6 = tensor.extract_slice %96[%8, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xf32> to tensor<32x32xf32>
// CHECK-NEXT:       %98 = tensor.empty() : tensor<64x32xf32>
// CHECK-NEXT:       %inserted_slice_7 = tensor.insert_slice %extracted_slice_6 into %98[%8, 0] [32, 32] [1, 1] {cv_communication_slice} : tensor<32x32xf32> into tensor<64x32xf32>
// CHECK-NEXT:       %99 = tt.dot %inserted_slice_5, %inserted_slice_7, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf32> * tensor<64x32xf32> -> tensor<128x32xf32>
// CHECK-NEXT:       %100 = arith.mulf %99, %40 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       %extracted_slice_8 = tensor.extract_slice %85[%10, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       %101 = tensor.empty() : tensor<64x32xbf16>
// CHECK-NEXT:       %inserted_slice_9 = tensor.insert_slice %extracted_slice_8 into %101[%10, 0] [32, 32] [1, 1] {cv_communication_slice} : tensor<32x32xbf16> into tensor<64x32xbf16>
// CHECK-NEXT:       %102 = tt.dot %88, %inserted_slice_9, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x32xbf16> -> tensor<128x32xf32>
// CHECK-NEXT:       %103 = arith.subf %100, %102 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       %104 = arith.addf %91, %103 {DataUse} : tensor<128x32xf32>
// CHECK-NEXT:       %extracted_slice_10 = tensor.extract_slice %104[%2, 0] [64, 32] [1, 1] {to_be_bubbled_slice} : tensor<128x32xf32> to tensor<64x32xf32>
// CHECK-NEXT:       %105 = tensor.empty() : tensor<128x32xf32>
// CHECK-NEXT:       %inserted_slice_11 = tensor.insert_slice %extracted_slice_10 into %105[%2, 0] [64, 32] [1, 1] {to_be_eliminated_slice} : tensor<64x32xf32> into tensor<128x32xf32>
// CHECK-NEXT:       scf.yield %inserted_slice_11 : tensor<128x32xf32>
// CHECK-NEXT:     } {Undefined, tiled_op}
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_gated_delta_rule_bwd_kernel_dhu_k128_blockdim128(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c524288_i64 = arith.constant 524288 : i64
    %c-1_i32 = arith.constant -1 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x32xf32>
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x32xf32>
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
    %27 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %28 = tt.splat %arg9 {DataUse} : i32 -> tensor<64xi32>
    %29 = tt.splat %arg8 {DataUse} : f32 -> tensor<128x32xf32>
    %30 = scf.for %arg10 = %c-1_i32 to %22 step %c1_i32 iter_args(%arg11 = %cst) -> (tensor<128x32xf32>)  : i32 {
      %31 = arith.subi %22, %arg10 : i32
      %32 = arith.addi %31, %c-1_i32 : i32
      %33 = arith.extsi %32 : i32 to i64
      %34 = arith.muli %33, %c524288_i64 : i64
      %35 = arith.addi %21, %34 : i64
      %36 = tt.addptr %arg5, %35 : !tt.ptr<bf16>, i64
      %37 = tt.make_tensor_ptr %36, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %23] {order = array<i32: 1, 0>} : <tensor<128x32xbf16>>
      %38 = arith.truncf %arg11 {DataUse} : tensor<128x32xf32> to tensor<128x32xbf16>
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
      %49 = tt.load %48 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
      %50 = math.exp %46 : f32
      %51 = math.exp %49 {DataUse} : tensor<64xf32>
      %52 = tt.make_tensor_ptr %16, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %53 = tt.make_tensor_ptr %17, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %54 = tt.make_tensor_ptr %15, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %23] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %55 = tt.load %54 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
      %56 = tt.make_tensor_ptr %13, [%26, %c128_i64], [%c4096_i64, %c1_i64], [%47, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %57 = tt.load %56 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
      %58 = tt.dot %57, %38, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x32xbf16> -> tensor<64x32xf32>
      %59 = tt.splat %47 {DataUse} : i32 -> tensor<64xi32>
      %60 = arith.addi %59, %27 {DataUse} : tensor<64xi32>
      %61 = arith.cmpi slt, %60, %28 {DataUse} : tensor<64xi32>
      %62 = tt.splat %46 {DataUse} : f32 -> tensor<64xf32>
      %63 = arith.subf %62, %49 {DataUse} : tensor<64xf32>
      %64 = math.exp %63 {DataUse} : tensor<64xf32>
      %65 = arith.select %61, %64, %cst_0 {DataUse} : tensor<64xi1>, tensor<64xf32>
      %66 = tt.expand_dims %65 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %67 = tt.broadcast %66 {DataUse} : tensor<64x1xf32> -> tensor<64x32xf32>
      %68 = arith.mulf %58, %67 {DataUse} : tensor<64x32xf32>
      %69 = tt.load %52 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
      %70 = arith.extf %69 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
      %71 = arith.addf %68, %70 {DataUse} : tensor<64x32xf32>
      %72 = arith.truncf %71 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %53, %72 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %73 = tt.make_tensor_ptr %14, [%c128_i64, %26], [%c1_i64, %c4096_i64], [%c0_i32, %47] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %74 = tt.make_tensor_ptr %12, [%c128_i64, %26], [%c1_i64, %c4096_i64], [%c0_i32, %47] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %75 = tt.load %73 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %76 = tt.load %74 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %77 = tt.splat %50 {DataUse} : f32 -> tensor<128x32xf32>
      %78 = arith.mulf %arg11, %77 {DataUse} : tensor<128x32xf32>
      %79 = tt.expand_dims %51 {DataUse, axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
      %80 = arith.extf %76 {DataUse} : tensor<128x64xbf16> to tensor<128x64xf32>
      %81 = tt.broadcast %79 {DataUse} : tensor<1x64xf32> -> tensor<128x64xf32>
      %82 = arith.mulf %80, %81 {DataUse} : tensor<128x64xf32>
      %83 = arith.extf %55 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
      %84 = tt.dot %82, %83, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf32> * tensor<64x32xf32> -> tensor<128x32xf32>
      %85 = arith.mulf %84, %29 {DataUse} : tensor<128x32xf32>
      %86 = tt.dot %75, %72, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x32xbf16> -> tensor<128x32xf32>
      %87 = arith.subf %85, %86 {DataUse} : tensor<128x32xf32>
      %88 = arith.addf %78, %87 {DataUse} : tensor<128x32xf32>
      scf.yield %88 : tensor<128x32xf32>
    } {Undefined}
    tt.return
  }
}


// -----
// Source: gaoyou/0701/delta_rule_fwd_h/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_gated_delta_rule_fwd_kernel_h_blockdim64(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x32xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %c16384_i64 = arith.constant 16384 : i64
// CHECK-NEXT:     %c16384_i32 = arith.constant 16384 : i32
// CHECK-NEXT:     %c524288_i64 = arith.constant 524288 : i64
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = arith.index_cast %2 : index to i32
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:     %6 = affine.apply #map()[%1]
// CHECK-NEXT:     %7 = affine.apply #map()[%1]
// CHECK-NEXT:     %8 = affine.apply #map()[%1]
// CHECK-NEXT:     %9 = arith.index_cast %8 : index to i32
// CHECK-NEXT:     %10 = affine.apply #map()[%1]
// CHECK-NEXT:     %11 = affine.apply #map()[%1]
// CHECK-NEXT:     %12 = affine.apply #map()[%1]
// CHECK-NEXT:     %13 = arith.index_cast %12 : index to i32
// CHECK-NEXT:     %14 = affine.apply #map()[%1]
// CHECK-NEXT:     %15 = arith.index_cast %14 : index to i32
// CHECK-NEXT:     %16 = tt.get_program_id x : i32
// CHECK-NEXT:     %17 = tt.get_program_id y : i32
// CHECK-NEXT:     %18 = arith.divsi %17, %c32_i32 : i32
// CHECK-NEXT:     %19 = arith.remsi %17, %c32_i32 : i32
// CHECK-NEXT:     %20 = arith.muli %18, %arg7 : i32
// CHECK-NEXT:     %21 = arith.addi %arg7, %c63_i32 : i32
// CHECK-NEXT:     %22 = arith.divsi %21, %c64_i32 : i32
// CHECK-NEXT:     %23 = arith.muli %18, %22 : i32
// CHECK-NEXT:     %24 = arith.muli %23, %c32_i32 : i32
// CHECK-NEXT:     %25 = arith.addi %24, %19 : i32
// CHECK-NEXT:     %26 = arith.extsi %25 : i32 to i64
// CHECK-NEXT:     %27 = arith.muli %26, %c16384_i64 : i64
// CHECK-NEXT:     %28 = arith.muli %20, %c32_i32 : i32
// CHECK-NEXT:     %29 = arith.addi %28, %19 : i32
// CHECK-NEXT:     %30 = arith.extsi %29 : i32 to i64
// CHECK-NEXT:     %31 = arith.muli %30, %c128_i64 : i64
// CHECK-NEXT:     %32 = tt.addptr %arg1, %31 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %33 = tt.addptr %arg0, %31 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %34 = tt.addptr %arg2, %31 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %35 = tt.addptr %arg3, %31 : !tt.ptr<bf16>, i64
// CHECK-NEXT:     %36 = arith.muli %17, %c16384_i32 : i32
// CHECK-NEXT:     %37 = tt.addptr %arg6, %36 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %38 = arith.muli %16, %c32_i32 : i32
// CHECK-NEXT:     %39 = arith.extsi %arg7 : i32 to i64
// CHECK-NEXT:     %40 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %41 = tt.splat %arg7 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %42 = tt.addptr %arg4, %30 : !tt.ptr<f32>, i64
// CHECK-NEXT:     %43:2 = scf.for %arg8 = %c0_i32 to %22 step %c1_i32 iter_args(%arg9 = %cst, %arg10 = %cst) -> (tensor<64x32xf32>, tensor<64x32xf32>)  : i32 {
// CHECK-NEXT:       %48 = arith.extsi %arg8 : i32 to i64
// CHECK-NEXT:       %49 = arith.muli %48, %c524288_i64 : i64
// CHECK-NEXT:       %50 = arith.addi %27, %49 : i64
// CHECK-NEXT:       %51 = tt.addptr %arg5, %50 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %52 = tt.make_tensor_ptr %51, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%3, %38] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x32xbf16>>
// CHECK-NEXT:       %53 = arith.truncf %arg9 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       %extracted_slice_2 = tensor.extract_slice %53[%2, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       tt.store %52, %extracted_slice_2 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x32xbf16>>
// CHECK-NEXT:       %54 = arith.addi %5, %c64_i32 : i32
// CHECK-NEXT:       %55 = tt.make_tensor_ptr %51, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%54, %38] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x32xbf16>>
// CHECK-NEXT:       %56 = arith.truncf %arg10 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %56[%4, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       tt.store %55, %extracted_slice_3 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x32xbf16>>
// CHECK-NEXT:       %57 = arith.muli %arg8, %c64_i32 : i32
// CHECK-NEXT:       %58 = tt.make_tensor_ptr %34, [%39, %c128_i64], [%c4096_i64, %c1_i64], [%57, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %59 = tt.load %58 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %53[%6, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       %60 = tensor.empty() : tensor<64x32xbf16>
// CHECK-NEXT:       %inserted_slice = tensor.insert_slice %extracted_slice_4 into %60[%6, 0] [32, 32] [1, 1] {cv_communication_slice} : tensor<32x32xbf16> into tensor<64x32xbf16>
// CHECK-NEXT:       %61 = tt.dot %59, %inserted_slice, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %62 = tt.make_tensor_ptr %34, [%39, %c128_i64], [%c4096_i64, %c1_i64], [%57, %c64_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %63 = tt.load %62 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %56[%7, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       %64 = tensor.empty() : tensor<64x32xbf16>
// CHECK-NEXT:       %inserted_slice_6 = tensor.insert_slice %extracted_slice_5 into %64[%7, 0] [32, 32] [1, 1] {cv_communication_slice} : tensor<32x32xbf16> into tensor<64x32xbf16>
// CHECK-NEXT:       %65 = tt.dot %63, %inserted_slice_6, %61 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %66 = tt.make_tensor_ptr %32, [%39, %c128_i64], [%c4096_i64, %c1_i64], [%57, %38] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
// CHECK-NEXT:       %67 = tt.load %66 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
// CHECK-NEXT:       %68 = arith.extf %67 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
// CHECK-NEXT:       %69 = arith.subf %68, %65 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %70 = arith.addi %57, %9 : i32
// CHECK-NEXT:       %71 = tt.make_tensor_ptr %35, [%39, %c128_i64], [%c4096_i64, %c1_i64], [%70, %38] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x32xbf16>>
// CHECK-NEXT:       %72 = arith.truncf %69 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       %extracted_slice_7 = tensor.extract_slice %72[%8, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       tt.store %71, %extracted_slice_7 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x32xbf16>>
// CHECK-NEXT:       %73 = arith.addi %arg8, %c1_i32 : i32
// CHECK-NEXT:       %74 = arith.muli %73, %c64_i32 : i32
// CHECK-NEXT:       %75 = arith.minsi %74, %arg7 : i32
// CHECK-NEXT:       %76 = arith.subi %75, %c1_i32 : i32
// CHECK-NEXT:       %77 = tt.splat %57 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %78 = arith.addi %77, %40 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %79 = arith.cmpi slt, %78, %41 {DataUse} : tensor<64xi32>
// CHECK-NEXT:       %80 = arith.muli %76, %c32_i32 : i32
// CHECK-NEXT:       %81 = arith.addi %28, %80 : i32
// CHECK-NEXT:       %82 = arith.addi %81, %19 : i32
// CHECK-NEXT:       %83 = arith.extsi %82 : i32 to i64
// CHECK-NEXT:       %84 = tt.addptr %arg4, %83 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %85 = tt.load %84 : !tt.ptr<f32>
// CHECK-NEXT:       %86 = tt.make_tensor_ptr %42, [%39], [%c32_i64], [%57] {order = array<i32: 0>} : <tensor<64xf32>>
// CHECK-NEXT:       %87 = tt.load %86 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
// CHECK-NEXT:       %88 = tt.splat %85 {DataUse} : f32 -> tensor<64xf32>
// CHECK-NEXT:       %89 = arith.subf %88, %87 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %90 = math.exp2 %89 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %91 = arith.select %79, %90, %cst_0 {DataUse} : tensor<64xi1>, tensor<64xf32>
// CHECK-NEXT:       %92 = tt.expand_dims %91 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %93 = tt.broadcast %92 {DataUse} : tensor<64x1xf32> -> tensor<64x32xf32>
// CHECK-NEXT:       %94 = arith.mulf %69, %93 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %95 = math.exp2 %85 : f32
// CHECK-NEXT:       %96 = tt.splat %95 {DataUse} : f32 -> tensor<64x32xf32>
// CHECK-NEXT:       %97 = arith.mulf %arg9, %96 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %98 = arith.mulf %arg10, %96 {DataUse} : tensor<64x32xf32>
// CHECK-NEXT:       %99 = arith.truncf %94 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
// CHECK-NEXT:       %100 = tt.make_tensor_ptr %33, [%c128_i64, %39], [%c1_i64, %c4096_i64], [%c0_i32, %57] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %101 = tt.load %100 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %extracted_slice_8 = tensor.extract_slice %99[%10, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       %102 = tensor.empty() : tensor<64x32xbf16>
// CHECK-NEXT:       %inserted_slice_9 = tensor.insert_slice %extracted_slice_8 into %102[%10, 0] [32, 32] [1, 1] {cv_communication_slice} : tensor<32x32xbf16> into tensor<64x32xbf16>
// CHECK-NEXT:       %103 = tt.dot %101, %inserted_slice_9, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %104 = arith.addf %103, %97 {DataUse, triton_cv12.add_from_dot} : tensor<64x32xf32>
// CHECK-NEXT:       %105 = tt.make_tensor_ptr %33, [%c128_i64, %39], [%c1_i64, %c4096_i64], [%c64_i32, %57] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %106 = tt.load %105 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %extracted_slice_10 = tensor.extract_slice %99[%11, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xbf16> to tensor<32x32xbf16>
// CHECK-NEXT:       %107 = tensor.empty() : tensor<64x32xbf16>
// CHECK-NEXT:       %inserted_slice_11 = tensor.insert_slice %extracted_slice_10 into %107[%11, 0] [32, 32] [1, 1] {cv_communication_slice} : tensor<32x32xbf16> into tensor<64x32xbf16>
// CHECK-NEXT:       %108 = tt.dot %106, %inserted_slice_11, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
// CHECK-NEXT:       %109 = arith.addf %108, %98 {DataUse, triton_cv12.add_from_dot} : tensor<64x32xf32>
// CHECK-NEXT:       scf.yield %104, %109 : tensor<64x32xf32>, tensor<64x32xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %44 = arith.muli %16, %c32_i32 : i32
// CHECK-NEXT:     %45 = tt.make_tensor_ptr %37, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%13, %44] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x32xf32>>
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %43#0[%12, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xf32> to tensor<32x32xf32>
// CHECK-NEXT:     tt.store %45, %extracted_slice {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x32xf32>>
// CHECK-NEXT:     %46 = arith.addi %15, %c64_i32 : i32
// CHECK-NEXT:     %47 = tt.make_tensor_ptr %37, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%46, %44] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x32xf32>>
// CHECK-NEXT:     %extracted_slice_1 = tensor.extract_slice %43#1[%14, 0] [32, 32] [1, 1] {to_be_bubbled_slice} : tensor<64x32xf32> to tensor<32x32xf32>
// CHECK-NEXT:     tt.store %47, %extracted_slice_1 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x32xf32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_gated_delta_rule_fwd_kernel_h_blockdim64(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c524288_i64 = arith.constant 524288 : i64
    %c16384_i32 = arith.constant 16384 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x32xf32>
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
    %24 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %25 = tt.splat %arg7 {DataUse} : i32 -> tensor<64xi32>
    %26 = tt.addptr %arg4, %14 : !tt.ptr<f32>, i64
    %27:2 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %cst_0, %arg10 = %cst_0) -> (tensor<64x32xf32>, tensor<64x32xf32>)  : i32 {
      %31 = arith.extsi %arg8 : i32 to i64
      %32 = arith.muli %31, %c524288_i64 : i64
      %33 = arith.addi %11, %32 : i64
      %34 = tt.addptr %arg5, %33 : !tt.ptr<bf16>, i64
      %35 = tt.make_tensor_ptr %34, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %36 = arith.truncf %arg9 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %35, %36 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %37 = tt.make_tensor_ptr %34, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c64_i32, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %38 = arith.truncf %arg10 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %37, %38 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %39 = arith.muli %arg8, %c64_i32 : i32
      %40 = tt.make_tensor_ptr %18, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %41 = tt.load %40 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %42 = tt.dot %41, %36, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %43 = tt.make_tensor_ptr %18, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %c64_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %44 = tt.load %43 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %45 = tt.dot %44, %38, %42 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %46 = tt.make_tensor_ptr %16, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %47 = tt.load %46 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x32xbf16>>
      %48 = arith.extf %47 {DataUse} : tensor<64x32xbf16> to tensor<64x32xf32>
      %49 = arith.subf %48, %45 {DataUse} : tensor<64x32xf32>
      %50 = tt.make_tensor_ptr %19, [%23, %c128_i64], [%c4096_i64, %c1_i64], [%39, %22] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>>
      %51 = arith.truncf %49 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
      tt.store %50, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xbf16>>
      %52 = arith.addi %arg8, %c1_i32 : i32
      %53 = arith.muli %52, %c64_i32 : i32
      %54 = arith.minsi %53, %arg7 : i32
      %55 = arith.subi %54, %c1_i32 : i32
      %56 = tt.splat %39 {DataUse} : i32 -> tensor<64xi32>
      %57 = arith.addi %56, %24 {DataUse} : tensor<64xi32>
      %58 = arith.cmpi slt, %57, %25 {DataUse} : tensor<64xi32>
      %59 = arith.muli %55, %c32_i32 : i32
      %60 = arith.addi %12, %59 : i32
      %61 = arith.addi %60, %3 : i32
      %62 = arith.extsi %61 : i32 to i64
      %63 = tt.addptr %arg4, %62 : !tt.ptr<f32>, i64
      %64 = tt.load %63 : !tt.ptr<f32>
      %65 = tt.make_tensor_ptr %26, [%23], [%c32_i64], [%39] {order = array<i32: 0>} : <tensor<64xf32>>
      %66 = tt.load %65 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xf32>>
      %67 = tt.splat %64 {DataUse} : f32 -> tensor<64xf32>
      %68 = arith.subf %67, %66 {DataUse} : tensor<64xf32>
      %69 = math.exp2 %68 {DataUse} : tensor<64xf32>
      %70 = arith.select %58, %69, %cst {DataUse} : tensor<64xi1>, tensor<64xf32>
      %71 = tt.expand_dims %70 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %72 = tt.broadcast %71 {DataUse} : tensor<64x1xf32> -> tensor<64x32xf32>
      %73 = arith.mulf %49, %72 {DataUse} : tensor<64x32xf32>
      %74 = math.exp2 %64 : f32
      %75 = tt.splat %74 {DataUse} : f32 -> tensor<64x32xf32>
      %76 = arith.mulf %arg9, %75 {DataUse} : tensor<64x32xf32>
      %77 = arith.mulf %arg10, %75 {DataUse} : tensor<64x32xf32>
      %78 = arith.truncf %73 {DataUse} : tensor<64x32xf32> to tensor<64x32xbf16>
      %79 = tt.make_tensor_ptr %17, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c0_i32, %39] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
      %80 = tt.load %79 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %81 = tt.dot %80, %78, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %82 = arith.addf %81, %76 {DataUse, triton_cv12.add_from_dot} : tensor<64x32xf32>
      %83 = tt.make_tensor_ptr %17, [%c128_i64, %23], [%c1_i64, %c4096_i64], [%c64_i32, %39] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
      %84 = tt.load %83 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %85 = tt.dot %84, %78, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x32xbf16> -> tensor<64x32xf32>
      %86 = arith.addf %85, %77 {DataUse, triton_cv12.add_from_dot} : tensor<64x32xf32>
      scf.yield %82, %86 : tensor<64x32xf32>, tensor<64x32xf32>
    } {DataUse}
    %28 = arith.muli %0, %c32_i32 : i32
    %29 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %28] {order = array<i32: 1, 0>} : <tensor<64x32xf32>>
    tt.store %29, %27#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf32>>
    %30 = tt.make_tensor_ptr %21, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c64_i32, %28] {order = array<i32: 1, 0>} : <tensor<64x32xf32>>
    tt.store %30, %27#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xf32>>
    tt.return
  }
}


// -----
// Source: gaoyou/0701/dllm/bwd_kv_ul_sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 16)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_bwd_kv_ul(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = tt.get_program_id x : i32
// CHECK-NEXT:     %6 = tt.get_num_programs x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %8 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %9 = tt.expand_dims %7 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %10 = tt.expand_dims %8 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %11 = tt.expand_dims %7 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %12 = tt.expand_dims %8 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %13 = arith.muli %9, %cst {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:     %14 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %15 = tt.addptr %14, %13 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %16 = tt.broadcast %15 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %17 = tt.broadcast %11 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
// CHECK-NEXT:     %18 = tt.addptr %16, %17 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %19 = tt.bitcast %18 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %20 = tt.load %19 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %21 = arith.remsi %5, %6 : i32
// CHECK-NEXT:     %22 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %23 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %24 = tt.splat %23 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %25 = tt.expand_dims %22 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %26 = tt.broadcast %25 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %27 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %28 = tt.splat %27 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %29 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %30 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %31 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %32 = tt.splat %31 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %33 = tt.splat %arg10 {DataUse} : f32 -> tensor<32x32xf32>
// CHECK-NEXT:     %34 = arith.cmpi ne, %20, %cst_4 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %35 = tt.splat %arg10 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     %36:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %37 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %38 = tt.load %37 : !tt.ptr<i32>
// CHECK-NEXT:       %39 = arith.subi %38, %arg15 : i32
// CHECK-NEXT:       %40 = arith.addi %39, %c31_i32 : i32
// CHECK-NEXT:       %41 = arith.divsi %40, %c32_i32 : i32
// CHECK-NEXT:       %42 = arith.addi %arg16, %41 : i32
// CHECK-NEXT:       %43 = arith.remsi %arg16, %6 : i32
// CHECK-NEXT:       %44 = arith.subi %21, %43 : i32
// CHECK-NEXT:       %45 = arith.addi %44, %6 : i32
// CHECK-NEXT:       %46 = arith.remsi %45, %6 : i32
// CHECK-NEXT:       %47 = arith.addi %arg16, %46 : i32
// CHECK-NEXT:       %48 = tt.splat %38 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %49 = tt.splat %38 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %50 = tt.splat %38 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:       %51 = tt.splat %38 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:       scf.for %arg17 = %47 to %42 step %6  : i32 {
// CHECK-NEXT:         %52 = arith.subi %arg17, %arg16 : i32
// CHECK-NEXT:         %53 = arith.muli %52, %c32_i32 : i32
// CHECK-NEXT:         %54 = arith.addi %arg15, %53 : i32
// CHECK-NEXT:         %55 = tt.splat %54 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %56 = arith.addi %55, %7 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %57 = tt.expand_dims %56 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %58 = arith.muli %57, %cst_0 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %59 = tt.addptr %24, %58 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %60 = tt.broadcast %59 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %61 = tt.addptr %60, %26 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %62 = tt.addptr %28, %58 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %63 = tt.broadcast %62 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %64 = tt.addptr %63, %26 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %65 = tt.addptr %30, %58 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %66 = tt.broadcast %65 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %67 = tt.addptr %66, %26 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %68 = tt.addptr %32, %58 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %69 = tt.broadcast %68 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %70 = tt.addptr %69, %26 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %71 = arith.cmpi slt, %57, %48 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %72 = tt.broadcast %71 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %73 = tt.load %61, %72, %cst_2 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %74 = tt.load %64, %72, %cst_2 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %75 = tt.trans %73 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %76 = tt.trans %74 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %77 = tt.splat %54 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:         %78 = arith.addi %77, %10 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %79 = arith.cmpi slt, %78, %49 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %80 = tt.splat %54 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:         %81 = arith.addi %80, %12 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %82 = arith.cmpi slt, %81, %50 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %83 = tt.broadcast %79 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %84 = tt.broadcast %82 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %85 = arith.andi %83, %84 {DataUse} : tensor<32x32xi1>
// CHECK-NEXT:         %86 = arith.muli %57, %cst_9 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %87 = arith.cmpi slt, %56, %51 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %88 = arith.uitofp %85 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
// CHECK-NEXT:         %89 = arith.subf %88, %cst_6 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %90 = arith.mulf %89, %cst_5 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %91:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_1, %arg20 = %cst_1) -> (tensor<32x128xf32>, tensor<32x128xf32>)  : i32 {
// CHECK-NEXT:           %94 = arith.muli %arg18, %c128_i32 : i32
// CHECK-NEXT:           %95 = tt.addptr %arg0, %94 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %96 = tt.splat %95 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %97 = tt.addptr %96, %86 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %98 = tt.broadcast %97 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %99 = tt.addptr %98, %26 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %100 = tt.addptr %arg3, %94 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %101 = tt.splat %100 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %102 = tt.addptr %101, %86 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %103 = tt.broadcast %102 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %104 = tt.addptr %103, %26 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %105 = arith.muli %arg18, %arg13 : i32
// CHECK-NEXT:           %106 = tt.addptr %arg4, %105 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %107 = tt.splat %106 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %108 = tt.addptr %107, %56 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:           %109 = tt.addptr %arg5, %105 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %110 = tt.splat %109 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %111 = tt.addptr %110, %56 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:           %112 = tt.load %99, %72, %cst_2 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %113 = tt.load %111, %87, %cst_8 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %114 = tt.dot %112, %75, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %115 = arith.mulf %114, %33 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %116 = arith.addf %115, %90 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %117 = arith.select %34, %116, %cst_3 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:           %118 = tt.load %104, %72, %cst_2 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %119 = tt.expand_dims %113 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %120 = tt.broadcast %119 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:           %121 = arith.subf %117, %120 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %122 = math.exp %121 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %123 = arith.truncf %122 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %extracted_slice_12 = tensor.extract_slice %123[%2, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:           %124 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:           %inserted_slice = tensor.insert_slice %extracted_slice_12 into %124[%2, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:           %125 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
// CHECK-NEXT:           %126 = tt.dot %125, %118, %arg20 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %127 = tt.load %108, %87, %cst_8 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:           %128 = tt.dot %118, %76, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %129 = tt.expand_dims %127 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %130 = tt.broadcast %129 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:           %131 = arith.subf %128, %130 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %132 = arith.mulf %122, %131 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %133 = arith.truncf %132 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %extracted_slice_13 = tensor.extract_slice %133[%3, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:           %134 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:           %inserted_slice_14 = tensor.insert_slice %extracted_slice_13 into %134[%3, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:           %135 = tt.trans %inserted_slice_14 {DataUse, order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
// CHECK-NEXT:           %136 = tt.dot %135, %112, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %137 = arith.mulf %136, %35 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %138 = arith.addf %arg19, %137 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %138, %126 : tensor<32x128xf32>, tensor<32x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %92 = arith.truncf %91#0 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %92[%4, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xbf16> to tensor<16x128xbf16>
// CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %67[%4, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128x!tt.ptr<bf16>> to tensor<16x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %extracted_slice_11 = tensor.extract_slice %72[%4, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xi1> to tensor<16x128xi1>
// CHECK-NEXT:         tt.store %extracted_slice_10, %extracted_slice, %extracted_slice_11 {tiled_op} : tensor<16x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %93 = arith.truncf %91#1 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:         tt.store %70, %93, %72 : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %38, %42 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_bwd_kv_ul(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
    %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
    %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
    %cst_4 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
    %cst_5 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_6 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
    %cst_8 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %cst_9 = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %8 = arith.muli %4, %cst_9 {MetaUse} : tensor<32x1xi32>
    %9 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %10 = tt.addptr %9, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %11 = tt.broadcast %10 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %12 = tt.broadcast %6 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
    %13 = tt.addptr %11, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %14 = tt.bitcast %13 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %16 = arith.remsi %0, %1 : i32
    %17 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %18 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
    %19 = tt.splat %18 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %20 = tt.expand_dims %17 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %21 = tt.broadcast %20 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
    %22 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
    %23 = tt.splat %22 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %24 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
    %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %26 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
    %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %28 = tt.splat %arg10 {DataUse} : f32 -> tensor<32x32xf32>
    %29 = arith.cmpi ne, %15, %cst_4 {DataUse} : tensor<32x32xi8>
    %30 = tt.splat %arg10 {DataUse} : f32 -> tensor<32x128xf32>
    %31:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
      %32 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = arith.subi %33, %arg15 : i32
      %35 = arith.addi %34, %c31_i32 : i32
      %36 = arith.divsi %35, %c32_i32 : i32
      %37 = arith.addi %arg16, %36 : i32
      %38 = arith.remsi %arg16, %1 : i32
      %39 = arith.subi %16, %38 : i32
      %40 = arith.addi %39, %1 : i32
      %41 = arith.remsi %40, %1 : i32
      %42 = arith.addi %arg16, %41 : i32
      %43 = tt.splat %33 {MetaUse} : i32 -> tensor<32x1xi32>
      %44 = tt.splat %33 {DataUse} : i32 -> tensor<32x1xi32>
      %45 = tt.splat %33 {DataUse} : i32 -> tensor<1x32xi32>
      %46 = tt.splat %33 {MetaUse} : i32 -> tensor<32xi32>
      scf.for %arg17 = %42 to %37 step %1  : i32 {
        %47 = arith.subi %arg17, %arg16 : i32
        %48 = arith.muli %47, %c32_i32 : i32
        %49 = arith.addi %arg15, %48 : i32
        %50 = tt.splat %49 {MetaUse} : i32 -> tensor<32xi32>
        %51 = arith.addi %50, %2 {MetaUse} : tensor<32xi32>
        %52 = tt.expand_dims %51 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %53 = arith.muli %52, %cst_8 {MetaUse} : tensor<32x1xi32>
        %54 = tt.addptr %19, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %55 = tt.broadcast %54 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %56 = tt.addptr %55, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %57 = tt.addptr %23, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %58 = tt.broadcast %57 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %59 = tt.addptr %58, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %60 = tt.addptr %25, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %61 = tt.broadcast %60 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %62 = tt.addptr %61, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %63 = tt.addptr %27, %53 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %64 = tt.broadcast %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %66 = arith.cmpi slt, %52, %43 {MetaUse} : tensor<32x1xi32>
        %67 = tt.broadcast %66 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
        %68 = tt.load %56, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %69 = tt.load %59, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %70 = tt.trans %68 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %71 = tt.trans %69 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %72 = tt.splat %49 {DataUse} : i32 -> tensor<32x1xi32>
        %73 = arith.addi %72, %5 {DataUse} : tensor<32x1xi32>
        %74 = arith.cmpi slt, %73, %44 {DataUse} : tensor<32x1xi32>
        %75 = tt.splat %49 {DataUse} : i32 -> tensor<1x32xi32>
        %76 = arith.addi %75, %7 {DataUse} : tensor<1x32xi32>
        %77 = arith.cmpi slt, %76, %45 {DataUse} : tensor<1x32xi32>
        %78 = tt.broadcast %74 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
        %79 = tt.broadcast %77 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
        %80 = arith.andi %78, %79 {DataUse} : tensor<32x32xi1>
        %81 = arith.muli %52, %cst {MetaUse} : tensor<32x1xi32>
        %82 = arith.cmpi slt, %51, %46 {MetaUse} : tensor<32xi32>
        %83 = arith.uitofp %80 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
        %84 = arith.subf %83, %cst_2 {DataUse} : tensor<32x32xf32>
        %85 = arith.mulf %84, %cst_3 {DataUse} : tensor<32x32xf32>
        %86:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_7, %arg20 = %cst_7) -> (tensor<32x128xf32>, tensor<32x128xf32>)  : i32 {
          %89 = arith.muli %arg18, %c128_i32 : i32
          %90 = tt.addptr %arg0, %89 : !tt.ptr<bf16>, i32
          %91 = tt.splat %90 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
          %92 = tt.addptr %91, %81 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %93 = tt.broadcast %92 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %94 = tt.addptr %93, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %95 = tt.addptr %arg3, %89 : !tt.ptr<bf16>, i32
          %96 = tt.splat %95 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
          %97 = tt.addptr %96, %81 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %98 = tt.broadcast %97 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %99 = tt.addptr %98, %21 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %100 = arith.muli %arg18, %arg13 : i32
          %101 = tt.addptr %arg4, %100 : !tt.ptr<f32>, i32
          %102 = tt.splat %101 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
          %103 = tt.addptr %102, %51 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
          %104 = tt.addptr %arg5, %100 : !tt.ptr<f32>, i32
          %105 = tt.splat %104 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
          %106 = tt.addptr %105, %51 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
          %107 = tt.load %94, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
          %108 = tt.load %106, %82, %cst_0 {DataUse} : tensor<32x!tt.ptr<f32>>
          %109 = tt.dot %107, %70, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %110 = arith.mulf %109, %28 {DataUse} : tensor<32x32xf32>
          %111 = arith.addf %110, %85 {DataUse} : tensor<32x32xf32>
          %112 = arith.select %29, %111, %cst_5 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
          %113 = tt.load %99, %67, %cst_6 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
          %114 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %115 = tt.broadcast %114 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
          %116 = arith.subf %112, %115 {DataUse} : tensor<32x32xf32>
          %117 = math.exp %116 {DataUse} : tensor<32x32xf32>
          %118 = arith.truncf %117 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
          %119 = tt.trans %118 {DataUse, order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
          %120 = tt.dot %119, %113, %arg20 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %121 = tt.load %103, %82, %cst_0 {DataUse} : tensor<32x!tt.ptr<f32>>
          %122 = tt.dot %113, %71, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %123 = tt.expand_dims %121 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %124 = tt.broadcast %123 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
          %125 = arith.subf %122, %124 {DataUse} : tensor<32x32xf32>
          %126 = arith.mulf %117, %125 {DataUse} : tensor<32x32xf32>
          %127 = arith.truncf %126 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
          %128 = tt.trans %127 {DataUse, order = array<i32: 1, 0>} : tensor<32x32xbf16> -> tensor<32x32xbf16>
          %129 = tt.dot %128, %107, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %130 = arith.mulf %129, %30 {DataUse} : tensor<32x128xf32>
          %131 = arith.addf %arg19, %130 {DataUse} : tensor<32x128xf32>
          scf.yield %131, %120 : tensor<32x128xf32>, tensor<32x128xf32>
        } {DataUse}
        %87 = arith.truncf %86#0 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
        tt.store %62, %87, %67 : tensor<32x128x!tt.ptr<bf16>>
        %88 = arith.truncf %86#1 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
        tt.store %65, %88, %67 : tensor<32x128x!tt.ptr<bf16>>
      }
      scf.yield %33, %37 : i32, i32
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/dllm/bwd_kv_ur_sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_bwd_kv_ur(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<64> : tensor<64x1xi32>
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c127_i32 = arith.constant 127 : i32
// CHECK-NEXT:     %c63_i32 = arith.constant 63 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0> : tensor<64x64xi8>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<1.000000e+06> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<1.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<640> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<640> : tensor<128x1xi32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map1()[%1]
// CHECK-NEXT:     %7 = affine.apply #map1()[%1]
// CHECK-NEXT:     %8 = affine.apply #map()[%1]
// CHECK-NEXT:     %9 = affine.apply #map()[%1]
// CHECK-NEXT:     %10 = tt.get_program_id x : i32
// CHECK-NEXT:     %11 = tt.get_num_programs x : i32
// CHECK-NEXT:     %12 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %13 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %14 = tt.expand_dims %12 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %15 = tt.expand_dims %13 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %16 = tt.expand_dims %12 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %17 = tt.expand_dims %13 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %18 = arith.muli %14, %cst {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %19 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
// CHECK-NEXT:     %20 = tt.addptr %19, %18 {MetaUse} : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
// CHECK-NEXT:     %21 = tt.broadcast %20 {MetaUse} : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
// CHECK-NEXT:     %22 = tt.broadcast %16 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %23 = tt.addptr %21, %22 {MetaUse} : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
// CHECK-NEXT:     %24 = tt.bitcast %23 {MetaUse} : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:     %25 = tt.load %24 {DataUse, was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:     %26 = arith.remsi %10, %11 : i32
// CHECK-NEXT:     %27 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %28 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %29 = tt.splat %28 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %30 = tt.expand_dims %27 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %31 = tt.broadcast %30 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %32 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %33 = tt.splat %32 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %34 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %35 = tt.splat %34 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %36 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %37 = tt.splat %36 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:     %38 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     %39 = arith.cmpi ne, %25, %cst_4 {DataUse} : tensor<64x64xi8>
// CHECK-NEXT:     %40 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %41 = tt.broadcast %30 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %42 = tt.splat %arg10 {DataUse} : f32 -> tensor<128x64xf32>
// CHECK-NEXT:     %43:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %44 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %45 = tt.load %44 : !tt.ptr<i32>
// CHECK-NEXT:       %46 = arith.subi %45, %arg15 : i32
// CHECK-NEXT:       %47 = arith.addi %46, %c63_i32 : i32
// CHECK-NEXT:       %48 = arith.divsi %47, %c64_i32 : i32
// CHECK-NEXT:       %49 = arith.addi %arg16, %48 : i32
// CHECK-NEXT:       %50 = arith.remsi %arg16, %11 : i32
// CHECK-NEXT:       %51 = arith.subi %26, %50 : i32
// CHECK-NEXT:       %52 = arith.addi %51, %11 : i32
// CHECK-NEXT:       %53 = arith.remsi %52, %11 : i32
// CHECK-NEXT:       %54 = arith.addi %arg16, %53 : i32
// CHECK-NEXT:       %55 = arith.addi %arg12, %arg15 : i32
// CHECK-NEXT:       %56 = tt.splat %45 {MetaUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:       %57 = tt.splat %45 {DataUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:       %58 = tt.splat %45 {DataUse} : i32 -> tensor<1x64xi32>
// CHECK-NEXT:       %59 = tt.splat %45 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %60 = arith.addi %46, %c127_i32 : i32
// CHECK-NEXT:       %61 = arith.divsi %60, %c128_i32 : i32
// CHECK-NEXT:       %62 = tt.splat %45 {MetaUse} : i32 -> tensor<128x1xi32>
// CHECK-NEXT:       %63 = tt.splat %45 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       scf.for %arg17 = %54 to %49 step %11  : i32 {
// CHECK-NEXT:         %64 = arith.subi %arg17, %arg16 : i32
// CHECK-NEXT:         %65 = arith.muli %64, %c64_i32 : i32
// CHECK-NEXT:         %66 = arith.addi %55, %65 : i32
// CHECK-NEXT:         %67 = tt.splat %66 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %68 = arith.addi %67, %12 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %69 = tt.expand_dims %68 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %70 = arith.muli %69, %cst_0 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %71 = tt.addptr %29, %70 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %72 = tt.broadcast %71 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %73 = tt.addptr %72, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %74 = tt.addptr %33, %70 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %75 = tt.broadcast %74 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %76 = tt.addptr %75, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %77 = tt.addptr %35, %70 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %78 = tt.broadcast %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %79 = tt.addptr %78, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %80 = tt.addptr %37, %70 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %81 = tt.broadcast %80 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %82 = tt.addptr %81, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %83 = arith.addi %arg15, %65 : i32
// CHECK-NEXT:         %84 = tt.splat %83 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %85 = arith.addi %84, %12 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %86 = tt.expand_dims %85 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %87 = arith.cmpi slt, %86, %56 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %88 = tt.broadcast %87 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:         %89 = tt.load %73, %88, %cst_2 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %90 = tt.load %76, %88, %cst_2 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %91 = tt.trans %89 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %92 = tt.trans %90 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %93 = tt.splat %83 {DataUse} : i32 -> tensor<64x1xi32>
// CHECK-NEXT:         %94 = arith.addi %93, %15 {DataUse} : tensor<64x1xi32>
// CHECK-NEXT:         %95 = arith.cmpi slt, %94, %57 {DataUse} : tensor<64x1xi32>
// CHECK-NEXT:         %96 = tt.splat %83 {DataUse} : i32 -> tensor<1x64xi32>
// CHECK-NEXT:         %97 = arith.addi %96, %17 {DataUse} : tensor<1x64xi32>
// CHECK-NEXT:         %98 = arith.cmpi slt, %97, %58 {DataUse} : tensor<1x64xi32>
// CHECK-NEXT:         %99 = tt.broadcast %95 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:         %100 = tt.broadcast %98 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:         %101 = arith.andi %99, %100 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:         %102 = arith.muli %86, %cst_9 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %103 = arith.cmpi slt, %85, %59 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %104 = arith.uitofp %101 {DataUse} : tensor<64x64xi1> to tensor<64x64xf32>
// CHECK-NEXT:         %105 = arith.subf %104, %cst_6 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %106 = arith.mulf %105, %cst_5 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %107 = arith.addi %64, %c1_i32 : i32
// CHECK-NEXT:         %108 = arith.divsi %65, %c128_i32 : i32
// CHECK-NEXT:         %109 = arith.addi %108, %c1_i32 : i32
// CHECK-NEXT:         %110 = arith.muli %109, %c128_i32 : i32
// CHECK-NEXT:         %111 = arith.divsi %110, %c64_i32 : i32
// CHECK-NEXT:         %112:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_1, %arg20 = %cst_1) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:           %115 = arith.muli %arg18, %c128_i32 : i32
// CHECK-NEXT:           %116 = tt.addptr %arg0, %115 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %117 = tt.splat %116 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %118 = tt.addptr %117, %102 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:           %119 = tt.broadcast %118 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %120 = tt.addptr %119, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:           %121 = tt.addptr %arg3, %115 : !tt.ptr<bf16>, i32
// CHECK-NEXT:           %122 = tt.splat %121 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %123 = tt.addptr %122, %102 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:           %124 = tt.broadcast %123 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %125 = tt.addptr %124, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:           %126 = arith.muli %arg18, %arg13 : i32
// CHECK-NEXT:           %127 = tt.addptr %arg4, %126 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %128 = tt.splat %127 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %129 = tt.addptr %128, %85 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:           %130 = tt.addptr %arg5, %126 : !tt.ptr<f32>, i32
// CHECK-NEXT:           %131 = tt.splat %130 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %132 = tt.addptr %131, %85 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:           %133 = tt.load %120, %88, %cst_2 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %134 = tt.load %132, %103, %cst_8 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %135 = tt.dot %133, %91, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:           %136 = arith.mulf %135, %38 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %137 = arith.addf %136, %106 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %138 = arith.select %39, %137, %cst_3 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:           %139 = tt.load %125, %88, %cst_2 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %140 = tt.expand_dims %134 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:           %141 = tt.broadcast %140 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:           %142 = arith.subf %138, %141 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %143 = math.exp %142 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %144 = arith.truncf %143 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:           %extracted_slice_19 = tensor.extract_slice %144[%2, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:           %145 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:           %inserted_slice = tensor.insert_slice %extracted_slice_19 into %145[%2, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:           %146 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:           %147 = tt.dot %146, %139, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %148 = arith.addf %147, %arg20 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:           %149 = tt.load %129, %103, %cst_8 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:           %150 = tt.dot %139, %92, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:           %151 = tt.expand_dims %149 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:           %152 = tt.broadcast %151 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:           %153 = arith.subf %150, %152 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %154 = arith.mulf %143, %153 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:           %155 = arith.truncf %154 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:           %extracted_slice_20 = tensor.extract_slice %155[%3, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:           %156 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:           %inserted_slice_21 = tensor.insert_slice %extracted_slice_20 into %156[%3, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:           %157 = tt.trans %inserted_slice_21 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:           %158 = tt.dot %157, %133, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %159 = arith.mulf %158, %40 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %160 = arith.addf %arg19, %159 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %161:2 = scf.for %arg21 = %107 to %111 step %c1_i32 iter_args(%arg22 = %160, %arg23 = %cst_1) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:             %173 = arith.muli %arg21, %c64_i32 : i32
// CHECK-NEXT:             %174 = arith.addi %arg15, %173 : i32
// CHECK-NEXT:             %175 = tt.splat %174 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:             %176 = arith.addi %175, %12 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:             %177 = tt.expand_dims %176 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:             %178 = arith.muli %177, %cst_9 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:             %179 = tt.addptr %117, %178 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:             %180 = tt.broadcast %179 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %181 = tt.addptr %180, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:             %182 = tt.addptr %122, %178 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:             %183 = tt.broadcast %182 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %184 = tt.addptr %183, %31 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:             %185 = tt.addptr %128, %176 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:             %186 = tt.addptr %131, %176 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:             %187 = arith.cmpi slt, %177, %56 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:             %188 = arith.cmpi slt, %176, %59 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:             %189 = tt.broadcast %187 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:             %190 = tt.load %181, %189, %cst_2 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %191 = tt.load %186, %188, %cst_8 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:             %192 = tt.dot %190, %91, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:             %193 = arith.mulf %192, %38 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %194 = tt.load %184, %189, %cst_2 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %195 = tt.expand_dims %191 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:             %196 = tt.broadcast %195 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:             %197 = arith.subf %193, %196 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %198 = math.exp %197 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %199 = arith.truncf %198 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:             %extracted_slice_22 = tensor.extract_slice %199[%4, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:             %200 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:             %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %200[%4, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:             %201 = tt.trans %inserted_slice_23 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:             %202 = tt.dot %201, %194, %arg23 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %203 = tt.load %185, %188, %cst_8 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:             %204 = tt.dot %194, %92, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:             %205 = tt.expand_dims %203 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:             %206 = tt.broadcast %205 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:             %207 = arith.subf %204, %206 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %208 = arith.mulf %198, %207 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:             %209 = arith.truncf %208 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:             %extracted_slice_24 = tensor.extract_slice %209[%5, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:             %210 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:             %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %210[%5, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:             %211 = tt.trans %inserted_slice_25 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:             %212 = tt.dot %211, %190, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %213 = arith.mulf %212, %40 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             %214 = arith.addf %arg22, %213 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             scf.yield %214, %202 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse, hivm.matmul_limited_in_cube}
// CHECK-NEXT:           %162 = arith.cmpi sgt, %111, %107 : i32
// CHECK-NEXT:           %163 = scf.if %162 -> (tensor<64x128xf32>) {
// CHECK-NEXT:             scf.yield %161#1 : tensor<64x128xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %cst_1 : tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse}
// CHECK-NEXT:           %164 = arith.addf %163, %148 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:           %165 = tt.splat %116 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %166 = tt.splat %121 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:           %167 = tt.splat %127 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %168 = tt.splat %130 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %169:2 = scf.for %arg21 = %109 to %61 step %c1_i32 iter_args(%arg22 = %161#0, %arg23 = %cst_1) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:             %173 = arith.muli %arg21, %c128_i32 : i32
// CHECK-NEXT:             %174 = arith.addi %arg15, %173 : i32
// CHECK-NEXT:             %175 = tt.splat %174 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:             %176 = arith.addi %175, %27 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:             %177 = tt.expand_dims %176 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:             %178 = arith.muli %177, %cst_13 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:             %179 = tt.addptr %165, %178 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:             %180 = tt.broadcast %179 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %181 = tt.addptr %180, %41 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:             %182 = tt.addptr %166, %178 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:             %183 = tt.broadcast %182 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %184 = tt.addptr %183, %41 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:             %185 = tt.addptr %167, %176 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:             %186 = tt.addptr %168, %176 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:             %187 = arith.cmpi slt, %177, %62 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:             %188 = arith.cmpi slt, %176, %63 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:             %189 = tt.broadcast %187 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:             %190 = tt.load %181, %189, %cst_10 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %191 = tt.load %186, %188, %cst_12 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:             %192 = tt.dot %190, %91, %cst_11 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:             %193 = arith.mulf %192, %42 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %194 = tt.load %184, %189, %cst_10 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:             %195 = tt.expand_dims %191 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:             %196 = tt.broadcast %195 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:             %197 = arith.subf %193, %196 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %198 = math.exp %197 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %199 = arith.truncf %198 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:             %extracted_slice_22 = tensor.extract_slice %199[%6, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:             %200 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:             %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %200[%6, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:             %201 = tt.trans %inserted_slice_23 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:             %202 = tt.dot %201, %194, %arg23 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %203 = tt.load %185, %188, %cst_12 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:             %204 = tt.dot %194, %92, %cst_11 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:             %205 = tt.expand_dims %203 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:             %206 = tt.broadcast %205 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:             %207 = arith.subf %204, %206 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %208 = arith.mulf %198, %207 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:             %209 = arith.truncf %208 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:             %extracted_slice_24 = tensor.extract_slice %209[%7, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:             %210 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:             %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %210[%7, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:             %211 = tt.trans %inserted_slice_25 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:             %212 = tt.dot %211, %190, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:             %213 = arith.mulf %212, %40 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             %214 = arith.addf %arg22, %213 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:             scf.yield %214, %202 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse, hivm.matmul_limited_in_cube}
// CHECK-NEXT:           %170 = arith.cmpi sgt, %61, %109 : i32
// CHECK-NEXT:           %171 = scf.if %170 -> (tensor<64x128xf32>) {
// CHECK-NEXT:             scf.yield %169#1 : tensor<64x128xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %cst_1 : tensor<64x128xf32>
// CHECK-NEXT:           } {DataUse}
// CHECK-NEXT:           %172 = arith.addf %171, %164 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:           scf.yield %169#0, %172 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %113 = arith.truncf %112#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %113[%8, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %79[%8, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128x!tt.ptr<bf16>> to tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %88[%8, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xi1> to tensor<32x128xi1>
// CHECK-NEXT:         tt.store %extracted_slice_14, %extracted_slice, %extracted_slice_15 {tiled_op} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %114 = arith.truncf %112#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %114[%9, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %82[%9, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128x!tt.ptr<bf16>> to tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %extracted_slice_18 = tensor.extract_slice %88[%9, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xi1> to tensor<32x128xi1>
// CHECK-NEXT:         tt.store %extracted_slice_17, %extracted_slice_16, %extracted_slice_18 {tiled_op} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %45, %49 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_bwd_kv_ur(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32>, %arg9: i32, %arg10: f32, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
    %cst = arith.constant {MetaUse} dense<640> : tensor<128x1xi32>
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
    %cst_2 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_3 = arith.constant {MetaUse} dense<640> : tensor<64x1xi32>
    %cst_4 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xf32>
    %cst_5 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
    %cst_6 = arith.constant {DataUse} dense<1.000000e+00> : tensor<64x64xf32>
    %cst_7 = arith.constant {DataUse} dense<1.000000e+06> : tensor<64x64xf32>
    %cst_8 = arith.constant {DataUse} dense<0> : tensor<64x64xi8>
    %cst_9 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<64x64xf32>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
    %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_12 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %cst_13 = arith.constant {MetaUse} dense<64> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %8 = arith.muli %4, %cst_13 {MetaUse} : tensor<64x1xi32>
    %9 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
    %10 = tt.addptr %9, %8 {MetaUse} : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
    %11 = tt.broadcast %10 {MetaUse} : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
    %12 = tt.broadcast %6 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %13 = tt.addptr %11, %12 {MetaUse} : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
    %14 = tt.bitcast %13 {MetaUse} : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
    %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
    %16 = arith.remsi %0, %1 : i32
    %17 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %18 = tt.addptr %arg1, %c0_i32 : !tt.ptr<bf16>, i32
    %19 = tt.splat %18 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %20 = tt.expand_dims %17 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %21 = tt.broadcast %20 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
    %22 = tt.addptr %arg2, %c0_i32 : !tt.ptr<bf16>, i32
    %23 = tt.splat %22 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %24 = tt.addptr %arg6, %c0_i32 : !tt.ptr<bf16>, i32
    %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %26 = tt.addptr %arg7, %c0_i32 : !tt.ptr<bf16>, i32
    %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %28 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x64xf32>
    %29 = arith.cmpi ne, %15, %cst_8 {DataUse} : tensor<64x64xi8>
    %30 = tt.splat %arg10 {DataUse} : f32 -> tensor<64x128xf32>
    %31 = tt.broadcast %20 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %32 = tt.splat %arg10 {DataUse} : f32 -> tensor<128x64xf32>
    %33:2 = scf.for %arg14 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
      %34 = tt.addptr %arg8, %arg14 : !tt.ptr<i32>, i32
      %35 = tt.load %34 : !tt.ptr<i32>
      %36 = arith.subi %35, %arg15 : i32
      %37 = arith.addi %36, %c63_i32 : i32
      %38 = arith.divsi %37, %c64_i32 : i32
      %39 = arith.addi %arg16, %38 : i32
      %40 = arith.remsi %arg16, %1 : i32
      %41 = arith.subi %16, %40 : i32
      %42 = arith.addi %41, %1 : i32
      %43 = arith.remsi %42, %1 : i32
      %44 = arith.addi %arg16, %43 : i32
      %45 = arith.addi %arg12, %arg15 : i32
      %46 = tt.splat %35 {MetaUse} : i32 -> tensor<64x1xi32>
      %47 = tt.splat %35 {DataUse} : i32 -> tensor<64x1xi32>
      %48 = tt.splat %35 {DataUse} : i32 -> tensor<1x64xi32>
      %49 = tt.splat %35 {MetaUse} : i32 -> tensor<64xi32>
      %50 = arith.addi %36, %c127_i32 : i32
      %51 = arith.divsi %50, %c128_i32 : i32
      %52 = tt.splat %35 {MetaUse} : i32 -> tensor<128x1xi32>
      %53 = tt.splat %35 {MetaUse} : i32 -> tensor<128xi32>
      scf.for %arg17 = %44 to %39 step %1  : i32 {
        %54 = arith.subi %arg17, %arg16 : i32
        %55 = arith.muli %54, %c64_i32 : i32
        %56 = arith.addi %45, %55 : i32
        %57 = tt.splat %56 {MetaUse} : i32 -> tensor<64xi32>
        %58 = arith.addi %57, %2 {MetaUse} : tensor<64xi32>
        %59 = tt.expand_dims %58 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %60 = arith.muli %59, %cst_12 {MetaUse} : tensor<64x1xi32>
        %61 = tt.addptr %19, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %62 = tt.broadcast %61 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %63 = tt.addptr %62, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %64 = tt.addptr %23, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %65 = tt.broadcast %64 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %66 = tt.addptr %65, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %67 = tt.addptr %25, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %68 = tt.broadcast %67 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %69 = tt.addptr %68, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %70 = tt.addptr %27, %60 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %71 = tt.broadcast %70 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %72 = tt.addptr %71, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %73 = arith.addi %arg15, %55 : i32
        %74 = tt.splat %73 {MetaUse} : i32 -> tensor<64xi32>
        %75 = arith.addi %74, %2 {MetaUse} : tensor<64xi32>
        %76 = tt.expand_dims %75 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %77 = arith.cmpi slt, %76, %46 {MetaUse} : tensor<64x1xi32>
        %78 = tt.broadcast %77 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
        %79 = tt.load %63, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
        %80 = tt.load %66, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
        %81 = tt.trans %79 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %82 = tt.trans %80 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %83 = tt.splat %73 {DataUse} : i32 -> tensor<64x1xi32>
        %84 = arith.addi %83, %5 {DataUse} : tensor<64x1xi32>
        %85 = arith.cmpi slt, %84, %47 {DataUse} : tensor<64x1xi32>
        %86 = tt.splat %73 {DataUse} : i32 -> tensor<1x64xi32>
        %87 = arith.addi %86, %7 {DataUse} : tensor<1x64xi32>
        %88 = arith.cmpi slt, %87, %48 {DataUse} : tensor<1x64xi32>
        %89 = tt.broadcast %85 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
        %90 = tt.broadcast %88 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
        %91 = arith.andi %89, %90 {DataUse} : tensor<64x64xi1>
        %92 = arith.muli %76, %cst_3 {MetaUse} : tensor<64x1xi32>
        %93 = arith.cmpi slt, %75, %49 {MetaUse} : tensor<64xi32>
        %94 = arith.uitofp %91 {DataUse} : tensor<64x64xi1> to tensor<64x64xf32>
        %95 = arith.subf %94, %cst_6 {DataUse} : tensor<64x64xf32>
        %96 = arith.mulf %95, %cst_7 {DataUse} : tensor<64x64xf32>
        %97 = arith.addi %54, %c1_i32 : i32
        %98 = arith.divsi %55, %c128_i32 : i32
        %99 = arith.addi %98, %c1_i32 : i32
        %100 = arith.muli %99, %c128_i32 : i32
        %101 = arith.divsi %100, %c64_i32 : i32
        %102:2 = scf.for %arg18 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg19 = %cst_11, %arg20 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
          %105 = arith.muli %arg18, %c128_i32 : i32
          %106 = tt.addptr %arg0, %105 : !tt.ptr<bf16>, i32
          %107 = tt.splat %106 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
          %108 = tt.addptr %107, %92 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
          %109 = tt.broadcast %108 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
          %110 = tt.addptr %109, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
          %111 = tt.addptr %arg3, %105 : !tt.ptr<bf16>, i32
          %112 = tt.splat %111 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
          %113 = tt.addptr %112, %92 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
          %114 = tt.broadcast %113 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
          %115 = tt.addptr %114, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
          %116 = arith.muli %arg18, %arg13 : i32
          %117 = tt.addptr %arg4, %116 : !tt.ptr<f32>, i32
          %118 = tt.splat %117 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
          %119 = tt.addptr %118, %75 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
          %120 = tt.addptr %arg5, %116 : !tt.ptr<f32>, i32
          %121 = tt.splat %120 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
          %122 = tt.addptr %121, %75 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
          %123 = tt.load %110, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
          %124 = tt.load %122, %93, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
          %125 = tt.dot %123, %81, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
          %126 = arith.mulf %125, %28 {DataUse} : tensor<64x64xf32>
          %127 = arith.addf %126, %96 {DataUse} : tensor<64x64xf32>
          %128 = arith.select %29, %127, %cst_9 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
          %129 = tt.load %115, %78, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
          %130 = tt.expand_dims %124 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
          %131 = tt.broadcast %130 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
          %132 = arith.subf %128, %131 {DataUse} : tensor<64x64xf32>
          %133 = math.exp %132 {DataUse} : tensor<64x64xf32>
          %134 = arith.truncf %133 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
          %135 = tt.trans %134 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
          %136 = tt.dot %135, %129, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
          %137 = arith.addf %136, %arg20 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
          %138 = tt.load %119, %93, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
          %139 = tt.dot %129, %82, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
          %140 = tt.expand_dims %138 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
          %141 = tt.broadcast %140 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
          %142 = arith.subf %139, %141 {DataUse} : tensor<64x64xf32>
          %143 = arith.mulf %133, %142 {DataUse} : tensor<64x64xf32>
          %144 = arith.truncf %143 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
          %145 = tt.trans %144 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
          %146 = tt.dot %145, %123, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
          %147 = arith.mulf %146, %30 {DataUse} : tensor<64x128xf32>
          %148 = arith.addf %arg19, %147 {DataUse} : tensor<64x128xf32>
          %149:2 = scf.for %arg21 = %97 to %101 step %c1_i32 iter_args(%arg22 = %148, %arg23 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
            %161 = arith.muli %arg21, %c64_i32 : i32
            %162 = arith.addi %arg15, %161 : i32
            %163 = tt.splat %162 {MetaUse} : i32 -> tensor<64xi32>
            %164 = arith.addi %163, %2 {MetaUse} : tensor<64xi32>
            %165 = tt.expand_dims %164 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
            %166 = arith.muli %165, %cst_3 {MetaUse} : tensor<64x1xi32>
            %167 = tt.addptr %107, %166 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
            %168 = tt.broadcast %167 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
            %169 = tt.addptr %168, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
            %170 = tt.addptr %112, %166 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
            %171 = tt.broadcast %170 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
            %172 = tt.addptr %171, %21 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
            %173 = tt.addptr %118, %164 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            %174 = tt.addptr %121, %164 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            %175 = arith.cmpi slt, %165, %46 {MetaUse} : tensor<64x1xi32>
            %176 = arith.cmpi slt, %164, %49 {MetaUse} : tensor<64xi32>
            %177 = tt.broadcast %175 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
            %178 = tt.load %169, %177, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
            %179 = tt.load %174, %176, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
            %180 = tt.dot %178, %81, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
            %181 = arith.mulf %180, %28 {DataUse} : tensor<64x64xf32>
            %182 = tt.load %172, %177, %cst_10 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
            %183 = tt.expand_dims %179 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
            %184 = tt.broadcast %183 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
            %185 = arith.subf %181, %184 {DataUse} : tensor<64x64xf32>
            %186 = math.exp %185 {DataUse} : tensor<64x64xf32>
            %187 = arith.truncf %186 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
            %188 = tt.trans %187 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
            %189 = tt.dot %188, %182, %arg23 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
            %190 = tt.load %173, %176, %cst_4 {DataUse} : tensor<64x!tt.ptr<f32>>
            %191 = tt.dot %182, %82, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
            %192 = tt.expand_dims %190 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
            %193 = tt.broadcast %192 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
            %194 = arith.subf %191, %193 {DataUse} : tensor<64x64xf32>
            %195 = arith.mulf %186, %194 {DataUse} : tensor<64x64xf32>
            %196 = arith.truncf %195 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
            %197 = tt.trans %196 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
            %198 = tt.dot %197, %178, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
            %199 = arith.mulf %198, %30 {DataUse} : tensor<64x128xf32>
            %200 = arith.addf %arg22, %199 {DataUse} : tensor<64x128xf32>
            scf.yield %200, %189 : tensor<64x128xf32>, tensor<64x128xf32>
          } {DataUse, hivm.matmul_limited_in_cube}
          %150 = arith.cmpi sgt, %101, %97 : i32
          %151 = scf.if %150 -> (tensor<64x128xf32>) {
            scf.yield %149#1 : tensor<64x128xf32>
          } else {
            scf.yield %cst_11 : tensor<64x128xf32>
          } {DataUse}
          %152 = arith.addf %151, %137 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
          %153 = tt.splat %106 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
          %154 = tt.splat %111 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
          %155 = tt.splat %117 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
          %156 = tt.splat %120 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
          %157:2 = scf.for %arg21 = %99 to %51 step %c1_i32 iter_args(%arg22 = %149#0, %arg23 = %cst_11) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
            %161 = arith.muli %arg21, %c128_i32 : i32
            %162 = arith.addi %arg15, %161 : i32
            %163 = tt.splat %162 {MetaUse} : i32 -> tensor<128xi32>
            %164 = arith.addi %163, %17 {MetaUse} : tensor<128xi32>
            %165 = tt.expand_dims %164 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
            %166 = arith.muli %165, %cst {MetaUse} : tensor<128x1xi32>
            %167 = tt.addptr %153, %166 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
            %168 = tt.broadcast %167 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
            %169 = tt.addptr %168, %31 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
            %170 = tt.addptr %154, %166 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
            %171 = tt.broadcast %170 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
            %172 = tt.addptr %171, %31 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
            %173 = tt.addptr %155, %164 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            %174 = tt.addptr %156, %164 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            %175 = arith.cmpi slt, %165, %52 {MetaUse} : tensor<128x1xi32>
            %176 = arith.cmpi slt, %164, %53 {MetaUse} : tensor<128xi32>
            %177 = tt.broadcast %175 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
            %178 = tt.load %169, %177, %cst_2 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
            %179 = tt.load %174, %176, %cst_0 {DataUse} : tensor<128x!tt.ptr<f32>>
            %180 = tt.dot %178, %81, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
            %181 = arith.mulf %180, %32 {DataUse} : tensor<128x64xf32>
            %182 = tt.load %172, %177, %cst_2 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
            %183 = tt.expand_dims %179 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
            %184 = tt.broadcast %183 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
            %185 = arith.subf %181, %184 {DataUse} : tensor<128x64xf32>
            %186 = math.exp %185 {DataUse} : tensor<128x64xf32>
            %187 = arith.truncf %186 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
            %188 = tt.trans %187 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
            %189 = tt.dot %188, %182, %arg23 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
            %190 = tt.load %173, %176, %cst_0 {DataUse} : tensor<128x!tt.ptr<f32>>
            %191 = tt.dot %182, %82, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
            %192 = tt.expand_dims %190 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
            %193 = tt.broadcast %192 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
            %194 = arith.subf %191, %193 {DataUse} : tensor<128x64xf32>
            %195 = arith.mulf %186, %194 {DataUse} : tensor<128x64xf32>
            %196 = arith.truncf %195 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
            %197 = tt.trans %196 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
            %198 = tt.dot %197, %178, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
            %199 = arith.mulf %198, %30 {DataUse} : tensor<64x128xf32>
            %200 = arith.addf %arg22, %199 {DataUse} : tensor<64x128xf32>
            scf.yield %200, %189 : tensor<64x128xf32>, tensor<64x128xf32>
          } {DataUse, hivm.matmul_limited_in_cube}
          %158 = arith.cmpi sgt, %51, %99 : i32
          %159 = scf.if %158 -> (tensor<64x128xf32>) {
            scf.yield %157#1 : tensor<64x128xf32>
          } else {
            scf.yield %cst_11 : tensor<64x128xf32>
          } {DataUse}
          %160 = arith.addf %159, %152 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
          scf.yield %157#0, %160 : tensor<64x128xf32>, tensor<64x128xf32>
        } {DataUse}
        %103 = arith.truncf %102#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
        tt.store %69, %103, %78 : tensor<64x128x!tt.ptr<bf16>>
        %104 = arith.truncf %102#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
        tt.store %72, %104, %78 : tensor<64x128x!tt.ptr<bf16>>
      }
      scf.yield %35, %39 : i32, i32
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/dllm/bwd_q_u_sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 16)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_bwd_q_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i32>, %arg8: i32, %arg9: f32, %arg10: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
// CHECK-NEXT:     %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_11 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map()[%1]
// CHECK-NEXT:     %7 = tt.get_program_id x : i32
// CHECK-NEXT:     %8 = tt.get_num_programs x : i32
// CHECK-NEXT:     %9 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %10 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %11 = tt.expand_dims %9 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %12 = tt.expand_dims %10 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %13 = tt.expand_dims %9 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %14 = tt.expand_dims %10 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %15 = arith.muli %11, %cst {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:     %16 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %17 = tt.addptr %16, %15 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %18 = tt.broadcast %17 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %19 = tt.broadcast %13 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
// CHECK-NEXT:     %20 = tt.addptr %18, %19 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %21 = tt.bitcast %20 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %22 = tt.load %21 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %23 = tt.splat %arg10 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %24 = tt.addptr %23, %15 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %25 = tt.broadcast %24 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %26 = tt.addptr %25, %19 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %27 = tt.bitcast %26 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %28 = tt.load %27 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %29 = arith.remsi %7, %8 : i32
// CHECK-NEXT:     %30 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %31 = tt.expand_dims %30 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %32 = tt.broadcast %31 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %33 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x32xf32>
// CHECK-NEXT:     %34 = arith.cmpi ne, %22, %cst_5 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %35 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     %36 = arith.cmpi ne, %28, %cst_5 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %37 = tt.broadcast %31 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %38:2 = scf.for %arg14 = %c0_i32 to %arg8 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %39 = tt.addptr %arg7, %arg14 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %40 = tt.load %39 : !tt.ptr<i32>
// CHECK-NEXT:       %41 = arith.subi %40, %arg15 : i32
// CHECK-NEXT:       %42 = arith.addi %41, %c31_i32 : i32
// CHECK-NEXT:       %43 = arith.divsi %42, %c32_i32 : i32
// CHECK-NEXT:       %44 = arith.addi %arg16, %43 : i32
// CHECK-NEXT:       %45 = arith.muli %arg16, %c5_i32 : i32
// CHECK-NEXT:       %46 = arith.remsi %45, %8 : i32
// CHECK-NEXT:       %47 = arith.subi %29, %46 : i32
// CHECK-NEXT:       %48 = arith.addi %47, %8 : i32
// CHECK-NEXT:       %49 = arith.remsi %48, %8 : i32
// CHECK-NEXT:       %50 = arith.addi %45, %49 : i32
// CHECK-NEXT:       %51 = arith.muli %44, %c5_i32 : i32
// CHECK-NEXT:       %52 = tt.splat %40 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %53 = tt.splat %40 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %54 = tt.splat %40 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:       %55 = tt.splat %40 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:       %56 = arith.addi %arg12, %arg15 : i32
// CHECK-NEXT:       %57 = arith.addi %arg12, %40 : i32
// CHECK-NEXT:       %58 = tt.splat %57 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %59 = tt.splat %57 {MetaUse} : i32 -> tensor<128x1xi32>
// CHECK-NEXT:       scf.for %arg17 = %50 to %51 step %8  : i32 {
// CHECK-NEXT:         %60 = arith.divsi %arg17, %c5_i32 : i32
// CHECK-NEXT:         %61 = arith.subi %60, %arg16 : i32
// CHECK-NEXT:         %62 = arith.remsi %arg17, %c5_i32 : i32
// CHECK-NEXT:         %63 = arith.muli %62, %c128_i32 : i32
// CHECK-NEXT:         %64 = tt.addptr %arg0, %63 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %65 = arith.muli %61, %c32_i32 : i32
// CHECK-NEXT:         %66 = arith.addi %arg15, %65 : i32
// CHECK-NEXT:         %67 = tt.splat %66 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %68 = arith.addi %67, %9 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %69 = tt.expand_dims %68 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %70 = arith.muli %69, %cst_0 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %71 = tt.splat %64 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %72 = tt.addptr %71, %70 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %73 = tt.broadcast %72 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %74 = tt.addptr %73, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %75 = tt.addptr %arg3, %63 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %76 = tt.splat %75 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %77 = tt.addptr %76, %70 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %78 = tt.broadcast %77 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %79 = tt.addptr %78, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %80 = tt.addptr %arg6, %63 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %81 = tt.splat %80 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %82 = tt.addptr %81, %70 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %83 = tt.broadcast %82 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %84 = tt.addptr %83, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %85 = arith.muli %62, %arg13 : i32
// CHECK-NEXT:         %86 = tt.addptr %arg4, %85 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %87 = tt.splat %86 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %88 = tt.addptr %87, %68 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %89 = tt.addptr %arg5, %85 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %90 = tt.splat %89 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %91 = tt.addptr %90, %68 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %92 = arith.cmpi slt, %69, %52 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %93 = arith.cmpi slt, %68, %54 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %94 = tt.broadcast %92 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %95 = tt.load %74, %94, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %96 = tt.load %79, %94, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %97 = tt.load %91, %93, %cst_2 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %98 = tt.load %88, %93, %cst_2 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %99 = tt.splat %66 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:         %100 = arith.addi %99, %12 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %101 = arith.cmpi slt, %100, %53 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %102 = tt.splat %66 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:         %103 = arith.addi %102, %14 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %104 = arith.cmpi slt, %103, %55 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %105 = tt.broadcast %101 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %106 = tt.broadcast %104 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %107 = arith.andi %105, %106 {DataUse} : tensor<32x32xi1>
// CHECK-NEXT:         %108 = arith.divsi %62, %c5_i32 : i32
// CHECK-NEXT:         %109 = arith.muli %108, %c128_i32 : i32
// CHECK-NEXT:         %110 = tt.addptr %arg1, %109 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %111 = arith.muli %69, %cst_9 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %112 = tt.splat %110 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %113 = tt.addptr %112, %111 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %114 = tt.broadcast %113 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %115 = tt.addptr %114, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %116 = tt.addptr %arg2, %109 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %117 = tt.splat %116 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %118 = tt.addptr %117, %111 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %119 = tt.broadcast %118 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %120 = tt.addptr %119, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %121 = tt.load %115, %94, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %122 = tt.trans %121 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %123 = tt.dot %95, %122, %cst_8 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %124 = arith.mulf %123, %33 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %125 = arith.uitofp %107 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
// CHECK-NEXT:         %126 = arith.subf %125, %cst_7 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %127 = arith.mulf %126, %cst_6 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %128 = arith.addf %124, %127 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %129 = arith.select %34, %128, %cst_4 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %130 = tt.expand_dims %97 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %131 = tt.broadcast %130 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %132 = arith.subf %129, %131 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %133 = math.exp %132 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %134 = tt.load %120, %94, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %135 = tt.trans %134 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %136 = tt.dot %96, %135, %cst_8 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %137 = tt.expand_dims %98 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %138 = tt.broadcast %137 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %139 = arith.subf %136, %138 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %140 = arith.mulf %133, %139 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %141 = arith.truncf %140 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %141[%2, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:         %142 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice into %142[%2, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:         %143 = tt.dot %inserted_slice, %121, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %144 = arith.mulf %143, %35 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %145 = arith.addf %144, %cst_1 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %146 = arith.addi %56, %65 : i32
// CHECK-NEXT:         %147 = tt.splat %146 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %148 = arith.addi %147, %9 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %149 = tt.expand_dims %148 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %150 = arith.muli %149, %cst_9 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %151 = tt.addptr %112, %150 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %152 = tt.broadcast %151 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %153 = tt.addptr %152, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %154 = tt.addptr %117, %150 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %155 = tt.broadcast %154 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %156 = tt.addptr %155, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %157 = arith.cmpi slt, %149, %58 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %158 = tt.broadcast %157 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %159 = tt.load %153, %158, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %160 = tt.trans %159 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %161 = tt.dot %95, %160, %cst_8 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %162 = arith.mulf %161, %33 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %163 = arith.addf %162, %127 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %164 = arith.select %36, %163, %cst_4 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %165 = arith.subf %164, %131 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %166 = math.exp %165 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %167 = tt.load %156, %158, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %168 = tt.trans %167 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %169 = tt.dot %96, %168, %cst_8 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %170 = arith.subf %169, %138 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %171 = arith.mulf %166, %170 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %172 = arith.truncf %171 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %172[%3, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:         %173 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:         %inserted_slice_13 = tensor.insert_slice %extracted_slice_12 into %173[%3, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:         %174 = tt.dot %inserted_slice_13, %159, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %175 = arith.mulf %174, %35 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %176 = arith.addf %145, %175 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %177 = arith.divsi %65, %c128_i32 : i32
// CHECK-NEXT:         %178 = arith.muli %177, %c128_i32 : i32
// CHECK-NEXT:         %179 = arith.divsi %178, %c32_i32 : i32
// CHECK-NEXT:         %180 = scf.for %arg18 = %179 to %61 step %c1_i32 iter_args(%arg19 = %176) -> (tensor<32x128xf32>)  : i32 {
// CHECK-NEXT:           %187 = arith.muli %arg18, %c32_i32 : i32
// CHECK-NEXT:           %188 = arith.addi %56, %187 : i32
// CHECK-NEXT:           %189 = tt.splat %188 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:           %190 = arith.addi %189, %9 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:           %191 = tt.expand_dims %190 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:           %192 = arith.muli %191, %cst_9 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %193 = tt.addptr %112, %192 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %194 = tt.broadcast %193 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %195 = tt.addptr %194, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %196 = tt.addptr %117, %192 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %197 = tt.broadcast %196 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %198 = tt.addptr %197, %32 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %199 = arith.cmpi slt, %191, %58 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %200 = tt.broadcast %199 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:           %201 = tt.load %195, %200, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %202 = tt.trans %201 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:           %203 = tt.dot %95, %202, %cst_8 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %204 = arith.mulf %203, %33 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %205 = arith.subf %204, %131 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %206 = math.exp %205 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %207 = tt.load %198, %200, %cst_3 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %208 = tt.trans %207 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:           %209 = tt.dot %96, %208, %cst_8 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %210 = arith.subf %209, %138 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %211 = arith.mulf %206, %210 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %212 = arith.truncf %211 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %extracted_slice_17 = tensor.extract_slice %212[%4, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:           %213 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:           %inserted_slice_18 = tensor.insert_slice %extracted_slice_17 into %213[%4, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:           %214 = tt.dot %inserted_slice_18, %201, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %215 = arith.mulf %214, %35 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %216 = arith.addf %arg19, %215 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %216 : tensor<32x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %181 = tt.splat %110 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %182 = tt.splat %116 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %183 = tt.broadcast %130 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %184 = tt.broadcast %137 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %185 = scf.for %arg18 = %c0_i32 to %177 step %c1_i32 iter_args(%arg19 = %180) -> (tensor<32x128xf32>)  : i32 {
// CHECK-NEXT:           %187 = arith.muli %arg18, %c128_i32 : i32
// CHECK-NEXT:           %188 = arith.addi %56, %187 : i32
// CHECK-NEXT:           %189 = tt.splat %188 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %190 = arith.addi %189, %30 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %191 = tt.expand_dims %190 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %192 = arith.muli %191, %cst_11 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %193 = tt.addptr %181, %192 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %194 = tt.broadcast %193 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %195 = tt.addptr %194, %37 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %196 = tt.addptr %182, %192 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %197 = tt.broadcast %196 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %198 = tt.addptr %197, %37 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %199 = arith.cmpi slt, %191, %59 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %200 = tt.broadcast %199 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %201 = tt.load %195, %200, %cst_10 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %202 = tt.trans %201 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:           %203 = tt.dot %95, %202, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %204 = arith.mulf %203, %35 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %205 = arith.subf %204, %183 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %206 = math.exp %205 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %207 = tt.load %198, %200, %cst_10 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %208 = tt.trans %207 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:           %209 = tt.dot %96, %208, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %210 = arith.subf %209, %184 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %211 = arith.mulf %206, %210 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %212 = arith.truncf %211 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:           %extracted_slice_17 = tensor.extract_slice %212[%5, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xbf16> to tensor<16x128xbf16>
// CHECK-NEXT:           %213 = tensor.empty() : tensor<32x128xbf16>
// CHECK-NEXT:           %inserted_slice_18 = tensor.insert_slice %extracted_slice_17 into %213[%5, 0] [16, 128] [1, 1] {cv_communication_slice} : tensor<16x128xbf16> into tensor<32x128xbf16>
// CHECK-NEXT:           %214 = tt.dot %inserted_slice_18, %201, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %215 = arith.mulf %214, %35 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %216 = arith.addf %arg19, %215 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %216 : tensor<32x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %186 = arith.truncf %185 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %186[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xbf16> to tensor<16x128xbf16>
// CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %84[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128x!tt.ptr<bf16>> to tensor<16x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %94[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xi1> to tensor<16x128xi1>
// CHECK-NEXT:         tt.store %extracted_slice_15, %extracted_slice_14, %extracted_slice_16 {tiled_op} : tensor<16x128x!tt.ptr<bf16>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %40, %44 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_bwd_q_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i32>, %arg8: i32, %arg9: f32, %arg10: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32) attributes {noinline = false} {
    %cst = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_1 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
    %cst_3 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
    %cst_4 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
    %cst_5 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
    %cst_6 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %cst_7 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_8 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32xf32>
    %cst_9 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
    %cst_10 = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_11 = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %8 = arith.muli %4, %cst_11 {MetaUse} : tensor<32x1xi32>
    %9 = tt.splat %arg11 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %10 = tt.addptr %9, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %11 = tt.broadcast %10 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %12 = tt.broadcast %6 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
    %13 = tt.addptr %11, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %14 = tt.bitcast %13 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %16 = tt.splat %arg10 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %17 = tt.addptr %16, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %18 = tt.broadcast %17 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %19 = tt.addptr %18, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %20 = tt.bitcast %19 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %21 = tt.load %20 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %22 = arith.remsi %0, %1 : i32
    %23 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %24 = tt.expand_dims %23 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %25 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
    %26 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x32xf32>
    %27 = arith.cmpi ne, %15, %cst_5 {DataUse} : tensor<32x32xi8>
    %28 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x128xf32>
    %29 = arith.cmpi ne, %21, %cst_5 {DataUse} : tensor<32x32xi8>
    %30 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %31:2 = scf.for %arg14 = %c0_i32 to %arg8 step %c1_i32 iter_args(%arg15 = %c0_i32, %arg16 = %c0_i32) -> (i32, i32)  : i32 {
      %32 = tt.addptr %arg7, %arg14 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = arith.subi %33, %arg15 : i32
      %35 = arith.addi %34, %c31_i32 : i32
      %36 = arith.divsi %35, %c32_i32 : i32
      %37 = arith.addi %arg16, %36 : i32
      %38 = arith.muli %arg16, %c5_i32 : i32
      %39 = arith.remsi %38, %1 : i32
      %40 = arith.subi %22, %39 : i32
      %41 = arith.addi %40, %1 : i32
      %42 = arith.remsi %41, %1 : i32
      %43 = arith.addi %38, %42 : i32
      %44 = arith.muli %37, %c5_i32 : i32
      %45 = tt.splat %33 {MetaUse} : i32 -> tensor<32x1xi32>
      %46 = tt.splat %33 {DataUse} : i32 -> tensor<32x1xi32>
      %47 = tt.splat %33 {MetaUse} : i32 -> tensor<32xi32>
      %48 = tt.splat %33 {DataUse} : i32 -> tensor<1x32xi32>
      %49 = arith.addi %arg12, %arg15 : i32
      %50 = arith.addi %arg12, %33 : i32
      %51 = tt.splat %50 {MetaUse} : i32 -> tensor<32x1xi32>
      %52 = tt.splat %50 {MetaUse} : i32 -> tensor<128x1xi32>
      scf.for %arg17 = %43 to %44 step %1  : i32 {
        %53 = arith.divsi %arg17, %c5_i32 : i32
        %54 = arith.subi %53, %arg16 : i32
        %55 = arith.remsi %arg17, %c5_i32 : i32
        %56 = arith.muli %55, %c128_i32 : i32
        %57 = tt.addptr %arg0, %56 : !tt.ptr<bf16>, i32
        %58 = arith.muli %54, %c32_i32 : i32
        %59 = arith.addi %arg15, %58 : i32
        %60 = tt.splat %59 {MetaUse} : i32 -> tensor<32xi32>
        %61 = arith.addi %60, %2 {MetaUse} : tensor<32xi32>
        %62 = tt.expand_dims %61 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %63 = arith.muli %62, %cst_10 {MetaUse} : tensor<32x1xi32>
        %64 = tt.splat %57 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %66 = tt.broadcast %65 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %67 = tt.addptr %66, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %68 = tt.addptr %arg3, %56 : !tt.ptr<bf16>, i32
        %69 = tt.splat %68 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %70 = tt.addptr %69, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %71 = tt.broadcast %70 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %72 = tt.addptr %71, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %73 = tt.addptr %arg6, %56 : !tt.ptr<bf16>, i32
        %74 = tt.splat %73 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %75 = tt.addptr %74, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %76 = tt.broadcast %75 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %77 = tt.addptr %76, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %78 = arith.muli %55, %arg13 : i32
        %79 = tt.addptr %arg4, %78 : !tt.ptr<f32>, i32
        %80 = tt.splat %79 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
        %81 = tt.addptr %80, %61 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %82 = tt.addptr %arg5, %78 : !tt.ptr<f32>, i32
        %83 = tt.splat %82 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
        %84 = tt.addptr %83, %61 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %85 = arith.cmpi slt, %62, %45 {MetaUse} : tensor<32x1xi32>
        %86 = arith.cmpi slt, %61, %47 {MetaUse} : tensor<32xi32>
        %87 = tt.broadcast %85 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
        %88 = tt.load %67, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %89 = tt.load %72, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %90 = tt.load %84, %86, %cst_8 {DataUse} : tensor<32x!tt.ptr<f32>>
        %91 = tt.load %81, %86, %cst_8 {DataUse} : tensor<32x!tt.ptr<f32>>
        %92 = tt.splat %59 {DataUse} : i32 -> tensor<32x1xi32>
        %93 = arith.addi %92, %5 {DataUse} : tensor<32x1xi32>
        %94 = arith.cmpi slt, %93, %46 {DataUse} : tensor<32x1xi32>
        %95 = tt.splat %59 {DataUse} : i32 -> tensor<1x32xi32>
        %96 = arith.addi %95, %7 {DataUse} : tensor<1x32xi32>
        %97 = arith.cmpi slt, %96, %48 {DataUse} : tensor<1x32xi32>
        %98 = tt.broadcast %94 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
        %99 = tt.broadcast %97 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
        %100 = arith.andi %98, %99 {DataUse} : tensor<32x32xi1>
        %101 = arith.divsi %55, %c5_i32 : i32
        %102 = arith.muli %101, %c128_i32 : i32
        %103 = tt.addptr %arg1, %102 : !tt.ptr<bf16>, i32
        %104 = arith.muli %62, %cst_1 {MetaUse} : tensor<32x1xi32>
        %105 = tt.splat %103 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %106 = tt.addptr %105, %104 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %107 = tt.broadcast %106 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %108 = tt.addptr %107, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %109 = tt.addptr %arg2, %102 : !tt.ptr<bf16>, i32
        %110 = tt.splat %109 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %111 = tt.addptr %110, %104 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %112 = tt.broadcast %111 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %113 = tt.addptr %112, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %114 = tt.load %108, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %115 = tt.trans %114 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %116 = tt.dot %88, %115, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %117 = arith.mulf %116, %26 {DataUse} : tensor<32x32xf32>
        %118 = arith.uitofp %100 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
        %119 = arith.subf %118, %cst_3 {DataUse} : tensor<32x32xf32>
        %120 = arith.mulf %119, %cst_4 {DataUse} : tensor<32x32xf32>
        %121 = arith.addf %117, %120 {DataUse} : tensor<32x32xf32>
        %122 = arith.select %27, %121, %cst_6 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
        %123 = tt.expand_dims %90 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %124 = tt.broadcast %123 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
        %125 = arith.subf %122, %124 {DataUse} : tensor<32x32xf32>
        %126 = math.exp %125 {DataUse} : tensor<32x32xf32>
        %127 = tt.load %113, %87, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %128 = tt.trans %127 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %129 = tt.dot %89, %128, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %130 = tt.expand_dims %91 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %131 = tt.broadcast %130 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
        %132 = arith.subf %129, %131 {DataUse} : tensor<32x32xf32>
        %133 = arith.mulf %126, %132 {DataUse} : tensor<32x32xf32>
        %134 = arith.truncf %133 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
        %135 = tt.dot %134, %114, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %136 = arith.mulf %135, %28 {DataUse} : tensor<32x128xf32>
        %137 = arith.addf %136, %cst_9 {DataUse} : tensor<32x128xf32>
        %138 = arith.addi %49, %58 : i32
        %139 = tt.splat %138 {MetaUse} : i32 -> tensor<32xi32>
        %140 = arith.addi %139, %2 {MetaUse} : tensor<32xi32>
        %141 = tt.expand_dims %140 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %142 = arith.muli %141, %cst_1 {MetaUse} : tensor<32x1xi32>
        %143 = tt.addptr %105, %142 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %144 = tt.broadcast %143 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %145 = tt.addptr %144, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %146 = tt.addptr %110, %142 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %147 = tt.broadcast %146 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %148 = tt.addptr %147, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %149 = arith.cmpi slt, %141, %51 {MetaUse} : tensor<32x1xi32>
        %150 = tt.broadcast %149 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
        %151 = tt.load %145, %150, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %152 = tt.trans %151 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %153 = tt.dot %88, %152, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %154 = arith.mulf %153, %26 {DataUse} : tensor<32x32xf32>
        %155 = arith.addf %154, %120 {DataUse} : tensor<32x32xf32>
        %156 = arith.select %29, %155, %cst_6 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
        %157 = arith.subf %156, %124 {DataUse} : tensor<32x32xf32>
        %158 = math.exp %157 {DataUse} : tensor<32x32xf32>
        %159 = tt.load %148, %150, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %160 = tt.trans %159 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %161 = tt.dot %89, %160, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %162 = arith.subf %161, %131 {DataUse} : tensor<32x32xf32>
        %163 = arith.mulf %158, %162 {DataUse} : tensor<32x32xf32>
        %164 = arith.truncf %163 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
        %165 = tt.dot %164, %151, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %166 = arith.mulf %165, %28 {DataUse} : tensor<32x128xf32>
        %167 = arith.addf %137, %166 {DataUse} : tensor<32x128xf32>
        %168 = arith.divsi %58, %c128_i32 : i32
        %169 = arith.muli %168, %c128_i32 : i32
        %170 = arith.divsi %169, %c32_i32 : i32
        %171 = scf.for %arg18 = %170 to %54 step %c1_i32 iter_args(%arg19 = %167) -> (tensor<32x128xf32>)  : i32 {
          %178 = arith.muli %arg18, %c32_i32 : i32
          %179 = arith.addi %49, %178 : i32
          %180 = tt.splat %179 {MetaUse} : i32 -> tensor<32xi32>
          %181 = arith.addi %180, %2 {MetaUse} : tensor<32xi32>
          %182 = tt.expand_dims %181 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
          %183 = arith.muli %182, %cst_1 {MetaUse} : tensor<32x1xi32>
          %184 = tt.addptr %105, %183 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %185 = tt.broadcast %184 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %186 = tt.addptr %185, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %187 = tt.addptr %110, %183 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %188 = tt.broadcast %187 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %189 = tt.addptr %188, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %190 = arith.cmpi slt, %182, %51 {MetaUse} : tensor<32x1xi32>
          %191 = tt.broadcast %190 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
          %192 = tt.load %186, %191, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
          %193 = tt.trans %192 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
          %194 = tt.dot %88, %193, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %195 = arith.mulf %194, %26 {DataUse} : tensor<32x32xf32>
          %196 = arith.subf %195, %124 {DataUse} : tensor<32x32xf32>
          %197 = math.exp %196 {DataUse} : tensor<32x32xf32>
          %198 = tt.load %189, %191, %cst_7 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
          %199 = tt.trans %198 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
          %200 = tt.dot %89, %199, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %201 = arith.subf %200, %131 {DataUse} : tensor<32x32xf32>
          %202 = arith.mulf %197, %201 {DataUse} : tensor<32x32xf32>
          %203 = arith.truncf %202 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
          %204 = tt.dot %203, %192, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %205 = arith.mulf %204, %28 {DataUse} : tensor<32x128xf32>
          %206 = arith.addf %arg19, %205 {DataUse} : tensor<32x128xf32>
          scf.yield %206 : tensor<32x128xf32>
        } {DataUse}
        %172 = tt.splat %103 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %173 = tt.splat %109 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %174 = tt.broadcast %123 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
        %175 = tt.broadcast %130 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
        %176 = scf.for %arg18 = %c0_i32 to %168 step %c1_i32 iter_args(%arg19 = %171) -> (tensor<32x128xf32>)  : i32 {
          %178 = arith.muli %arg18, %c128_i32 : i32
          %179 = arith.addi %49, %178 : i32
          %180 = tt.splat %179 {MetaUse} : i32 -> tensor<128xi32>
          %181 = arith.addi %180, %23 {MetaUse} : tensor<128xi32>
          %182 = tt.expand_dims %181 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %183 = arith.muli %182, %cst {MetaUse} : tensor<128x1xi32>
          %184 = tt.addptr %172, %183 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %185 = tt.broadcast %184 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %186 = tt.addptr %185, %30 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %187 = tt.addptr %173, %183 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %188 = tt.broadcast %187 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %189 = tt.addptr %188, %30 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %190 = arith.cmpi slt, %182, %52 {MetaUse} : tensor<128x1xi32>
          %191 = tt.broadcast %190 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
          %192 = tt.load %186, %191, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %193 = tt.trans %192 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
          %194 = tt.dot %88, %193, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %195 = arith.mulf %194, %28 {DataUse} : tensor<32x128xf32>
          %196 = arith.subf %195, %174 {DataUse} : tensor<32x128xf32>
          %197 = math.exp %196 {DataUse} : tensor<32x128xf32>
          %198 = tt.load %189, %191, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %199 = tt.trans %198 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
          %200 = tt.dot %89, %199, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %201 = arith.subf %200, %175 {DataUse} : tensor<32x128xf32>
          %202 = arith.mulf %197, %201 {DataUse} : tensor<32x128xf32>
          %203 = arith.truncf %202 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
          %204 = tt.dot %203, %192, %cst_9 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %205 = arith.mulf %204, %28 {DataUse} : tensor<32x128xf32>
          %206 = arith.addf %arg19, %205 {DataUse} : tensor<32x128xf32>
          scf.yield %206 : tensor<32x128xf32>
        } {DataUse}
        %177 = arith.truncf %176 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
        tt.store %77, %177, %87 : tensor<32x128x!tt.ptr<bf16>>
      }
      scf.yield %33, %37 : i32, i32
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/dllm/fwd_u_sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 16)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_da_fwd_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32>, %arg6: i32, %arg7: f32, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
// CHECK-NEXT:     %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_11 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_13 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x1xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map()[%1]
// CHECK-NEXT:     %7 = affine.apply #map()[%1]
// CHECK-NEXT:     %8 = tt.get_program_id x : i32
// CHECK-NEXT:     %9 = tt.get_num_programs x : i32
// CHECK-NEXT:     %10 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %11 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %12 = tt.expand_dims %10 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %13 = tt.expand_dims %11 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:     %14 = tt.expand_dims %10 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %15 = tt.expand_dims %11 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
// CHECK-NEXT:     %16 = arith.muli %12, %cst {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:     %17 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %18 = tt.addptr %17, %16 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %19 = tt.broadcast %18 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %20 = tt.broadcast %14 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
// CHECK-NEXT:     %21 = tt.addptr %19, %20 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %22 = tt.bitcast %21 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %23 = tt.load %22 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %24 = tt.splat %arg9 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
// CHECK-NEXT:     %25 = tt.addptr %24, %16 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
// CHECK-NEXT:     %26 = tt.broadcast %25 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
// CHECK-NEXT:     %27 = tt.addptr %26, %20 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
// CHECK-NEXT:     %28 = tt.bitcast %27 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %29 = tt.load %28 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
// CHECK-NEXT:     %30 = arith.remsi %8, %9 : i32
// CHECK-NEXT:     %31 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %32 = tt.expand_dims %31 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %33 = tt.broadcast %32 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %34 = tt.splat %arg7 {DataUse} : f32 -> tensor<32x32xf32>
// CHECK-NEXT:     %35 = arith.cmpi ne, %23, %cst_6 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %36 = arith.cmpi ne, %29, %cst_6 {DataUse} : tensor<32x32xi8>
// CHECK-NEXT:     %37 = tt.broadcast %32 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %38 = tt.splat %arg7 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     %39:2 = scf.for %arg12 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
// CHECK-NEXT:       %40 = tt.addptr %arg5, %arg12 : !tt.ptr<i32>, i32
// CHECK-NEXT:       %41 = tt.load %40 : !tt.ptr<i32>
// CHECK-NEXT:       %42 = arith.subi %41, %arg13 : i32
// CHECK-NEXT:       %43 = arith.addi %42, %c31_i32 : i32
// CHECK-NEXT:       %44 = arith.divsi %43, %c32_i32 : i32
// CHECK-NEXT:       %45 = arith.addi %arg14, %44 : i32
// CHECK-NEXT:       %46 = arith.muli %arg14, %c5_i32 : i32
// CHECK-NEXT:       %47 = arith.remsi %46, %9 : i32
// CHECK-NEXT:       %48 = arith.subi %30, %47 : i32
// CHECK-NEXT:       %49 = arith.addi %48, %9 : i32
// CHECK-NEXT:       %50 = arith.remsi %49, %9 : i32
// CHECK-NEXT:       %51 = arith.addi %46, %50 : i32
// CHECK-NEXT:       %52 = arith.muli %45, %c5_i32 : i32
// CHECK-NEXT:       %53 = tt.splat %41 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %54 = tt.splat %41 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %55 = tt.splat %41 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:       %56 = tt.splat %41 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:       %57 = arith.addi %arg10, %arg13 : i32
// CHECK-NEXT:       %58 = arith.addi %arg10, %41 : i32
// CHECK-NEXT:       %59 = tt.splat %58 {MetaUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:       %60 = tt.splat %58 {MetaUse} : i32 -> tensor<128x1xi32>
// CHECK-NEXT:       scf.for %arg15 = %51 to %52 step %9  : i32 {
// CHECK-NEXT:         %61 = arith.divsi %arg15, %c5_i32 : i32
// CHECK-NEXT:         %62 = arith.subi %61, %arg14 : i32
// CHECK-NEXT:         %63 = arith.remsi %arg15, %c5_i32 : i32
// CHECK-NEXT:         %64 = arith.muli %63, %c128_i32 : i32
// CHECK-NEXT:         %65 = tt.addptr %arg0, %64 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %66 = arith.muli %62, %c32_i32 : i32
// CHECK-NEXT:         %67 = arith.addi %arg13, %66 : i32
// CHECK-NEXT:         %68 = tt.splat %67 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %69 = arith.addi %68, %10 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %70 = tt.expand_dims %69 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %71 = arith.muli %70, %cst_0 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %72 = tt.splat %65 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %73 = tt.addptr %72, %71 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %74 = tt.broadcast %73 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %75 = tt.addptr %74, %33 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %76 = tt.addptr %arg3, %64 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %77 = tt.splat %76 {MetaUse} : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
// CHECK-NEXT:         %78 = tt.addptr %77, %71 {MetaUse} : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
// CHECK-NEXT:         %79 = tt.broadcast %78 {MetaUse} : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
// CHECK-NEXT:         %80 = tt.addptr %79, %33 {MetaUse} : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
// CHECK-NEXT:         %81 = arith.muli %63, %arg11 : i32
// CHECK-NEXT:         %82 = tt.addptr %arg4, %81 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %83 = tt.splat %82 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %84 = tt.addptr %83, %69 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %85 = arith.cmpi slt, %70, %53 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %86 = arith.cmpi slt, %69, %55 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %87 = tt.broadcast %85 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %88 = tt.load %75, %87, %cst_4 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %89 = tt.splat %67 {DataUse} : i32 -> tensor<32x1xi32>
// CHECK-NEXT:         %90 = arith.addi %89, %13 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %91 = arith.cmpi slt, %90, %54 {DataUse} : tensor<32x1xi32>
// CHECK-NEXT:         %92 = tt.splat %67 {DataUse} : i32 -> tensor<1x32xi32>
// CHECK-NEXT:         %93 = arith.addi %92, %15 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %94 = arith.cmpi slt, %93, %56 {DataUse} : tensor<1x32xi32>
// CHECK-NEXT:         %95 = tt.broadcast %91 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %96 = tt.broadcast %94 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
// CHECK-NEXT:         %97 = arith.andi %95, %96 {DataUse} : tensor<32x32xi1>
// CHECK-NEXT:         %98 = arith.divsi %63, %c5_i32 : i32
// CHECK-NEXT:         %99 = arith.muli %98, %c128_i32 : i32
// CHECK-NEXT:         %100 = tt.addptr %arg1, %99 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %101 = arith.muli %70, %cst_10 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %102 = tt.splat %100 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %103 = tt.addptr %102, %101 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %104 = tt.broadcast %103 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %105 = tt.addptr %104, %33 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %106 = tt.addptr %arg2, %99 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %107 = tt.splat %106 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %108 = tt.addptr %107, %101 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %109 = tt.broadcast %108 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %110 = tt.addptr %109, %33 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %111 = tt.load %105, %87, %cst_4 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %112 = tt.trans %111 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %113 = tt.dot %88, %112, %cst_9 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %114 = arith.mulf %113, %34 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %115 = tt.load %110, %87, %cst_4 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %116 = arith.uitofp %97 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
// CHECK-NEXT:         %117 = arith.subf %116, %cst_8 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %118 = arith.mulf %117, %cst_7 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %119 = arith.addf %114, %118 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %120 = arith.select %35, %119, %cst_5 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %121 = "tt.reduce"(%120) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %189 = arith.maxnumf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %189 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %122 = arith.maxnumf %121, %cst_3 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %123 = tt.expand_dims %122 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %124 = tt.broadcast %123 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %125 = arith.subf %120, %124 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %126 = math.exp %125 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %127 = arith.subf %cst_3, %122 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %128 = math.exp %127 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %129 = arith.mulf %128, %cst_2 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %130 = "tt.reduce"(%126) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %189 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %189 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %131 = arith.addf %129, %130 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %132 = tt.expand_dims %128 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %133 = arith.mulf %132, %cst_13 {DataUse} : tensor<32x1xf32>
// CHECK-NEXT:         %134 = tt.broadcast %133 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %135 = arith.truncf %126 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %135[%2, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:         %136 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice into %136[%2, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:         %137 = tt.dot %inserted_slice, %115, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %138 = arith.addf %137, %134 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:         %139 = arith.addi %57, %66 : i32
// CHECK-NEXT:         %140 = tt.splat %139 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %141 = arith.addi %140, %10 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %142 = tt.expand_dims %141 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %143 = arith.muli %142, %cst_10 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %144 = tt.addptr %102, %143 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %145 = tt.broadcast %144 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %146 = tt.addptr %145, %33 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %147 = tt.addptr %107, %143 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:         %148 = tt.broadcast %147 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %149 = tt.addptr %148, %33 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:         %150 = arith.cmpi slt, %142, %59 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %151 = tt.broadcast %150 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:         %152 = tt.load %146, %151, %cst_4 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %153 = tt.trans %152 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:         %154 = tt.dot %88, %153, %cst_9 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:         %155 = arith.mulf %154, %34 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %156 = tt.load %149, %151, %cst_4 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %157 = arith.addf %155, %118 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %158 = arith.select %36, %157, %cst_5 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK-NEXT:         %159 = "tt.reduce"(%158) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %189 = arith.maxnumf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %189 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %160 = arith.maxnumf %122, %159 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %161 = tt.expand_dims %160 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %162 = tt.broadcast %161 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:         %163 = arith.subf %158, %162 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %164 = math.exp %163 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:         %165 = arith.subf %122, %160 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %166 = math.exp %165 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %167 = arith.mulf %166, %131 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %168 = "tt.reduce"(%164) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %189 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %189 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:         %169 = arith.addf %167, %168 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %170 = tt.expand_dims %166 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %171 = tt.broadcast %170 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %172 = arith.mulf %171, %138 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %173 = arith.truncf %164 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %173[%3, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:         %174 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:         %inserted_slice_15 = tensor.insert_slice %extracted_slice_14 into %174[%3, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:         %175 = tt.dot %inserted_slice_15, %156, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %176 = arith.addf %175, %172 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:         %177 = arith.divsi %66, %c128_i32 : i32
// CHECK-NEXT:         %178 = arith.muli %177, %c128_i32 : i32
// CHECK-NEXT:         %179 = arith.divsi %178, %c32_i32 : i32
// CHECK-NEXT:         %180:3 = scf.for %arg16 = %179 to %62 step %c1_i32 iter_args(%arg17 = %176, %arg18 = %160, %arg19 = %169) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
// CHECK-NEXT:           %189 = arith.muli %arg16, %c32_i32 : i32
// CHECK-NEXT:           %190 = arith.addi %57, %189 : i32
// CHECK-NEXT:           %191 = tt.splat %190 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:           %192 = arith.addi %191, %10 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:           %193 = tt.expand_dims %192 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:           %194 = arith.muli %193, %cst_10 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %195 = tt.addptr %102, %194 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %196 = tt.broadcast %195 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %197 = tt.addptr %196, %33 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %198 = tt.addptr %107, %194 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
// CHECK-NEXT:           %199 = tt.broadcast %198 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %200 = tt.addptr %199, %33 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
// CHECK-NEXT:           %201 = arith.cmpi slt, %193, %59 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:           %202 = tt.broadcast %201 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
// CHECK-NEXT:           %203 = tt.load %197, %202, %cst_4 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %204 = tt.trans %203 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
// CHECK-NEXT:           %205 = tt.dot %88, %204, %cst_9 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
// CHECK-NEXT:           %206 = arith.mulf %205, %34 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %207 = tt.load %200, %202, %cst_4 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %208 = "tt.reduce"(%206) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %226 = arith.maxnumf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %226 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %209 = arith.maxnumf %arg18, %208 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %210 = tt.expand_dims %209 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %211 = tt.broadcast %210 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
// CHECK-NEXT:           %212 = arith.subf %206, %211 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %213 = math.exp %212 {DataUse} : tensor<32x32xf32>
// CHECK-NEXT:           %214 = arith.subf %arg18, %209 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %215 = math.exp %214 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %216 = arith.mulf %215, %arg19 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %217 = "tt.reduce"(%213) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %226 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %226 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %218 = arith.addf %216, %217 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %219 = tt.expand_dims %215 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %220 = tt.broadcast %219 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:           %221 = arith.mulf %220, %arg17 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %222 = arith.truncf %213 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
// CHECK-NEXT:           %extracted_slice_22 = tensor.extract_slice %222[%4, 0] [16, 32] [1, 1] {to_be_bubbled_slice} : tensor<32x32xbf16> to tensor<16x32xbf16>
// CHECK-NEXT:           %223 = tensor.empty() : tensor<32x32xbf16>
// CHECK-NEXT:           %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %223[%4, 0] [16, 32] [1, 1] {cv_communication_slice} : tensor<16x32xbf16> into tensor<32x32xbf16>
// CHECK-NEXT:           %224 = tt.dot %inserted_slice_23, %207, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %225 = arith.addf %224, %221 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %225, %209, %218 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %181 = tt.splat %100 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %182 = tt.splat %106 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %183:3 = scf.for %arg16 = %c0_i32 to %177 step %c1_i32 iter_args(%arg17 = %180#0, %arg18 = %180#1, %arg19 = %180#2) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
// CHECK-NEXT:           %189 = arith.muli %arg16, %c128_i32 : i32
// CHECK-NEXT:           %190 = arith.addi %57, %189 : i32
// CHECK-NEXT:           %191 = tt.splat %190 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %192 = arith.addi %191, %31 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %193 = tt.expand_dims %192 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %194 = arith.muli %193, %cst_12 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %195 = tt.addptr %181, %194 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %196 = tt.broadcast %195 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %197 = tt.addptr %196, %37 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %198 = tt.addptr %182, %194 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %199 = tt.broadcast %198 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %200 = tt.addptr %199, %37 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %201 = arith.cmpi slt, %193, %60 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %202 = tt.broadcast %201 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %203 = tt.load %197, %202, %cst_11 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %204 = tt.trans %203 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:           %205 = tt.dot %88, %204, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %206 = arith.mulf %205, %38 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %207 = tt.load %200, %202, %cst_11 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %208 = "tt.reduce"(%206) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %226 = arith.maxnumf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %226 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x128xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %209 = arith.maxnumf %arg18, %208 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %210 = tt.expand_dims %209 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %211 = tt.broadcast %210 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:           %212 = arith.subf %206, %211 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %213 = math.exp %212 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %214 = arith.subf %arg18, %209 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %215 = math.exp %214 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %216 = arith.mulf %215, %arg19 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %217 = "tt.reduce"(%213) <{axis = 1 : i32}> ({
// CHECK-NEXT:           ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:             %226 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:             tt.reduce.return %226 : f32
// CHECK-NEXT:           }) {DataUse} : (tensor<32x128xf32>) -> tensor<32xf32>
// CHECK-NEXT:           %218 = arith.addf %216, %217 {DataUse} : tensor<32xf32>
// CHECK-NEXT:           %219 = tt.expand_dims %215 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:           %220 = tt.broadcast %219 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:           %221 = arith.mulf %220, %arg17 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:           %222 = arith.truncf %213 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
// CHECK-NEXT:           %extracted_slice_22 = tensor.extract_slice %222[%5, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xbf16> to tensor<16x128xbf16>
// CHECK-NEXT:           %223 = tensor.empty() : tensor<32x128xbf16>
// CHECK-NEXT:           %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %223[%5, 0] [16, 128] [1, 1] {cv_communication_slice} : tensor<16x128xbf16> into tensor<32x128xbf16>
// CHECK-NEXT:           %224 = tt.dot %inserted_slice_23, %207, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
// CHECK-NEXT:           %225 = arith.addf %224, %221 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
// CHECK-NEXT:           scf.yield %225, %209, %218 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         %184 = tt.expand_dims %183#2 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %185 = tt.broadcast %184 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %186 = arith.divf %183#0, %185 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %187 = math.log %183#2 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %188 = arith.addf %187, %183#1 {DataUse} : tensor<32xf32>
// CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %186[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xf32> to tensor<16x128xf32>
// CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %80[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128x!tt.ptr<f32>> to tensor<16x128x!tt.ptr<f32>>
// CHECK-NEXT:         %extracted_slice_18 = tensor.extract_slice %87[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xi1> to tensor<16x128xi1>
// CHECK-NEXT:         tt.store %extracted_slice_17, %extracted_slice_16, %extracted_slice_18 {tiled_op} : tensor<16x128x!tt.ptr<f32>>
// CHECK-NEXT:         %extracted_slice_19 = tensor.extract_slice %188[%7] [16] [1] {to_be_bubbled_slice} : tensor<32xf32> to tensor<16xf32>
// CHECK-NEXT:         %extracted_slice_20 = tensor.extract_slice %84[%7] [16] [1] {to_be_bubbled_slice} : tensor<32x!tt.ptr<f32>> to tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:         %extracted_slice_21 = tensor.extract_slice %86[%7] [16] [1] {to_be_bubbled_slice} : tensor<32xi1> to tensor<16xi1>
// CHECK-NEXT:         tt.store %extracted_slice_20, %extracted_slice_19, %extracted_slice_21 {tiled_op} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %41, %45 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_da_fwd_u(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32>, %arg6: i32, %arg7: f32, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x1xf32>
    %cst_0 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_2 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
    %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x32xf32>
    %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<32x32xf32>
    %cst_5 = arith.constant {DataUse} dense<1.000000e+06> : tensor<32x32xf32>
    %cst_6 = arith.constant {DataUse} dense<0> : tensor<32x32xi8>
    %cst_7 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32x32xf32>
    %c31_i32 = arith.constant 31 : i32
    %cst_8 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<32x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %cst_9 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<32xf32>
    %cst_10 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32xf32>
    %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
    %cst_12 = arith.constant {MetaUse} dense<640> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_13 = arith.constant {MetaUse} dense<64> : tensor<32x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.make_range {DataUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %4 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %5 = tt.expand_dims %3 {DataUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %6 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %7 = tt.expand_dims %3 {DataUse, axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %8 = arith.muli %4, %cst_13 {MetaUse} : tensor<32x1xi32>
    %9 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %10 = tt.addptr %9, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %11 = tt.broadcast %10 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %12 = tt.broadcast %6 {MetaUse} : tensor<1x32xi32> -> tensor<32x32xi32>
    %13 = tt.addptr %11, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %14 = tt.bitcast %13 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %15 = tt.load %14 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %16 = tt.splat %arg9 {MetaUse} : !tt.ptr<i1> -> tensor<32x1x!tt.ptr<i1>>
    %17 = tt.addptr %16, %8 {MetaUse} : tensor<32x1x!tt.ptr<i1>>, tensor<32x1xi32>
    %18 = tt.broadcast %17 {MetaUse} : tensor<32x1x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i1>>
    %19 = tt.addptr %18, %12 {MetaUse} : tensor<32x32x!tt.ptr<i1>>, tensor<32x32xi32>
    %20 = tt.bitcast %19 {MetaUse} : tensor<32x32x!tt.ptr<i1>> -> tensor<32x32x!tt.ptr<i8>>
    %21 = tt.load %20 {DataUse, was_bool_to_int8 = true} : tensor<32x32x!tt.ptr<i8>>
    %22 = arith.remsi %0, %1 : i32
    %23 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %24 = tt.expand_dims %23 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %25 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
    %26 = tt.splat %arg7 {DataUse} : f32 -> tensor<32x32xf32>
    %27 = arith.cmpi ne, %15, %cst_6 {DataUse} : tensor<32x32xi8>
    %28 = arith.cmpi ne, %21, %cst_6 {DataUse} : tensor<32x32xi8>
    %29 = tt.broadcast %24 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %30 = tt.splat %arg7 {DataUse} : f32 -> tensor<32x128xf32>
    %31:2 = scf.for %arg12 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
      %32 = tt.addptr %arg5, %arg12 : !tt.ptr<i32>, i32
      %33 = tt.load %32 : !tt.ptr<i32>
      %34 = arith.subi %33, %arg13 : i32
      %35 = arith.addi %34, %c31_i32 : i32
      %36 = arith.divsi %35, %c32_i32 : i32
      %37 = arith.addi %arg14, %36 : i32
      %38 = arith.muli %arg14, %c5_i32 : i32
      %39 = arith.remsi %38, %1 : i32
      %40 = arith.subi %22, %39 : i32
      %41 = arith.addi %40, %1 : i32
      %42 = arith.remsi %41, %1 : i32
      %43 = arith.addi %38, %42 : i32
      %44 = arith.muli %37, %c5_i32 : i32
      %45 = tt.splat %33 {MetaUse} : i32 -> tensor<32x1xi32>
      %46 = tt.splat %33 {DataUse} : i32 -> tensor<32x1xi32>
      %47 = tt.splat %33 {MetaUse} : i32 -> tensor<32xi32>
      %48 = tt.splat %33 {DataUse} : i32 -> tensor<1x32xi32>
      %49 = arith.addi %arg10, %arg13 : i32
      %50 = arith.addi %arg10, %33 : i32
      %51 = tt.splat %50 {MetaUse} : i32 -> tensor<32x1xi32>
      %52 = tt.splat %50 {MetaUse} : i32 -> tensor<128x1xi32>
      scf.for %arg15 = %43 to %44 step %1  : i32 {
        %53 = arith.divsi %arg15, %c5_i32 : i32
        %54 = arith.subi %53, %arg14 : i32
        %55 = arith.remsi %arg15, %c5_i32 : i32
        %56 = arith.muli %55, %c128_i32 : i32
        %57 = tt.addptr %arg0, %56 : !tt.ptr<bf16>, i32
        %58 = arith.muli %54, %c32_i32 : i32
        %59 = arith.addi %arg13, %58 : i32
        %60 = tt.splat %59 {MetaUse} : i32 -> tensor<32xi32>
        %61 = arith.addi %60, %2 {MetaUse} : tensor<32xi32>
        %62 = tt.expand_dims %61 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %63 = arith.muli %62, %cst_12 {MetaUse} : tensor<32x1xi32>
        %64 = tt.splat %57 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %63 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %66 = tt.broadcast %65 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %67 = tt.addptr %66, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %68 = tt.addptr %arg3, %56 : !tt.ptr<f32>, i32
        %69 = tt.splat %68 {MetaUse} : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
        %70 = tt.addptr %69, %63 {MetaUse} : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
        %71 = tt.broadcast %70 {MetaUse} : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
        %72 = tt.addptr %71, %25 {MetaUse} : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
        %73 = arith.muli %55, %arg11 : i32
        %74 = tt.addptr %arg4, %73 : !tt.ptr<f32>, i32
        %75 = tt.splat %74 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
        %76 = tt.addptr %75, %61 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %77 = arith.cmpi slt, %62, %45 {MetaUse} : tensor<32x1xi32>
        %78 = arith.cmpi slt, %61, %47 {MetaUse} : tensor<32xi32>
        %79 = tt.broadcast %77 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
        %80 = tt.load %67, %79, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %81 = tt.splat %59 {DataUse} : i32 -> tensor<32x1xi32>
        %82 = arith.addi %81, %5 {DataUse} : tensor<32x1xi32>
        %83 = arith.cmpi slt, %82, %46 {DataUse} : tensor<32x1xi32>
        %84 = tt.splat %59 {DataUse} : i32 -> tensor<1x32xi32>
        %85 = arith.addi %84, %7 {DataUse} : tensor<1x32xi32>
        %86 = arith.cmpi slt, %85, %48 {DataUse} : tensor<1x32xi32>
        %87 = tt.broadcast %83 {DataUse} : tensor<32x1xi1> -> tensor<32x32xi1>
        %88 = tt.broadcast %86 {DataUse} : tensor<1x32xi1> -> tensor<32x32xi1>
        %89 = arith.andi %87, %88 {DataUse} : tensor<32x32xi1>
        %90 = arith.divsi %55, %c5_i32 : i32
        %91 = arith.muli %90, %c128_i32 : i32
        %92 = tt.addptr %arg1, %91 : !tt.ptr<bf16>, i32
        %93 = arith.muli %62, %cst_2 {MetaUse} : tensor<32x1xi32>
        %94 = tt.splat %92 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %95 = tt.addptr %94, %93 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %96 = tt.broadcast %95 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %97 = tt.addptr %96, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %98 = tt.addptr %arg2, %91 : !tt.ptr<bf16>, i32
        %99 = tt.splat %98 {MetaUse} : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
        %100 = tt.addptr %99, %93 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %101 = tt.broadcast %100 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %102 = tt.addptr %101, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %103 = tt.load %97, %79, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %104 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %105 = tt.dot %80, %104, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %106 = arith.mulf %105, %26 {DataUse} : tensor<32x32xf32>
        %107 = tt.load %102, %79, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %108 = arith.uitofp %89 {DataUse} : tensor<32x32xi1> to tensor<32x32xf32>
        %109 = arith.subf %108, %cst_4 {DataUse} : tensor<32x32xf32>
        %110 = arith.mulf %109, %cst_5 {DataUse} : tensor<32x32xf32>
        %111 = arith.addf %106, %110 {DataUse} : tensor<32x32xf32>
        %112 = arith.select %27, %111, %cst_7 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
        %113 = "tt.reduce"(%112) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %179 = arith.maxnumf %arg16, %arg17 : f32
          tt.reduce.return %179 : f32
        }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
        %114 = arith.maxnumf %113, %cst_9 {DataUse} : tensor<32xf32>
        %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %116 = tt.broadcast %115 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
        %117 = arith.subf %112, %116 {DataUse} : tensor<32x32xf32>
        %118 = math.exp %117 {DataUse} : tensor<32x32xf32>
        %119 = arith.subf %cst_9, %114 {DataUse} : tensor<32xf32>
        %120 = math.exp %119 {DataUse} : tensor<32xf32>
        %121 = arith.mulf %120, %cst_10 {DataUse} : tensor<32xf32>
        %122 = "tt.reduce"(%118) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %179 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %179 : f32
        }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
        %123 = arith.addf %121, %122 {DataUse} : tensor<32xf32>
        %124 = tt.expand_dims %120 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %125 = arith.mulf %124, %cst {DataUse} : tensor<32x1xf32>
        %126 = tt.broadcast %125 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
        %127 = arith.truncf %118 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
        %128 = tt.dot %127, %107, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %129 = arith.addf %128, %126 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
        %130 = arith.addi %49, %58 : i32
        %131 = tt.splat %130 {MetaUse} : i32 -> tensor<32xi32>
        %132 = arith.addi %131, %2 {MetaUse} : tensor<32xi32>
        %133 = tt.expand_dims %132 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %134 = arith.muli %133, %cst_2 {MetaUse} : tensor<32x1xi32>
        %135 = tt.addptr %94, %134 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %136 = tt.broadcast %135 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %137 = tt.addptr %136, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %138 = tt.addptr %99, %134 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
        %139 = tt.broadcast %138 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
        %140 = tt.addptr %139, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
        %141 = arith.cmpi slt, %133, %51 {MetaUse} : tensor<32x1xi32>
        %142 = tt.broadcast %141 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
        %143 = tt.load %137, %142, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %144 = tt.trans %143 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
        %145 = tt.dot %80, %144, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
        %146 = arith.mulf %145, %26 {DataUse} : tensor<32x32xf32>
        %147 = tt.load %140, %142, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
        %148 = arith.addf %146, %110 {DataUse} : tensor<32x32xf32>
        %149 = arith.select %28, %148, %cst_7 {DataUse} : tensor<32x32xi1>, tensor<32x32xf32>
        %150 = "tt.reduce"(%149) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %179 = arith.maxnumf %arg16, %arg17 : f32
          tt.reduce.return %179 : f32
        }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
        %151 = arith.maxnumf %114, %150 {DataUse} : tensor<32xf32>
        %152 = tt.expand_dims %151 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %153 = tt.broadcast %152 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
        %154 = arith.subf %149, %153 {DataUse} : tensor<32x32xf32>
        %155 = math.exp %154 {DataUse} : tensor<32x32xf32>
        %156 = arith.subf %114, %151 {DataUse} : tensor<32xf32>
        %157 = math.exp %156 {DataUse} : tensor<32xf32>
        %158 = arith.mulf %157, %123 {DataUse} : tensor<32xf32>
        %159 = "tt.reduce"(%155) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %179 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %179 : f32
        }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
        %160 = arith.addf %158, %159 {DataUse} : tensor<32xf32>
        %161 = tt.expand_dims %157 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %162 = tt.broadcast %161 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
        %163 = arith.mulf %162, %129 {DataUse} : tensor<32x128xf32>
        %164 = arith.truncf %155 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
        %165 = tt.dot %164, %147, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
        %166 = arith.addf %165, %163 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
        %167 = arith.divsi %58, %c128_i32 : i32
        %168 = arith.muli %167, %c128_i32 : i32
        %169 = arith.divsi %168, %c32_i32 : i32
        %170:3 = scf.for %arg16 = %169 to %54 step %c1_i32 iter_args(%arg17 = %166, %arg18 = %151, %arg19 = %160) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
          %179 = arith.muli %arg16, %c32_i32 : i32
          %180 = arith.addi %49, %179 : i32
          %181 = tt.splat %180 {MetaUse} : i32 -> tensor<32xi32>
          %182 = arith.addi %181, %2 {MetaUse} : tensor<32xi32>
          %183 = tt.expand_dims %182 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
          %184 = arith.muli %183, %cst_2 {MetaUse} : tensor<32x1xi32>
          %185 = tt.addptr %94, %184 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %186 = tt.broadcast %185 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %187 = tt.addptr %186, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %188 = tt.addptr %99, %184 {MetaUse} : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
          %189 = tt.broadcast %188 {MetaUse} : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x128x!tt.ptr<bf16>>
          %190 = tt.addptr %189, %25 {MetaUse} : tensor<32x128x!tt.ptr<bf16>>, tensor<32x128xi32>
          %191 = arith.cmpi slt, %183, %51 {MetaUse} : tensor<32x1xi32>
          %192 = tt.broadcast %191 {MetaUse} : tensor<32x1xi1> -> tensor<32x128xi1>
          %193 = tt.load %187, %192, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
          %194 = tt.trans %193 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xbf16> -> tensor<128x32xbf16>
          %195 = tt.dot %80, %194, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x32xbf16> -> tensor<32x32xf32>
          %196 = arith.mulf %195, %26 {DataUse} : tensor<32x32xf32>
          %197 = tt.load %190, %192, %cst_8 {DataUse} : tensor<32x128x!tt.ptr<bf16>>
          %198 = "tt.reduce"(%196) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %215 = arith.maxnumf %arg20, %arg21 : f32
            tt.reduce.return %215 : f32
          }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
          %199 = arith.maxnumf %arg18, %198 {DataUse} : tensor<32xf32>
          %200 = tt.expand_dims %199 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %201 = tt.broadcast %200 {DataUse} : tensor<32x1xf32> -> tensor<32x32xf32>
          %202 = arith.subf %196, %201 {DataUse} : tensor<32x32xf32>
          %203 = math.exp %202 {DataUse} : tensor<32x32xf32>
          %204 = arith.subf %arg18, %199 {DataUse} : tensor<32xf32>
          %205 = math.exp %204 {DataUse} : tensor<32xf32>
          %206 = arith.mulf %205, %arg19 {DataUse} : tensor<32xf32>
          %207 = "tt.reduce"(%203) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %215 = arith.addf %arg20, %arg21 : f32
            tt.reduce.return %215 : f32
          }) {DataUse} : (tensor<32x32xf32>) -> tensor<32xf32>
          %208 = arith.addf %206, %207 {DataUse} : tensor<32xf32>
          %209 = tt.expand_dims %205 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %210 = tt.broadcast %209 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
          %211 = arith.mulf %210, %arg17 {DataUse} : tensor<32x128xf32>
          %212 = arith.truncf %203 {DataUse} : tensor<32x32xf32> to tensor<32x32xbf16>
          %213 = tt.dot %212, %197, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x32xbf16> * tensor<32x128xbf16> -> tensor<32x128xf32>
          %214 = arith.addf %213, %211 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
          scf.yield %214, %199, %208 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
        } {DataUse}
        %171 = tt.splat %92 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %172 = tt.splat %98 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %173:3 = scf.for %arg16 = %c0_i32 to %167 step %c1_i32 iter_args(%arg17 = %170#0, %arg18 = %170#1, %arg19 = %170#2) -> (tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>)  : i32 {
          %179 = arith.muli %arg16, %c128_i32 : i32
          %180 = arith.addi %49, %179 : i32
          %181 = tt.splat %180 {MetaUse} : i32 -> tensor<128xi32>
          %182 = arith.addi %181, %23 {MetaUse} : tensor<128xi32>
          %183 = tt.expand_dims %182 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %184 = arith.muli %183, %cst_0 {MetaUse} : tensor<128x1xi32>
          %185 = tt.addptr %171, %184 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %186 = tt.broadcast %185 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %187 = tt.addptr %186, %29 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %188 = tt.addptr %172, %184 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %189 = tt.broadcast %188 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %190 = tt.addptr %189, %29 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %191 = arith.cmpi slt, %183, %52 {MetaUse} : tensor<128x1xi32>
          %192 = tt.broadcast %191 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
          %193 = tt.load %187, %192, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %194 = tt.trans %193 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
          %195 = tt.dot %80, %194, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %196 = arith.mulf %195, %30 {DataUse} : tensor<32x128xf32>
          %197 = tt.load %190, %192, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %198 = "tt.reduce"(%196) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %215 = arith.maxnumf %arg20, %arg21 : f32
            tt.reduce.return %215 : f32
          }) {DataUse} : (tensor<32x128xf32>) -> tensor<32xf32>
          %199 = arith.maxnumf %arg18, %198 {DataUse} : tensor<32xf32>
          %200 = tt.expand_dims %199 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %201 = tt.broadcast %200 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
          %202 = arith.subf %196, %201 {DataUse} : tensor<32x128xf32>
          %203 = math.exp %202 {DataUse} : tensor<32x128xf32>
          %204 = arith.subf %arg18, %199 {DataUse} : tensor<32xf32>
          %205 = math.exp %204 {DataUse} : tensor<32xf32>
          %206 = arith.mulf %205, %arg19 {DataUse} : tensor<32xf32>
          %207 = "tt.reduce"(%203) <{axis = 1 : i32}> ({
          ^bb0(%arg20: f32, %arg21: f32):
            %215 = arith.addf %arg20, %arg21 : f32
            tt.reduce.return %215 : f32
          }) {DataUse} : (tensor<32x128xf32>) -> tensor<32xf32>
          %208 = arith.addf %206, %207 {DataUse} : tensor<32xf32>
          %209 = tt.expand_dims %205 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
          %210 = tt.broadcast %209 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
          %211 = arith.mulf %210, %arg17 {DataUse} : tensor<32x128xf32>
          %212 = arith.truncf %203 {DataUse} : tensor<32x128xf32> to tensor<32x128xbf16>
          %213 = tt.dot %212, %197, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xbf16> * tensor<128x128xbf16> -> tensor<32x128xf32>
          %214 = arith.addf %213, %211 {DataUse, triton_cv12.add_from_dot} : tensor<32x128xf32>
          scf.yield %214, %199, %208 : tensor<32x128xf32>, tensor<32xf32>, tensor<32xf32>
        } {DataUse}
        %174 = tt.expand_dims %173#2 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %175 = tt.broadcast %174 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
        %176 = arith.divf %173#0, %175 {DataUse} : tensor<32x128xf32>
        %177 = math.log %173#2 {DataUse} : tensor<32xf32>
        %178 = arith.addf %177, %173#1 {DataUse} : tensor<32xf32>
        tt.store %72, %176, %79 : tensor<32x128x!tt.ptr<f32>>
        tt.store %76, %178, %78 : tensor<32x!tt.ptr<f32>>
      }
      scf.yield %33, %37 : i32, i32
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/dv_local/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_bwd_kernel_dv_local(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: f32, %arg6: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xbf16>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = tt.get_program_id x : i32
// CHECK-NEXT:     %4 = tt.get_program_id y : i32
// CHECK-NEXT:     %5 = arith.muli %4, %arg6 : i32
// CHECK-NEXT:     %6 = arith.muli %3, %c64_i32 : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %8 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %9 = tt.splat %6 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %10 = tt.splat %6 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %11 = arith.addi %9, %7 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %12 = arith.addi %10, %8 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %13 = tt.splat %arg6 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %14 = tt.splat %arg6 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %15 = arith.cmpi slt, %11, %13 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:     %16 = arith.cmpi slt, %12, %14 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %17 = arith.muli %5, %c32_i32 : i32
// CHECK-NEXT:     %18 = arith.muli %4, %c32_i32 : i32
// CHECK-NEXT:     %19 = arith.extsi %arg6 : i32 to i64
// CHECK-NEXT:     %20 = tt.splat %arg5 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     %21 = tt.expand_dims %12 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %22 = tt.expand_dims %12 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %23 = tt.broadcast %21 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %24 = tt.broadcast %22 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %25 = arith.cmpi sle, %23, %24 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %26 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %27 = tt.expand_dims %16 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %28 = tt.broadcast %26 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %29 = tt.broadcast %27 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %30 = arith.andi %28, %29 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %31 = arith.andi %25, %30 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
// CHECK-NEXT:       %32 = arith.addi %17, %arg7 : i32
// CHECK-NEXT:       %33 = arith.muli %32, %c128_i32 : i32
// CHECK-NEXT:       %34 = tt.addptr %arg0, %33 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %35 = tt.addptr %arg1, %33 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %36 = tt.addptr %arg3, %33 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %37 = tt.addptr %arg4, %33 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %38 = arith.addi %18, %arg7 : i32
// CHECK-NEXT:       %39 = arith.muli %38, %arg6 : i32
// CHECK-NEXT:       %40 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %41 = tt.splat %40 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %42 = tt.addptr %41, %11 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
// CHECK-NEXT:       %43 = tt.load %42, %15, %cst_1 {DataUse} : tensor<64x!tt.ptr<bf16>>
// CHECK-NEXT:       %44 = arith.extf %43 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:       %45 = tt.make_tensor_ptr %35, [%19, %c128_i64], [%c4096_i64, %c1_i64], [%6, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %46 = tt.make_tensor_ptr %34, [%c128_i64, %19], [%c1_i64, %c4096_i64], [%c0_i32, %6] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
// CHECK-NEXT:       %47 = tt.load %45 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %48 = tt.load %46 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-NEXT:       %49 = tt.dot %47, %48, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %50 = arith.mulf %49, %20 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %51 = arith.addf %50, %cst {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %52 = tt.expand_dims %44 {DataUse, axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
// CHECK-NEXT:       %53 = tt.expand_dims %44 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %54 = tt.broadcast %52 {DataUse} : tensor<1x64xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %55 = tt.broadcast %53 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %56 = arith.subf %54, %55 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %57 = math.exp %56 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %58 = arith.mulf %51, %57 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %59 = arith.select %31, %58, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:       %60 = arith.truncf %59 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %61 = tt.make_tensor_ptr %36, [%19, %c128_i64], [%c4096_i64, %c1_i64], [%6, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %62 = tt.make_tensor_ptr %37, [%19, %c128_i64], [%c4096_i64, %c1_i64], [%6, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %63 = tt.load %61 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %60[%2, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       %64 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:       %inserted_slice = tensor.insert_slice %extracted_slice into %64[%2, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:       %65 = tt.dot %inserted_slice, %63, %cst_0 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:       %66 = arith.truncf %65 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %62, %66 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_bwd_kernel_dv_local(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: f32, %arg6: i32) attributes {noinline = false} {
    %cst = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
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
    %4 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %5 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.splat %3 {MetaUse} : i32 -> tensor<64xi32>
    %7 = tt.splat %3 {DataUse} : i32 -> tensor<64xi32>
    %8 = arith.addi %6, %4 {MetaUse} : tensor<64xi32>
    %9 = arith.addi %7, %5 {DataUse} : tensor<64xi32>
    %10 = tt.splat %arg6 {MetaUse} : i32 -> tensor<64xi32>
    %11 = tt.splat %arg6 {DataUse} : i32 -> tensor<64xi32>
    %12 = arith.cmpi slt, %8, %10 {MetaUse} : tensor<64xi32>
    %13 = arith.cmpi slt, %9, %11 {DataUse} : tensor<64xi32>
    %14 = arith.muli %2, %c32_i32 : i32
    %15 = arith.muli %1, %c32_i32 : i32
    %16 = arith.extsi %arg6 : i32 to i64
    %17 = tt.splat %arg5 {DataUse} : f32 -> tensor<64x64xf32>
    %18 = tt.expand_dims %9 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %19 = tt.expand_dims %9 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %20 = tt.broadcast %18 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
    %21 = tt.broadcast %19 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %22 = arith.cmpi sle, %20, %21 {DataUse} : tensor<64x64xi32>
    %23 = tt.expand_dims %13 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %24 = tt.expand_dims %13 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %25 = tt.broadcast %23 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
    %26 = tt.broadcast %24 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
    %27 = arith.andi %25, %26 {DataUse} : tensor<64x64xi1>
    %28 = arith.andi %22, %27 {DataUse} : tensor<64x64xi1>
    scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %29 = arith.addi %14, %arg7 : i32
      %30 = arith.muli %29, %c128_i32 : i32
      %31 = tt.addptr %arg0, %30 : !tt.ptr<bf16>, i32
      %32 = tt.addptr %arg1, %30 : !tt.ptr<bf16>, i32
      %33 = tt.addptr %arg3, %30 : !tt.ptr<bf16>, i32
      %34 = tt.addptr %arg4, %30 : !tt.ptr<bf16>, i32
      %35 = arith.addi %15, %arg7 : i32
      %36 = arith.muli %35, %arg6 : i32
      %37 = tt.addptr %arg2, %36 : !tt.ptr<bf16>, i32
      %38 = tt.splat %37 {MetaUse} : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
      %39 = tt.addptr %38, %8 {MetaUse} : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
      %40 = tt.load %39, %12, %cst {DataUse} : tensor<64x!tt.ptr<bf16>>
      %41 = arith.extf %40 {DataUse} : tensor<64xbf16> to tensor<64xf32>
      %42 = tt.make_tensor_ptr %32, [%16, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %43 = tt.make_tensor_ptr %31, [%c128_i64, %16], [%c1_i64, %c4096_i64], [%c0_i32, %3] {order = array<i32: 0, 1>} : <tensor<128x64xbf16>>
      %44 = tt.load %42 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
      %45 = tt.load %43 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
      %46 = tt.dot %44, %45, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
      %47 = arith.mulf %46, %17 {DataUse} : tensor<64x64xf32>
      %48 = arith.addf %47, %cst_1 {DataUse} : tensor<64x64xf32>
      %49 = tt.expand_dims %41 {DataUse, axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
      %50 = tt.expand_dims %41 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %51 = tt.broadcast %49 {DataUse} : tensor<1x64xf32> -> tensor<64x64xf32>
      %52 = tt.broadcast %50 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
      %53 = arith.subf %51, %52 {DataUse} : tensor<64x64xf32>
      %54 = math.exp %53 {DataUse} : tensor<64x64xf32>
      %55 = arith.mulf %48, %54 {DataUse} : tensor<64x64xf32>
      %56 = arith.select %28, %55, %cst_1 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
      %57 = arith.truncf %56 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
      %58 = tt.make_tensor_ptr %33, [%16, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %59 = tt.make_tensor_ptr %34, [%16, %c128_i64], [%c4096_i64, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
      %60 = tt.load %58 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
      %61 = tt.dot %57, %60, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
      %62 = arith.truncf %61 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %59, %62 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_bwd/sep_md_false.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c4194304_i32 = arith.constant 4194304 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<64> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<true> : tensor<128x64xi1>
// CHECK-NEXT:     %c65536_i32 = arith.constant 65536 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = tt.get_program_id x : i32
// CHECK-NEXT:     %6 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %8 = tt.expand_dims %6 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %9 = tt.broadcast %8 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:     %10 = tt.splat %arg9 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:     scf.for %arg10 = %5 to %c65536_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %11 = arith.divsi %arg10, %c64_i32 : i32
// CHECK-NEXT:       %12 = arith.muli %11, %c64_i32 : i32
// CHECK-NEXT:       %13 = arith.subi %arg10, %12 : i32
// CHECK-NEXT:       %14 = arith.muli %11, %c8192_i32 : i32
// CHECK-NEXT:       %15 = arith.extsi %14 : i32 to i64
// CHECK-NEXT:       %16 = arith.remsi %11, %c8_i32 : i32
// CHECK-NEXT:       %17 = arith.muli %16, %c524288_i32 : i32
// CHECK-NEXT:       %18 = arith.divsi %11, %c8_i32 : i32
// CHECK-NEXT:       %19 = arith.muli %18, %c4194304_i32 : i32
// CHECK-NEXT:       %20 = arith.addi %17, %19 : i32
// CHECK-NEXT:       %21 = arith.extsi %20 : i32 to i64
// CHECK-NEXT:       %22 = tt.addptr %arg0, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %23 = tt.addptr %arg1, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %24 = tt.addptr %arg2, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %25 = tt.addptr %arg3, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %26 = tt.addptr %arg4, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %27 = tt.addptr %arg5, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %28 = tt.addptr %arg6, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %29 = tt.addptr %arg7, %15 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %30 = tt.addptr %arg8, %15 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %31 = arith.muli %13, %c128_i32 : i32
// CHECK-NEXT:       %32 = tt.splat %31 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %33 = arith.addi %32, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %34 = tt.expand_dims %33 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %35 = arith.muli %34, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %36 = tt.splat %23 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %37 = tt.addptr %36, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %38 = tt.broadcast %37 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %39 = tt.addptr %38, %9 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %40 = tt.load %39 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %41 = tt.splat %24 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %42 = tt.addptr %41, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %43 = tt.broadcast %42 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %44 = tt.addptr %43, %9 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %45 = tt.load %44 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %46 = tt.splat %22 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %47 = tt.splat %25 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %48 = tt.splat %29 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %49 = tt.splat %30 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %50 = tt.trans %40 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %51 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %52 = tt.splat %26 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %53:2 = scf.for %arg11 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg12 = %cst_1, %arg13 = %cst_1) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
// CHECK-NEXT:         %64 = arith.muli %arg11, %c128_i32 : i32
// CHECK-NEXT:         %65 = tt.splat %64 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %66 = arith.addi %65, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %67 = tt.expand_dims %66 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %68 = arith.muli %67, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %69 = tt.addptr %46, %68 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %70 = tt.broadcast %69 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %71 = tt.addptr %70, %9 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %72 = tt.load %71 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %73 = tt.addptr %47, %68 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %74 = tt.broadcast %73 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %75 = tt.addptr %74, %9 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %76 = tt.load %75 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %77 = tt.addptr %48, %66 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %78 = tt.load %77 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %79 = tt.addptr %49, %66 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %80 = tt.load %79 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %81 = tt.dot %72, %50, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %82 = arith.mulf %81, %10 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %83 = tt.expand_dims %78 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %84 = tt.broadcast %83 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %85 = arith.subf %82, %84 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %86 = math.exp %85 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %87 = arith.truncf %86 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %87[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:         %88 = tensor.empty() : tensor<128x128xf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice into %88[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf16> into tensor<128x128xf16>
// CHECK-NEXT:         %89 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %90 = tt.dot %89, %76, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %91 = tt.dot %76, %51, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %92 = tt.expand_dims %80 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %93 = tt.broadcast %92 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %94 = arith.subf %91, %93 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %95 = arith.extf %87 {DataUse} : tensor<128x128xf16> to tensor<128x128xf32>
// CHECK-NEXT:         %96 = arith.mulf %95, %94 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %97 = arith.mulf %96, %10 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %98 = arith.truncf %97 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %extracted_slice_3 = tensor.extract_slice %98[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:         %99 = tensor.empty() : tensor<128x128xf16>
// CHECK-NEXT:         %inserted_slice_4 = tensor.insert_slice %extracted_slice_3 into %99[%3, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf16> into tensor<128x128xf16>
// CHECK-NEXT:         %100 = tt.trans %inserted_slice_4 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %101 = tt.dot %100, %72, %arg12 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %extracted_slice_5 = tensor.extract_slice %98[%4, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:         %102 = tensor.empty() : tensor<128x128xf16>
// CHECK-NEXT:         %inserted_slice_6 = tensor.insert_slice %extracted_slice_5 into %102[%4, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf16> into tensor<128x128xf16>
// CHECK-NEXT:         %103 = tt.dot %inserted_slice_6, %40, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %104 = tt.addptr %52, %68 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %105 = tt.broadcast %104 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %106 = tt.addptr %105, %9 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %107 = arith.truncf %103 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:         %108 = tt.atomic_rmw fadd, acq_rel, gpu, %106, %107, %cst_0 {Undefined} : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
// CHECK-NEXT:         scf.yield %101, %90 : tensor<128x64xf32>, tensor<128x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %54 = tt.splat %27 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %55 = tt.addptr %54, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %56 = tt.broadcast %55 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %57 = tt.addptr %56, %9 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %58 = arith.truncf %53#0 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %57, %58 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %59 = tt.splat %28 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %60 = tt.addptr %59, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %61 = tt.broadcast %60 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %62 = tt.addptr %61, %9 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %63 = arith.truncf %53#1 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %62, %63 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cst_1 = arith.constant {MetaUse} dense<true> : tensor<128x64xi1>
    %cst_2 = arith.constant {MetaUse} dense<64> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c4194304_i32 = arith.constant 4194304 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %4 = tt.broadcast %3 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
    %5 = tt.splat %arg9 {DataUse} : f32 -> tensor<128x128xf32>
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
      %27 = tt.splat %26 {MetaUse} : i32 -> tensor<128xi32>
      %28 = arith.addi %27, %2 {MetaUse} : tensor<128xi32>
      %29 = tt.expand_dims %28 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %30 = arith.muli %29, %cst_2 {MetaUse} : tensor<128x1xi32>
      %31 = tt.splat %18 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %32 = tt.addptr %31, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %33 = tt.broadcast %32 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %34 = tt.addptr %33, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %35 = tt.load %34 {DataUse} : tensor<128x64x!tt.ptr<f16>>
      %36 = tt.splat %19 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %37 = tt.addptr %36, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %38 = tt.broadcast %37 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %39 = tt.addptr %38, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %40 = tt.load %39 {DataUse} : tensor<128x64x!tt.ptr<f16>>
      %41 = tt.splat %17 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.splat %20 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %43 = tt.splat %24 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %44 = tt.splat %25 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %45 = tt.trans %35 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %46 = tt.trans %40 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %47 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %48:2 = scf.for %arg11 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg12 = %cst_0, %arg13 = %cst_0) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
        %59 = arith.muli %arg11, %c128_i32 : i32
        %60 = tt.splat %59 {MetaUse} : i32 -> tensor<128xi32>
        %61 = arith.addi %60, %2 {MetaUse} : tensor<128xi32>
        %62 = tt.expand_dims %61 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %63 = arith.muli %62, %cst_2 {MetaUse} : tensor<128x1xi32>
        %64 = tt.addptr %41, %63 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %65 = tt.broadcast %64 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %66 = tt.addptr %65, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %67 = tt.load %66 {DataUse} : tensor<128x64x!tt.ptr<f16>>
        %68 = tt.addptr %42, %63 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %69 = tt.broadcast %68 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %70 = tt.addptr %69, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %71 = tt.load %70 {DataUse} : tensor<128x64x!tt.ptr<f16>>
        %72 = tt.addptr %43, %61 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %73 = tt.load %72 {DataUse} : tensor<128x!tt.ptr<f32>>
        %74 = tt.addptr %44, %61 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %75 = tt.load %74 {DataUse} : tensor<128x!tt.ptr<f32>>
        %76 = tt.dot %67, %45, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %77 = arith.mulf %76, %5 {DataUse} : tensor<128x128xf32>
        %78 = tt.expand_dims %73 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %79 = tt.broadcast %78 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %80 = arith.subf %77, %79 {DataUse} : tensor<128x128xf32>
        %81 = math.exp %80 {DataUse} : tensor<128x128xf32>
        %82 = arith.truncf %81 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
        %83 = tt.trans %82 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %84 = tt.dot %83, %71, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %85 = tt.dot %71, %46, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %86 = tt.expand_dims %75 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %87 = tt.broadcast %86 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %88 = arith.subf %85, %87 {DataUse} : tensor<128x128xf32>
        %89 = arith.extf %82 {DataUse} : tensor<128x128xf16> to tensor<128x128xf32>
        %90 = arith.mulf %89, %88 {DataUse} : tensor<128x128xf32>
        %91 = arith.mulf %90, %5 {DataUse} : tensor<128x128xf32>
        %92 = arith.truncf %91 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
        %93 = tt.trans %92 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %94 = tt.dot %93, %67, %arg12 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %95 = tt.dot %92, %35, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %96 = tt.addptr %47, %63 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %97 = tt.broadcast %96 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %98 = tt.addptr %97, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %99 = arith.truncf %95 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
        %100 = tt.atomic_rmw fadd, acq_rel, gpu, %98, %99, %cst_1 {Undefined} : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
        scf.yield %94, %84 : tensor<128x64xf32>, tensor<128x64xf32>
      } {DataUse}
      %49 = tt.splat %22 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %50 = tt.addptr %49, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %51 = tt.broadcast %50 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %52 = tt.addptr %51, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %53 = arith.truncf %48#0 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %52, %53 : tensor<128x64x!tt.ptr<f16>>
      %54 = tt.splat %23 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %55 = tt.addptr %54, %30 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %56 = tt.broadcast %55 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %57 = tt.addptr %56, %4 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %58 = arith.truncf %48#1 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %57, %58 : tensor<128x64x!tt.ptr<f16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_bwd/sep_md_true.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c4194304_i32 = arith.constant 4194304 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<64> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<64> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<true> : tensor<64x64xi1>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<true> : tensor<128x64xi1>
// CHECK-NEXT:     %c65536_i32 = arith.constant 65536 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c8064_i32 = arith.constant 8064 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map1()[%1]
// CHECK-NEXT:     %6 = affine.apply #map1()[%1]
// CHECK-NEXT:     %7 = affine.apply #map1()[%1]
// CHECK-NEXT:     %8 = tt.get_program_id x : i32
// CHECK-NEXT:     %9 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %10 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %11 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %12 = tt.make_range {DataUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %13 = tt.expand_dims %9 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %14 = tt.broadcast %13 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:     %15 = tt.broadcast %13 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %16 = tt.splat %arg9 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     %17 = tt.splat %arg9 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:     scf.for %arg10 = %8 to %c65536_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %18 = arith.divsi %arg10, %c64_i32 : i32
// CHECK-NEXT:       %19 = arith.muli %18, %c64_i32 : i32
// CHECK-NEXT:       %20 = arith.subi %arg10, %19 : i32
// CHECK-NEXT:       %21 = arith.muli %18, %c8192_i32 : i32
// CHECK-NEXT:       %22 = arith.extsi %21 : i32 to i64
// CHECK-NEXT:       %23 = arith.remsi %18, %c8_i32 : i32
// CHECK-NEXT:       %24 = arith.muli %23, %c524288_i32 : i32
// CHECK-NEXT:       %25 = arith.divsi %18, %c8_i32 : i32
// CHECK-NEXT:       %26 = arith.muli %25, %c4194304_i32 : i32
// CHECK-NEXT:       %27 = arith.addi %24, %26 : i32
// CHECK-NEXT:       %28 = arith.extsi %27 : i32 to i64
// CHECK-NEXT:       %29 = tt.addptr %arg0, %28 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %30 = tt.addptr %arg1, %28 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %31 = tt.addptr %arg2, %28 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %32 = tt.addptr %arg3, %28 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %33 = tt.addptr %arg4, %28 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %34 = tt.addptr %arg5, %28 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %35 = tt.addptr %arg6, %28 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %36 = tt.addptr %arg7, %22 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %37 = tt.addptr %arg8, %22 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %38 = arith.muli %20, %c128_i32 : i32
// CHECK-NEXT:       %39 = tt.splat %38 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %40 = tt.splat %38 {DataUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %41 = arith.addi %39, %11 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %42 = arith.addi %40, %12 {DataUse} : tensor<128xi32>
// CHECK-NEXT:       %43 = tt.expand_dims %41 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %44 = arith.muli %43, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %45 = tt.splat %30 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %46 = tt.addptr %45, %44 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %47 = tt.broadcast %46 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %48 = tt.addptr %47, %14 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %49 = tt.load %48 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %50 = tt.splat %31 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %51 = tt.addptr %50, %44 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %52 = tt.broadcast %51 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %53 = tt.addptr %52, %14 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %54 = tt.load %53 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %55 = tt.splat %29 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %56 = tt.splat %32 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %57 = tt.splat %36 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %58 = tt.splat %37 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %59 = tt.trans %49 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %60 = tt.expand_dims %42 {DataUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:       %61 = tt.broadcast %60 {DataUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:       %62 = tt.trans %54 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %63 = tt.splat %33 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %64:2 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %cst_3, %arg13 = %cst_3) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
// CHECK-NEXT:         %86 = arith.muli %arg11, %c64_i32 : i32
// CHECK-NEXT:         %87 = arith.addi %38, %86 : i32
// CHECK-NEXT:         %88 = tt.splat %87 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %89 = tt.splat %87 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %90 = arith.addi %88, %9 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %91 = arith.addi %89, %10 {DataUse} : tensor<64xi32>
// CHECK-NEXT:         %92 = tt.expand_dims %90 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %93 = tt.expand_dims %91 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %94 = arith.muli %92, %cst_0 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %95 = tt.addptr %55, %94 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %96 = tt.broadcast %95 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %97 = tt.addptr %96, %15 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
// CHECK-NEXT:         %98 = tt.load %97 {DataUse} : tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %99 = tt.addptr %56, %94 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %101 = tt.addptr %100, %15 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
// CHECK-NEXT:         %102 = tt.load %101 {DataUse} : tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %103 = tt.addptr %57, %90 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %104 = tt.load %103 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %105 = tt.addptr %58, %90 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %106 = tt.load %105 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %107 = tt.dot %98, %59, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %108 = arith.mulf %107, %16 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %109 = tt.expand_dims %104 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %110 = tt.broadcast %109 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %111 = arith.subf %108, %110 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %112 = math.exp %111 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %113 = tt.broadcast %93 {DataUse} : tensor<64x1xi32> -> tensor<64x128xi32>
// CHECK-NEXT:         %114 = arith.cmpi sge, %113, %61 {DataUse} : tensor<64x128xi32>
// CHECK-NEXT:         %115 = arith.select %114, %112, %cst_5 {DataUse} : tensor<64x128xi1>, tensor<64x128xf32>
// CHECK-NEXT:         %116 = arith.truncf %115 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %116[%2, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf16> to tensor<32x128xf16>
// CHECK-NEXT:         %117 = tensor.empty() : tensor<64x128xf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice into %117[%2, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xf16> into tensor<64x128xf16>
// CHECK-NEXT:         %118 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
// CHECK-NEXT:         %119 = tt.dot %118, %102, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %120 = tt.dot %102, %62, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %121 = tt.expand_dims %106 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %122 = tt.broadcast %121 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %123 = arith.subf %120, %122 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %124 = arith.extf %116 {DataUse} : tensor<64x128xf16> to tensor<64x128xf32>
// CHECK-NEXT:         %125 = arith.mulf %124, %123 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %126 = arith.mulf %125, %16 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %127 = arith.truncf %126 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
// CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %127[%3, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf16> to tensor<32x128xf16>
// CHECK-NEXT:         %128 = tensor.empty() : tensor<64x128xf16>
// CHECK-NEXT:         %inserted_slice_8 = tensor.insert_slice %extracted_slice_7 into %128[%3, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xf16> into tensor<64x128xf16>
// CHECK-NEXT:         %129 = tt.trans %inserted_slice_8 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
// CHECK-NEXT:         %130 = tt.dot %129, %98, %arg12 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %extracted_slice_9 = tensor.extract_slice %127[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf16> to tensor<32x128xf16>
// CHECK-NEXT:         %131 = tensor.empty() : tensor<64x128xf16>
// CHECK-NEXT:         %inserted_slice_10 = tensor.insert_slice %extracted_slice_9 into %131[%4, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xf16> into tensor<64x128xf16>
// CHECK-NEXT:         %132 = tt.dot %inserted_slice_10, %49, %cst_4 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %133 = tt.addptr %63, %94 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %134 = tt.broadcast %133 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
// CHECK-NEXT:         %135 = tt.addptr %134, %15 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
// CHECK-NEXT:         %136 = arith.truncf %132 {DataUse} : tensor<64x64xf32> to tensor<64x64xf16>
// CHECK-NEXT:         %137 = tt.atomic_rmw fadd, acq_rel, gpu, %135, %136, %cst_1 {Undefined} : (tensor<64x64x!tt.ptr<f16>>, tensor<64x64xf16>, tensor<64x64xi1>) -> tensor<64x64xf16>
// CHECK-NEXT:         scf.yield %130, %119 : tensor<128x64xf32>, tensor<128x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %65 = arith.addi %38, %c128_i32 : i32
// CHECK-NEXT:       %66 = arith.subi %c8064_i32, %38 : i32
// CHECK-NEXT:       %67 = arith.divsi %66, %c128_i32 : i32
// CHECK-NEXT:       %68 = tt.splat %29 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %69 = tt.splat %32 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %70 = tt.splat %36 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %71 = tt.splat %37 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %72 = tt.trans %49 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %73 = tt.trans %54 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
// CHECK-NEXT:       %74 = tt.splat %33 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %75:2 = scf.for %arg11 = %c0_i32 to %67 step %c1_i32 iter_args(%arg12 = %64#0, %arg13 = %64#1) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
// CHECK-NEXT:         %86 = arith.muli %arg11, %c128_i32 : i32
// CHECK-NEXT:         %87 = arith.addi %65, %86 : i32
// CHECK-NEXT:         %88 = tt.splat %87 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %89 = arith.addi %88, %11 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %90 = tt.expand_dims %89 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %91 = arith.muli %90, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %92 = tt.addptr %68, %91 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %93 = tt.broadcast %92 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %94 = tt.addptr %93, %14 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %95 = tt.load %94 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %96 = tt.addptr %69, %91 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %97 = tt.broadcast %96 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %98 = tt.addptr %97, %14 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %99 = tt.load %98 {DataUse} : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %100 = tt.addptr %70, %89 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %101 = tt.load %100 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %102 = tt.addptr %71, %89 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:         %103 = tt.load %102 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %104 = tt.dot %95, %72, %cst_6 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %105 = arith.mulf %104, %17 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %106 = tt.expand_dims %101 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %107 = tt.broadcast %106 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %108 = arith.subf %105, %107 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %109 = math.exp %108 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %110 = arith.truncf %109 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %110[%5, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:         %111 = tensor.empty() : tensor<128x128xf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice into %111[%5, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf16> into tensor<128x128xf16>
// CHECK-NEXT:         %112 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %113 = tt.dot %112, %99, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %114 = tt.dot %99, %73, %cst_6 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %115 = tt.expand_dims %103 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %116 = tt.broadcast %115 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %117 = arith.subf %114, %116 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %118 = arith.extf %110 {DataUse} : tensor<128x128xf16> to tensor<128x128xf32>
// CHECK-NEXT:         %119 = arith.mulf %118, %117 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %120 = arith.mulf %119, %17 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %121 = arith.truncf %120 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %121[%6, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:         %122 = tensor.empty() : tensor<128x128xf16>
// CHECK-NEXT:         %inserted_slice_8 = tensor.insert_slice %extracted_slice_7 into %122[%6, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf16> into tensor<128x128xf16>
// CHECK-NEXT:         %123 = tt.trans %inserted_slice_8 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %124 = tt.dot %123, %95, %arg12 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %extracted_slice_9 = tensor.extract_slice %121[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:         %125 = tensor.empty() : tensor<128x128xf16>
// CHECK-NEXT:         %inserted_slice_10 = tensor.insert_slice %extracted_slice_9 into %125[%7, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf16> into tensor<128x128xf16>
// CHECK-NEXT:         %126 = tt.dot %inserted_slice_10, %49, %cst_3 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %127 = tt.addptr %74, %91 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:         %128 = tt.broadcast %127 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:         %129 = tt.addptr %128, %14 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:         %130 = arith.truncf %126 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:         %131 = tt.atomic_rmw fadd, acq_rel, gpu, %129, %130, %cst_2 {Undefined} : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
// CHECK-NEXT:         scf.yield %124, %113 : tensor<128x64xf32>, tensor<128x64xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %76 = tt.splat %34 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %77 = tt.addptr %76, %44 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %78 = tt.broadcast %77 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %79 = tt.addptr %78, %14 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %80 = arith.truncf %75#0 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %79, %80 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %81 = tt.splat %35 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %82 = tt.addptr %81, %44 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %83 = tt.broadcast %82 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:       %84 = tt.addptr %83, %14 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
// CHECK-NEXT:       %85 = arith.truncf %75#1 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
// CHECK-NEXT:       tt.store %84, %85 : tensor<128x64x!tt.ptr<f16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
    %c8064_i32 = arith.constant 8064 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cst_3 = arith.constant {MetaUse} dense<true> : tensor<128x64xi1>
    %cst_4 = arith.constant {MetaUse} dense<true> : tensor<64x64xi1>
    %cst_5 = arith.constant {MetaUse} dense<64> : tensor<64x1xi32>
    %cst_6 = arith.constant {MetaUse} dense<64> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c4194304_i32 = arith.constant 4194304 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.make_range {DataUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %6 = tt.broadcast %5 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
    %7 = tt.broadcast %5 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %8 = tt.splat %arg9 {DataUse} : f32 -> tensor<64x128xf32>
    %9 = tt.splat %arg9 {DataUse} : f32 -> tensor<128x128xf32>
    scf.for %arg10 = %0 to %c65536_i32 step %c28_i32  : i32 {
      %10 = arith.divsi %arg10, %c64_i32 : i32
      %11 = arith.muli %10, %c64_i32 : i32
      %12 = arith.subi %arg10, %11 : i32
      %13 = arith.muli %10, %c8192_i32 : i32
      %14 = arith.extsi %13 : i32 to i64
      %15 = arith.remsi %10, %c8_i32 : i32
      %16 = arith.muli %15, %c524288_i32 : i32
      %17 = arith.divsi %10, %c8_i32 : i32
      %18 = arith.muli %17, %c4194304_i32 : i32
      %19 = arith.addi %16, %18 : i32
      %20 = arith.extsi %19 : i32 to i64
      %21 = tt.addptr %arg0, %20 : !tt.ptr<f16>, i64
      %22 = tt.addptr %arg1, %20 : !tt.ptr<f16>, i64
      %23 = tt.addptr %arg2, %20 : !tt.ptr<f16>, i64
      %24 = tt.addptr %arg3, %20 : !tt.ptr<f16>, i64
      %25 = tt.addptr %arg4, %20 : !tt.ptr<f16>, i64
      %26 = tt.addptr %arg5, %20 : !tt.ptr<f16>, i64
      %27 = tt.addptr %arg6, %20 : !tt.ptr<f16>, i64
      %28 = tt.addptr %arg7, %14 : !tt.ptr<f32>, i64
      %29 = tt.addptr %arg8, %14 : !tt.ptr<f32>, i64
      %30 = arith.muli %12, %c128_i32 : i32
      %31 = tt.splat %30 {MetaUse} : i32 -> tensor<128xi32>
      %32 = tt.splat %30 {DataUse} : i32 -> tensor<128xi32>
      %33 = arith.addi %31, %3 {MetaUse} : tensor<128xi32>
      %34 = arith.addi %32, %4 {DataUse} : tensor<128xi32>
      %35 = tt.expand_dims %33 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %36 = arith.muli %35, %cst_6 {MetaUse} : tensor<128x1xi32>
      %37 = tt.splat %22 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %38 = tt.addptr %37, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %39 = tt.broadcast %38 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %40 = tt.addptr %39, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %41 = tt.load %40 {DataUse} : tensor<128x64x!tt.ptr<f16>>
      %42 = tt.splat %23 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %43 = tt.addptr %42, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %44 = tt.broadcast %43 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %45 = tt.addptr %44, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %46 = tt.load %45 {DataUse} : tensor<128x64x!tt.ptr<f16>>
      %47 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %48 = tt.splat %24 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %49 = tt.splat %28 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %50 = tt.splat %29 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %51 = tt.trans %41 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %52 = tt.expand_dims %34 {DataUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %53 = tt.broadcast %52 {DataUse} : tensor<1x128xi32> -> tensor<64x128xi32>
      %54 = tt.trans %46 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %55 = tt.splat %25 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %56:2 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %cst_2, %arg13 = %cst_2) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
        %78 = arith.muli %arg11, %c64_i32 : i32
        %79 = arith.addi %30, %78 : i32
        %80 = tt.splat %79 {MetaUse} : i32 -> tensor<64xi32>
        %81 = tt.splat %79 {DataUse} : i32 -> tensor<64xi32>
        %82 = arith.addi %80, %1 {MetaUse} : tensor<64xi32>
        %83 = arith.addi %81, %2 {DataUse} : tensor<64xi32>
        %84 = tt.expand_dims %82 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %85 = tt.expand_dims %83 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %86 = arith.muli %84, %cst_5 {MetaUse} : tensor<64x1xi32>
        %87 = tt.addptr %47, %86 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %88 = tt.broadcast %87 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
        %89 = tt.addptr %88, %7 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        %90 = tt.load %89 {DataUse} : tensor<64x64x!tt.ptr<f16>>
        %91 = tt.addptr %48, %86 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %92 = tt.broadcast %91 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
        %93 = tt.addptr %92, %7 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        %94 = tt.load %93 {DataUse} : tensor<64x64x!tt.ptr<f16>>
        %95 = tt.addptr %49, %82 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %96 = tt.load %95 {DataUse} : tensor<64x!tt.ptr<f32>>
        %97 = tt.addptr %50, %82 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %98 = tt.load %97 {DataUse} : tensor<64x!tt.ptr<f32>>
        %99 = tt.dot %90, %51, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
        %100 = arith.mulf %99, %8 {DataUse} : tensor<64x128xf32>
        %101 = tt.expand_dims %96 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %102 = tt.broadcast %101 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
        %103 = arith.subf %100, %102 {DataUse} : tensor<64x128xf32>
        %104 = math.exp %103 {DataUse} : tensor<64x128xf32>
        %105 = tt.broadcast %85 {DataUse} : tensor<64x1xi32> -> tensor<64x128xi32>
        %106 = arith.cmpi sge, %105, %53 {DataUse} : tensor<64x128xi32>
        %107 = arith.select %106, %104, %cst_0 {DataUse} : tensor<64x128xi1>, tensor<64x128xf32>
        %108 = arith.truncf %107 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
        %109 = tt.trans %108 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
        %110 = tt.dot %109, %94, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
        %111 = tt.dot %94, %54, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
        %112 = tt.expand_dims %98 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %113 = tt.broadcast %112 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
        %114 = arith.subf %111, %113 {DataUse} : tensor<64x128xf32>
        %115 = arith.extf %108 {DataUse} : tensor<64x128xf16> to tensor<64x128xf32>
        %116 = arith.mulf %115, %114 {DataUse} : tensor<64x128xf32>
        %117 = arith.mulf %116, %8 {DataUse} : tensor<64x128xf32>
        %118 = arith.truncf %117 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
        %119 = tt.trans %118 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
        %120 = tt.dot %119, %90, %arg12 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
        %121 = tt.dot %118, %41, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
        %122 = tt.addptr %55, %86 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %123 = tt.broadcast %122 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x64x!tt.ptr<f16>>
        %124 = tt.addptr %123, %7 {MetaUse} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
        %125 = arith.truncf %121 {DataUse} : tensor<64x64xf32> to tensor<64x64xf16>
        %126 = tt.atomic_rmw fadd, acq_rel, gpu, %124, %125, %cst_4 {Undefined} : (tensor<64x64x!tt.ptr<f16>>, tensor<64x64xf16>, tensor<64x64xi1>) -> tensor<64x64xf16>
        scf.yield %120, %110 : tensor<128x64xf32>, tensor<128x64xf32>
      } {DataUse}
      %57 = arith.addi %30, %c128_i32 : i32
      %58 = arith.subi %c8064_i32, %30 : i32
      %59 = arith.divsi %58, %c128_i32 : i32
      %60 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %61 = tt.splat %24 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %62 = tt.splat %28 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %63 = tt.splat %29 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %64 = tt.trans %41 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %65 = tt.trans %46 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %66 = tt.splat %25 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %67:2 = scf.for %arg11 = %c0_i32 to %59 step %c1_i32 iter_args(%arg12 = %56#0, %arg13 = %56#1) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
        %78 = arith.muli %arg11, %c128_i32 : i32
        %79 = arith.addi %57, %78 : i32
        %80 = tt.splat %79 {MetaUse} : i32 -> tensor<128xi32>
        %81 = arith.addi %80, %3 {MetaUse} : tensor<128xi32>
        %82 = tt.expand_dims %81 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %83 = arith.muli %82, %cst_6 {MetaUse} : tensor<128x1xi32>
        %84 = tt.addptr %60, %83 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %85 = tt.broadcast %84 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %86 = tt.addptr %85, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %87 = tt.load %86 {DataUse} : tensor<128x64x!tt.ptr<f16>>
        %88 = tt.addptr %61, %83 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %89 = tt.broadcast %88 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %90 = tt.addptr %89, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %91 = tt.load %90 {DataUse} : tensor<128x64x!tt.ptr<f16>>
        %92 = tt.addptr %62, %81 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %93 = tt.load %92 {DataUse} : tensor<128x!tt.ptr<f32>>
        %94 = tt.addptr %63, %81 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %95 = tt.load %94 {DataUse} : tensor<128x!tt.ptr<f32>>
        %96 = tt.dot %87, %64, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %97 = arith.mulf %96, %9 {DataUse} : tensor<128x128xf32>
        %98 = tt.expand_dims %93 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %99 = tt.broadcast %98 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %100 = arith.subf %97, %99 {DataUse} : tensor<128x128xf32>
        %101 = math.exp %100 {DataUse} : tensor<128x128xf32>
        %102 = arith.truncf %101 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
        %103 = tt.trans %102 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %104 = tt.dot %103, %91, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %105 = tt.dot %91, %65, %cst {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %106 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %107 = tt.broadcast %106 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %108 = arith.subf %105, %107 {DataUse} : tensor<128x128xf32>
        %109 = arith.extf %102 {DataUse} : tensor<128x128xf16> to tensor<128x128xf32>
        %110 = arith.mulf %109, %108 {DataUse} : tensor<128x128xf32>
        %111 = arith.mulf %110, %9 {DataUse} : tensor<128x128xf32>
        %112 = arith.truncf %111 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
        %113 = tt.trans %112 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %114 = tt.dot %113, %87, %arg12 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %115 = tt.dot %112, %41, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %116 = tt.addptr %66, %83 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %117 = tt.broadcast %116 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %118 = tt.addptr %117, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %119 = arith.truncf %115 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
        %120 = tt.atomic_rmw fadd, acq_rel, gpu, %118, %119, %cst_3 {Undefined} : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
        scf.yield %114, %104 : tensor<128x64xf32>, tensor<128x64xf32>
      } {DataUse}
      %68 = tt.splat %26 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %69 = tt.addptr %68, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %70 = tt.broadcast %69 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %71 = tt.addptr %70, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %72 = arith.truncf %67#0 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %71, %72 : tensor<128x64x!tt.ptr<f16>>
      %73 = tt.splat %27 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %74 = tt.addptr %73, %36 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %75 = tt.broadcast %74 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %76 = tt.addptr %75, %6 {MetaUse} : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %77 = arith.truncf %67#1 {DataUse} : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %76, %77 : tensor<128x64x!tt.ptr<f16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_bwd_fp8/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 16)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 2048)>
// CHECK-NEXT: #map2 = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant dense<true> : tensor<16x128xi1>
// CHECK-NEXT:     %cst_0 = arith.constant 4.480000e+02 : f32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c131072_i32 = arith.constant 131072 : i32
// CHECK-NEXT:     %c1048576_i32 = arith.constant 1048576 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
// CHECK-NEXT:     %cst_3 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_4 = arith.constant 9.99999997E-7 : f32
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map1()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map()[%1]
// CHECK-NEXT:     %7 = affine.apply #map2()[%1]
// CHECK-NEXT:     %8 = tt.get_program_id x : i32
// CHECK-NEXT:     %9 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %10 = tt.expand_dims %9 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %11 = tt.broadcast %10 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %12 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
// CHECK-NEXT:     %13 = tt.broadcast %10 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
// CHECK-NEXT:     %14 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:     scf.for %arg13 = %8 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %15 = arith.divsi %arg13, %c8_i32 : i32
// CHECK-NEXT:       %16 = arith.muli %15, %c8_i32 : i32
// CHECK-NEXT:       %17 = arith.subi %arg13, %16 : i32
// CHECK-NEXT:       %18 = arith.muli %15, %c1024_i32 : i32
// CHECK-NEXT:       %19 = arith.extsi %18 : i32 to i64
// CHECK-NEXT:       %20 = arith.remsi %15, %c8_i32 : i32
// CHECK-NEXT:       %21 = arith.muli %20, %c131072_i32 : i32
// CHECK-NEXT:       %22 = arith.divsi %15, %c8_i32 : i32
// CHECK-NEXT:       %23 = arith.muli %22, %c1048576_i32 : i32
// CHECK-NEXT:       %24 = arith.addi %21, %23 : i32
// CHECK-NEXT:       %25 = arith.extsi %24 : i32 to i64
// CHECK-NEXT:       %26 = tt.addptr %arg0, %25 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %27 = tt.addptr %arg1, %25 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %28 = tt.addptr %arg2, %25 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %29 = tt.addptr %arg3, %25 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %30 = tt.addptr %arg4, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %31 = tt.addptr %arg5, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %32 = tt.addptr %arg6, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %33 = tt.addptr %arg7, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %34 = tt.addptr %arg8, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %35 = arith.divsi %18, %c128_i32 : i32
// CHECK-NEXT:       %36 = arith.extsi %35 : i32 to i64
// CHECK-NEXT:       %37 = tt.addptr %arg10, %36 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %38 = tt.addptr %arg11, %36 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %39 = arith.muli %17, %c128_i32 : i32
// CHECK-NEXT:       %40 = tt.splat %39 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %41 = arith.addi %40, %9 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %42 = tt.expand_dims %41 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %43 = arith.muli %42, %cst_1 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %44 = tt.splat %27 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %45 = tt.addptr %44, %43 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
// CHECK-NEXT:       %46 = tt.broadcast %45 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %47 = tt.addptr %46, %11 {MetaUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
// CHECK-NEXT:       %48 = tt.load %47 {DataUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %49 = tt.splat %28 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %50 = tt.addptr %49, %43 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %51 = tt.broadcast %50 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %52 = tt.addptr %51, %11 {MetaUse} : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
// CHECK-NEXT:       %53 = tt.load %52 {DataUse} : tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %54 = arith.divsi %39, %c128_i32 : i32
// CHECK-NEXT:       %55 = tt.addptr %38, %54 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %56 = tt.load %55 : !tt.ptr<f32>
// CHECK-NEXT:       %57 = tt.splat %26 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %58 = tt.trans %48 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:       %59 = tt.splat %56 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:       %60 = tt.splat %33 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:       %61 = tt.splat %29 {MetaUse} : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
// CHECK-NEXT:       %62 = tt.trans %53 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:       %63 = tt.splat %34 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:       %64 = tt.splat %30 {MetaUse} : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
// CHECK-NEXT:       %65:2 = scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg15 = %cst_6, %arg16 = %cst_6) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %74 = arith.muli %arg14, %c32_i32 : i32
// CHECK-NEXT:         %75 = tt.splat %74 {MetaUse} : i32 -> tensor<32xi32>
// CHECK-NEXT:         %76 = arith.addi %75, %12 {MetaUse} : tensor<32xi32>
// CHECK-NEXT:         %77 = arith.divsi %74, %c128_i32 : i32
// CHECK-NEXT:         %78 = tt.addptr %37, %77 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %79 = tt.load %78 : !tt.ptr<f32>
// CHECK-NEXT:         %80 = tt.expand_dims %76 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
// CHECK-NEXT:         %81 = arith.muli %80, %cst_2 {MetaUse} : tensor<32x1xi32>
// CHECK-NEXT:         %82 = tt.addptr %57, %81 {MetaUse} : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
// CHECK-NEXT:         %83 = tt.broadcast %82 {MetaUse} : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %84 = tt.addptr %83, %13 {MetaUse} : tensor<32x128x!tt.ptr<f8E4M3FN>>, tensor<32x128xi32>
// CHECK-NEXT:         %85 = tt.load %84 {DataUse} : tensor<32x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %86 = tt.dot %85, %58, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
// CHECK-NEXT:         %87 = arith.cmpf une, %86, %86 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %88 = arith.select %87, %cst_5, %86 {DataUse} : tensor<32x128xi1>, tensor<32x128xf32>
// CHECK-NEXT:         %89 = tt.splat %79 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:         %90 = arith.mulf %88, %89 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %91 = arith.mulf %90, %59 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %92 = tt.addptr %60, %76 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %93 = tt.load %92 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %94 = arith.mulf %91, %14 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %95 = tt.expand_dims %93 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %96 = tt.broadcast %95 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %97 = arith.subf %94, %96 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %98 = math.exp %97 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %99 = tt.addptr %61, %81 {MetaUse} : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {MetaUse} : tensor<32x1x!tt.ptr<f16>> -> tensor<32x128x!tt.ptr<f16>>
// CHECK-NEXT:         %101 = tt.addptr %100, %13 {MetaUse} : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
// CHECK-NEXT:         %102 = tt.load %101 {DataUse} : tensor<32x128x!tt.ptr<f16>>
// CHECK-NEXT:         %103 = arith.truncf %98 {DataUse} : tensor<32x128xf32> to tensor<32x128xf16>
// CHECK-NEXT:         %extracted_slice_8 = tensor.extract_slice %103[%2, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xf16> to tensor<16x128xf16>
// CHECK-NEXT:         %104 = tensor.empty() : tensor<32x128xf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_8 into %104[%2, 0] [16, 128] [1, 1] {cv_communication_slice} : tensor<16x128xf16> into tensor<32x128xf16>
// CHECK-NEXT:         %105 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<32x128xf16> -> tensor<128x32xf16>
// CHECK-NEXT:         %106 = tt.dot %105, %102, %arg15 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %107 = tt.dot %102, %62, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xf16> * tensor<128x128xf16> -> tensor<32x128xf32>
// CHECK-NEXT:         %108 = tt.addptr %63, %76 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
// CHECK-NEXT:         %109 = tt.load %108 {DataUse} : tensor<32x!tt.ptr<f32>>
// CHECK-NEXT:         %110 = tt.expand_dims %109 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
// CHECK-NEXT:         %111 = tt.broadcast %110 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
// CHECK-NEXT:         %112 = arith.subf %107, %111 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %113 = arith.mulf %98, %112 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %114 = arith.mulf %113, %14 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %115 = math.absf %114 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %116 = tt.reshape %115 allow_reorder {DataUse} : tensor<32x128xf32> -> tensor<4096xf32>
// CHECK-NEXT:         %extracted_slice_9 = tensor.extract_slice %116[%3] [2048] [1] {to_be_bubbled_slice} : tensor<4096xf32> to tensor<2048xf32>
// CHECK-NEXT:         %117 = "tt.reduce"(%extracted_slice_9) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg17: f32, %arg18: f32):
// CHECK-NEXT:           %141 = arith.maxnumf %arg17, %arg18 : f32
// CHECK-NEXT:           tt.reduce.return %141 : f32
// CHECK-NEXT:         }) {tiled_op} : (tensor<2048xf32>) -> f32
// CHECK-NEXT:         %118 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:         %inserted = tensor.insert %117 into %118[%1] {vv_communication} : tensor<2xf32>
// CHECK-NEXT:         %119 = "tt.reduce"(%inserted) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg17: f32, %arg18: f32):
// CHECK-NEXT:           %141 = arith.maximumf %arg17, %arg18 : f32
// CHECK-NEXT:           tt.reduce.return %141 : f32
// CHECK-NEXT:         }) {tiled_op} : (tensor<2xf32>) -> f32
// CHECK-NEXT:         %120 = arith.cmpf ogt, %119, %cst_4 : f32
// CHECK-NEXT:         %121 = scf.if %120 -> (f32) {
// CHECK-NEXT:           %141 = arith.divf %cst_0, %119 : f32
// CHECK-NEXT:           scf.yield %141 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %cst_3 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         %122 = tt.splat %121 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:         %123 = arith.mulf %114, %122 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %124 = tt.fp_to_fp %123 {DataUse}, rounding = rtne : tensor<32x128xf32> -> tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %124[%4, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xf8E4M3FN> to tensor<16x128xf8E4M3FN>
// CHECK-NEXT:         %125 = tensor.empty() : tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice_11 = tensor.insert_slice %extracted_slice_10 into %125[%4, 0] [16, 128] [1, 1] {cv_communication_slice} : tensor<16x128xf8E4M3FN> into tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %126 = tt.dot %inserted_slice_11, %48, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
// CHECK-NEXT:         %127 = arith.divf %56, %121 : f32
// CHECK-NEXT:         %128 = tt.splat %127 {DataUse} : f32 -> tensor<32x128xf32>
// CHECK-NEXT:         %129 = arith.mulf %126, %128 {DataUse} : tensor<32x128xf32>
// CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %124[%5, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xf8E4M3FN> to tensor<16x128xf8E4M3FN>
// CHECK-NEXT:         %130 = tensor.empty() : tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice_13 = tensor.insert_slice %extracted_slice_12 into %130[%5, 0] [16, 128] [1, 1] {cv_communication_slice} : tensor<16x128xf8E4M3FN> into tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %131 = tt.trans %inserted_slice_13 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN> -> tensor<128x32xf8E4M3FN>
// CHECK-NEXT:         %132 = tt.dot %131, %85, %cst_6 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x32xf8E4M3FN> * tensor<32x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %133 = arith.divf %79, %121 : f32
// CHECK-NEXT:         %134 = tt.splat %133 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:         %135 = arith.mulf %132, %134 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %136 = arith.addf %arg16, %135 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %137 = tt.addptr %64, %81 {MetaUse} : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
// CHECK-NEXT:         %138 = tt.broadcast %137 {MetaUse} : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
// CHECK-NEXT:         %139 = tt.addptr %138, %13 {MetaUse} : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
// CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %129[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128xf32> to tensor<16x128xf32>
// CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %139[%6, 0] [16, 128] [1, 1] {to_be_bubbled_slice} : tensor<32x128x!tt.ptr<f32>> to tensor<16x128x!tt.ptr<f32>>
// CHECK-NEXT:         %140 = tt.atomic_rmw fadd, acq_rel, gpu, %extracted_slice_15, %extracted_slice_14, %cst {Undefined, tiled_op} : (tensor<16x128x!tt.ptr<f32>>, tensor<16x128xf32>, tensor<16x128xi1>) -> tensor<16x128xf32>
// CHECK-NEXT:         scf.yield %106, %136 : tensor<128x128xf32>, tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %66 = tt.splat %31 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %67 = tt.addptr %66, %43 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %68 = tt.broadcast %67 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %69 = tt.addptr %68, %11 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %65#1[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:       %extracted_slice_7 = tensor.extract_slice %69[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128x!tt.ptr<f32>> to tensor<64x128x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_7, %extracted_slice {tiled_op} : tensor<64x128x!tt.ptr<f32>>
// CHECK-NEXT:       %70 = tt.splat %32 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %71 = tt.addptr %70, %43 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %72 = tt.broadcast %71 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %73 = tt.addptr %72, %11 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       tt.store %73, %65#0 : tensor<128x128x!tt.ptr<f32>>
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
    %cst = arith.constant {MetaUse} dense<true> : tensor<32x128xi1>
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<32x128xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant {MetaUse} dense<128> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %cst_5 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_6 = arith.constant 4.480000e+02 : f32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %2 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %3 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %4 = tt.make_range {MetaUse, end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %5 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<32x128xi32>
    %6 = tt.splat %arg9 {DataUse} : f32 -> tensor<32x128xf32>
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
      %32 = tt.splat %31 {MetaUse} : i32 -> tensor<128xi32>
      %33 = arith.addi %32, %1 {MetaUse} : tensor<128xi32>
      %34 = tt.expand_dims %33 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %35 = arith.muli %34, %cst_5 {MetaUse} : tensor<128x1xi32>
      %36 = tt.splat %19 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
      %37 = tt.addptr %36, %35 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
      %38 = tt.broadcast %37 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
      %39 = tt.addptr %38, %3 {MetaUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
      %40 = tt.load %39 {DataUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>
      %41 = tt.splat %20 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.addptr %41, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %43 = tt.broadcast %42 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
      %44 = tt.addptr %43, %3 {MetaUse} : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
      %45 = tt.load %44 {DataUse} : tensor<128x128x!tt.ptr<f16>>
      %46 = arith.divsi %31, %c128_i32 : i32
      %47 = tt.addptr %30, %46 : !tt.ptr<f32>, i32
      %48 = tt.load %47 : !tt.ptr<f32>
      %49 = tt.splat %18 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
      %50 = tt.trans %40 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
      %51 = tt.splat %48 {DataUse} : f32 -> tensor<32x128xf32>
      %52 = tt.splat %25 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
      %53 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
      %54 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
      %55 = tt.splat %26 {MetaUse} : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
      %56 = tt.splat %22 {MetaUse} : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
      %57:2 = scf.for %arg14 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg15 = %cst_0, %arg16 = %cst_0) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
        %66 = arith.muli %arg14, %c32_i32 : i32
        %67 = tt.splat %66 {MetaUse} : i32 -> tensor<32xi32>
        %68 = arith.addi %67, %4 {MetaUse} : tensor<32xi32>
        %69 = arith.divsi %66, %c128_i32 : i32
        %70 = tt.addptr %29, %69 : !tt.ptr<f32>, i32
        %71 = tt.load %70 : !tt.ptr<f32>
        %72 = tt.expand_dims %68 {MetaUse, axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %73 = arith.muli %72, %cst_4 {MetaUse} : tensor<32x1xi32>
        %74 = tt.addptr %49, %73 {MetaUse} : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
        %75 = tt.broadcast %74 {MetaUse} : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x128x!tt.ptr<f8E4M3FN>>
        %76 = tt.addptr %75, %5 {MetaUse} : tensor<32x128x!tt.ptr<f8E4M3FN>>, tensor<32x128xi32>
        %77 = tt.load %76 {DataUse} : tensor<32x128x!tt.ptr<f8E4M3FN>>
        %78 = tt.dot %77, %50, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
        %79 = arith.cmpf une, %78, %78 {DataUse} : tensor<32x128xf32>
        %80 = arith.select %79, %cst_1, %78 {DataUse} : tensor<32x128xi1>, tensor<32x128xf32>
        %81 = tt.splat %71 {DataUse} : f32 -> tensor<32x128xf32>
        %82 = arith.mulf %80, %81 {DataUse} : tensor<32x128xf32>
        %83 = arith.mulf %82, %51 {DataUse} : tensor<32x128xf32>
        %84 = tt.addptr %52, %68 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %85 = tt.load %84 {DataUse} : tensor<32x!tt.ptr<f32>>
        %86 = arith.mulf %83, %6 {DataUse} : tensor<32x128xf32>
        %87 = tt.expand_dims %85 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %88 = tt.broadcast %87 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
        %89 = arith.subf %86, %88 {DataUse} : tensor<32x128xf32>
        %90 = math.exp %89 {DataUse} : tensor<32x128xf32>
        %91 = tt.addptr %53, %73 {MetaUse} : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
        %92 = tt.broadcast %91 {MetaUse} : tensor<32x1x!tt.ptr<f16>> -> tensor<32x128x!tt.ptr<f16>>
        %93 = tt.addptr %92, %5 {MetaUse} : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
        %94 = tt.load %93 {DataUse} : tensor<32x128x!tt.ptr<f16>>
        %95 = arith.truncf %90 {DataUse} : tensor<32x128xf32> to tensor<32x128xf16>
        %96 = tt.trans %95 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xf16> -> tensor<128x32xf16>
        %97 = tt.dot %96, %94, %arg15 {DataUse, triton_cv12.normalized_dot} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
        %98 = tt.dot %94, %54, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xf16> * tensor<128x128xf16> -> tensor<32x128xf32>
        %99 = tt.addptr %55, %68 {MetaUse} : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
        %100 = tt.load %99 {DataUse} : tensor<32x!tt.ptr<f32>>
        %101 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<32xf32> -> tensor<32x1xf32>
        %102 = tt.broadcast %101 {DataUse} : tensor<32x1xf32> -> tensor<32x128xf32>
        %103 = arith.subf %98, %102 {DataUse} : tensor<32x128xf32>
        %104 = arith.mulf %90, %103 {DataUse} : tensor<32x128xf32>
        %105 = arith.mulf %104, %6 {DataUse} : tensor<32x128xf32>
        %106 = math.absf %105 {DataUse} : tensor<32x128xf32>
        %107 = tt.reshape %106 allow_reorder {DataUse} : tensor<32x128xf32> -> tensor<4096xf32>
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
        %111 = tt.splat %110 {DataUse} : f32 -> tensor<32x128xf32>
        %112 = arith.mulf %105, %111 {DataUse} : tensor<32x128xf32>
        %113 = tt.fp_to_fp %112 {DataUse}, rounding = rtne : tensor<32x128xf32> -> tensor<32x128xf8E4M3FN>
        %114 = tt.dot %113, %40, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<32x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<32x128xf32>
        %115 = arith.divf %48, %110 : f32
        %116 = tt.splat %115 {DataUse} : f32 -> tensor<32x128xf32>
        %117 = arith.mulf %114, %116 {DataUse} : tensor<32x128xf32>
        %118 = tt.trans %113 {DataUse, order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN> -> tensor<128x32xf8E4M3FN>
        %119 = tt.dot %118, %77, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<128x32xf8E4M3FN> * tensor<32x128xf8E4M3FN> -> tensor<128x128xf32>
        %120 = arith.divf %71, %110 : f32
        %121 = tt.splat %120 {DataUse} : f32 -> tensor<128x128xf32>
        %122 = arith.mulf %119, %121 {DataUse} : tensor<128x128xf32>
        %123 = arith.addf %arg16, %122 {DataUse} : tensor<128x128xf32>
        %124 = tt.addptr %56, %73 {MetaUse} : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
        %125 = tt.broadcast %124 {MetaUse} : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
        %126 = tt.addptr %125, %5 {MetaUse} : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
        %127 = tt.atomic_rmw fadd, acq_rel, gpu, %126, %117, %cst {Undefined} : (tensor<32x128x!tt.ptr<f32>>, tensor<32x128xf32>, tensor<32x128xi1>) -> tensor<32x128xf32>
        scf.yield %97, %123 : tensor<128x128xf32>, tensor<128x128xf32>
      } {DataUse}
      %58 = tt.splat %23 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %59 = tt.addptr %58, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %60 = tt.broadcast %59 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %61 = tt.addptr %60, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %61, %57#1 : tensor<128x128x!tt.ptr<f32>>
      %62 = tt.splat %24 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %63 = tt.addptr %62, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %65 = tt.addptr %64, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %65, %57#0 : tensor<128x128x!tt.ptr<f32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_bwd_fp8_Q2/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 4096)>
// CHECK-NEXT: #map2 = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_bwd(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %cst = arith.constant dense<true> : tensor<32x128xi1>
// CHECK-NEXT:     %cst_0 = arith.constant 4.480000e+02 : f32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c131072_i32 = arith.constant 131072 : i32
// CHECK-NEXT:     %c1048576_i32 = arith.constant 1048576 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_3 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_4 = arith.constant 9.99999997E-7 : f32
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map1()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map()[%1]
// CHECK-NEXT:     %7 = affine.apply #map2()[%1]
// CHECK-NEXT:     %8 = tt.get_program_id x : i32
// CHECK-NEXT:     %9 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %10 = tt.expand_dims %9 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %11 = tt.broadcast %10 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %12 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %13 = tt.broadcast %10 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %14 = tt.splat %arg9 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:     scf.for %arg13 = %8 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %15 = arith.divsi %arg13, %c8_i32 : i32
// CHECK-NEXT:       %16 = arith.muli %15, %c8_i32 : i32
// CHECK-NEXT:       %17 = arith.subi %arg13, %16 : i32
// CHECK-NEXT:       %18 = arith.muli %15, %c1024_i32 : i32
// CHECK-NEXT:       %19 = arith.extsi %18 : i32 to i64
// CHECK-NEXT:       %20 = arith.remsi %15, %c8_i32 : i32
// CHECK-NEXT:       %21 = arith.muli %20, %c131072_i32 : i32
// CHECK-NEXT:       %22 = arith.divsi %15, %c8_i32 : i32
// CHECK-NEXT:       %23 = arith.muli %22, %c1048576_i32 : i32
// CHECK-NEXT:       %24 = arith.addi %21, %23 : i32
// CHECK-NEXT:       %25 = arith.extsi %24 : i32 to i64
// CHECK-NEXT:       %26 = tt.addptr %arg0, %25 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %27 = tt.addptr %arg1, %25 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %28 = tt.addptr %arg2, %25 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %29 = tt.addptr %arg3, %25 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %30 = tt.addptr %arg4, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %31 = tt.addptr %arg5, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %32 = tt.addptr %arg6, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %33 = tt.addptr %arg7, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %34 = tt.addptr %arg8, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %35 = arith.divsi %18, %c128_i32 : i32
// CHECK-NEXT:       %36 = arith.extsi %35 : i32 to i64
// CHECK-NEXT:       %37 = tt.addptr %arg10, %36 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %38 = tt.addptr %arg11, %36 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %39 = arith.muli %17, %c128_i32 : i32
// CHECK-NEXT:       %40 = tt.splat %39 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %41 = arith.addi %40, %9 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %42 = tt.expand_dims %41 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %43 = arith.muli %42, %cst_1 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %44 = tt.splat %27 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %45 = tt.addptr %44, %43 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
// CHECK-NEXT:       %46 = tt.broadcast %45 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %47 = tt.addptr %46, %11 {MetaUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
// CHECK-NEXT:       %48 = tt.load %47 {DataUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %49 = tt.splat %28 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
// CHECK-NEXT:       %50 = tt.addptr %49, %43 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
// CHECK-NEXT:       %51 = tt.broadcast %50 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %52 = tt.addptr %51, %11 {MetaUse} : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
// CHECK-NEXT:       %53 = tt.load %52 {DataUse} : tensor<128x128x!tt.ptr<f16>>
// CHECK-NEXT:       %54 = arith.divsi %39, %c128_i32 : i32
// CHECK-NEXT:       %55 = tt.addptr %38, %54 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %56 = tt.load %55 : !tt.ptr<f32>
// CHECK-NEXT:       %57 = tt.splat %26 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:       %58 = tt.trans %48 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:       %59 = tt.splat %56 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:       %60 = tt.splat %33 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %61 = tt.splat %29 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
// CHECK-NEXT:       %62 = tt.trans %53 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:       %63 = tt.splat %34 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %64 = tt.splat %30 {MetaUse} : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
// CHECK-NEXT:       %65:2 = scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg15 = %cst_6, %arg16 = %cst_6) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %74 = arith.muli %arg14, %c64_i32 : i32
// CHECK-NEXT:         %75 = tt.splat %74 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %76 = arith.addi %75, %12 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %77 = arith.divsi %74, %c128_i32 : i32
// CHECK-NEXT:         %78 = tt.addptr %37, %77 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %79 = tt.load %78 : !tt.ptr<f32>
// CHECK-NEXT:         %80 = tt.expand_dims %76 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %81 = arith.muli %80, %cst_2 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %82 = tt.addptr %57, %81 {MetaUse} : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
// CHECK-NEXT:         %83 = tt.broadcast %82 {MetaUse} : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %84 = tt.addptr %83, %13 {MetaUse} : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
// CHECK-NEXT:         %85 = tt.load %84 {DataUse} : tensor<64x128x!tt.ptr<f8E4M3FN>>
// CHECK-NEXT:         %86 = tt.dot %85, %58, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
// CHECK-NEXT:         %87 = arith.cmpf une, %86, %86 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %88 = arith.select %87, %cst_5, %86 {DataUse} : tensor<64x128xi1>, tensor<64x128xf32>
// CHECK-NEXT:         %89 = tt.splat %79 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:         %90 = arith.mulf %88, %89 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %91 = arith.mulf %90, %59 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %92 = tt.addptr %60, %76 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %93 = tt.load %92 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %94 = arith.mulf %91, %14 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %95 = tt.expand_dims %93 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %96 = tt.broadcast %95 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %97 = arith.subf %94, %96 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %98 = math.exp %97 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %99 = tt.addptr %61, %81 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
// CHECK-NEXT:         %101 = tt.addptr %100, %13 {MetaUse} : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
// CHECK-NEXT:         %102 = tt.load %101 {DataUse} : tensor<64x128x!tt.ptr<f16>>
// CHECK-NEXT:         %103 = arith.truncf %98 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
// CHECK-NEXT:         %extracted_slice_8 = tensor.extract_slice %103[%2, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf16> to tensor<32x128xf16>
// CHECK-NEXT:         %104 = tensor.empty() : tensor<64x128xf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_8 into %104[%2, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xf16> into tensor<64x128xf16>
// CHECK-NEXT:         %105 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
// CHECK-NEXT:         %106 = tt.dot %105, %102, %arg15 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %107 = tt.dot %102, %62, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x128xf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %108 = tt.addptr %63, %76 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:         %109 = tt.load %108 {DataUse} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:         %110 = tt.expand_dims %109 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %111 = tt.broadcast %110 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %112 = arith.subf %107, %111 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %113 = arith.mulf %98, %112 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %114 = arith.mulf %113, %14 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %115 = math.absf %114 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %116 = tt.reshape %115 allow_reorder {DataUse} : tensor<64x128xf32> -> tensor<8192xf32>
// CHECK-NEXT:         %extracted_slice_9 = tensor.extract_slice %116[%3] [4096] [1] {to_be_bubbled_slice} : tensor<8192xf32> to tensor<4096xf32>
// CHECK-NEXT:         %117 = "tt.reduce"(%extracted_slice_9) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg17: f32, %arg18: f32):
// CHECK-NEXT:           %141 = arith.maxnumf %arg17, %arg18 : f32
// CHECK-NEXT:           tt.reduce.return %141 : f32
// CHECK-NEXT:         }) {tiled_op} : (tensor<4096xf32>) -> f32
// CHECK-NEXT:         %118 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:         %inserted = tensor.insert %117 into %118[%1] {vv_communication} : tensor<2xf32>
// CHECK-NEXT:         %119 = "tt.reduce"(%inserted) <{axis = 0 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg17: f32, %arg18: f32):
// CHECK-NEXT:           %141 = arith.maximumf %arg17, %arg18 : f32
// CHECK-NEXT:           tt.reduce.return %141 : f32
// CHECK-NEXT:         }) {tiled_op} : (tensor<2xf32>) -> f32
// CHECK-NEXT:         %120 = arith.cmpf ogt, %119, %cst_4 : f32
// CHECK-NEXT:         %121 = scf.if %120 -> (f32) {
// CHECK-NEXT:           %141 = arith.divf %cst_0, %119 : f32
// CHECK-NEXT:           scf.yield %141 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %cst_3 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         %122 = tt.splat %121 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:         %123 = arith.mulf %114, %122 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %124 = tt.fp_to_fp %123 {DataUse}, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %extracted_slice_10 = tensor.extract_slice %124[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf8E4M3FN> to tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %125 = tensor.empty() : tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice_11 = tensor.insert_slice %extracted_slice_10 into %125[%4, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xf8E4M3FN> into tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %126 = tt.dot %inserted_slice_11, %48, %cst_5 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
// CHECK-NEXT:         %127 = arith.divf %56, %121 : f32
// CHECK-NEXT:         %128 = tt.splat %127 {DataUse} : f32 -> tensor<64x128xf32>
// CHECK-NEXT:         %129 = arith.mulf %126, %128 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %extracted_slice_12 = tensor.extract_slice %124[%5, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf8E4M3FN> to tensor<32x128xf8E4M3FN>
// CHECK-NEXT:         %130 = tensor.empty() : tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice_13 = tensor.insert_slice %extracted_slice_12 into %130[%5, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xf8E4M3FN> into tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %131 = tt.trans %inserted_slice_13 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
// CHECK-NEXT:         %132 = tt.dot %131, %85, %cst_6 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xf8E4M3FN> * tensor<64x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %133 = arith.divf %79, %121 : f32
// CHECK-NEXT:         %134 = tt.splat %133 {DataUse} : f32 -> tensor<128x128xf32>
// CHECK-NEXT:         %135 = arith.mulf %132, %134 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %136 = arith.addf %arg16, %135 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %137 = tt.addptr %64, %81 {MetaUse} : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
// CHECK-NEXT:         %138 = tt.broadcast %137 {MetaUse} : tensor<64x1x!tt.ptr<f32>> -> tensor<64x128x!tt.ptr<f32>>
// CHECK-NEXT:         %139 = tt.addptr %138, %13 {MetaUse} : tensor<64x128x!tt.ptr<f32>>, tensor<64x128xi32>
// CHECK-NEXT:         %extracted_slice_14 = tensor.extract_slice %129[%6, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf32> to tensor<32x128xf32>
// CHECK-NEXT:         %extracted_slice_15 = tensor.extract_slice %139[%6, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128x!tt.ptr<f32>> to tensor<32x128x!tt.ptr<f32>>
// CHECK-NEXT:         %140 = tt.atomic_rmw fadd, acq_rel, gpu, %extracted_slice_15, %extracted_slice_14, %cst {Undefined, tiled_op} : (tensor<32x128x!tt.ptr<f32>>, tensor<32x128xf32>, tensor<32x128xi1>) -> tensor<32x128xf32>
// CHECK-NEXT:         scf.yield %106, %136 : tensor<128x128xf32>, tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %66 = tt.splat %31 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %67 = tt.addptr %66, %43 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %68 = tt.broadcast %67 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %69 = tt.addptr %68, %11 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %65#1[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:       %extracted_slice_7 = tensor.extract_slice %69[%7, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128x!tt.ptr<f32>> to tensor<64x128x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_7, %extracted_slice {tiled_op} : tensor<64x128x!tt.ptr<f32>>
// CHECK-NEXT:       %70 = tt.splat %32 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
// CHECK-NEXT:       %71 = tt.addptr %70, %43 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
// CHECK-NEXT:       %72 = tt.broadcast %71 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
// CHECK-NEXT:       %73 = tt.addptr %72, %11 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
// CHECK-NEXT:       tt.store %73, %65#0 : tensor<128x128x!tt.ptr<f32>>
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
    %cst = arith.constant {MetaUse} dense<true> : tensor<64x128xi1>
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_5 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_6 = arith.constant 4.480000e+02 : f32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %2 = tt.expand_dims %1 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %3 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %4 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %5 = tt.broadcast %2 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
    %6 = tt.splat %arg9 {DataUse} : f32 -> tensor<64x128xf32>
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
      %32 = tt.splat %31 {MetaUse} : i32 -> tensor<128xi32>
      %33 = arith.addi %32, %1 {MetaUse} : tensor<128xi32>
      %34 = tt.expand_dims %33 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %35 = arith.muli %34, %cst_5 {MetaUse} : tensor<128x1xi32>
      %36 = tt.splat %19 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
      %37 = tt.addptr %36, %35 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
      %38 = tt.broadcast %37 {MetaUse} : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x128x!tt.ptr<f8E4M3FN>>
      %39 = tt.addptr %38, %3 {MetaUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>, tensor<128x128xi32>
      %40 = tt.load %39 {DataUse} : tensor<128x128x!tt.ptr<f8E4M3FN>>
      %41 = tt.splat %20 {MetaUse} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.addptr %41, %35 {MetaUse} : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %43 = tt.broadcast %42 {MetaUse} : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
      %44 = tt.addptr %43, %3 {MetaUse} : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
      %45 = tt.load %44 {DataUse} : tensor<128x128x!tt.ptr<f16>>
      %46 = arith.divsi %31, %c128_i32 : i32
      %47 = tt.addptr %30, %46 : !tt.ptr<f32>, i32
      %48 = tt.load %47 : !tt.ptr<f32>
      %49 = tt.splat %18 {MetaUse} : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
      %50 = tt.trans %40 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
      %51 = tt.splat %48 {DataUse} : f32 -> tensor<64x128xf32>
      %52 = tt.splat %25 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %53 = tt.splat %21 {MetaUse} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>>
      %54 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
      %55 = tt.splat %26 {MetaUse} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %56 = tt.splat %22 {MetaUse} : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
      %57:2 = scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg15 = %cst_0, %arg16 = %cst_0) -> (tensor<128x128xf32>, tensor<128x128xf32>)  : i32 {
        %66 = arith.muli %arg14, %c64_i32 : i32
        %67 = tt.splat %66 {MetaUse} : i32 -> tensor<64xi32>
        %68 = arith.addi %67, %4 {MetaUse} : tensor<64xi32>
        %69 = arith.divsi %66, %c128_i32 : i32
        %70 = tt.addptr %29, %69 : !tt.ptr<f32>, i32
        %71 = tt.load %70 : !tt.ptr<f32>
        %72 = tt.expand_dims %68 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %73 = arith.muli %72, %cst_4 {MetaUse} : tensor<64x1xi32>
        %74 = tt.addptr %49, %73 {MetaUse} : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
        %75 = tt.broadcast %74 {MetaUse} : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
        %76 = tt.addptr %75, %5 {MetaUse} : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
        %77 = tt.load %76 {DataUse} : tensor<64x128x!tt.ptr<f8E4M3FN>>
        %78 = tt.dot %77, %50, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
        %79 = arith.cmpf une, %78, %78 {DataUse} : tensor<64x128xf32>
        %80 = arith.select %79, %cst_1, %78 {DataUse} : tensor<64x128xi1>, tensor<64x128xf32>
        %81 = tt.splat %71 {DataUse} : f32 -> tensor<64x128xf32>
        %82 = arith.mulf %80, %81 {DataUse} : tensor<64x128xf32>
        %83 = arith.mulf %82, %51 {DataUse} : tensor<64x128xf32>
        %84 = tt.addptr %52, %68 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %85 = tt.load %84 {DataUse} : tensor<64x!tt.ptr<f32>>
        %86 = arith.mulf %83, %6 {DataUse} : tensor<64x128xf32>
        %87 = tt.expand_dims %85 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %88 = tt.broadcast %87 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
        %89 = arith.subf %86, %88 {DataUse} : tensor<64x128xf32>
        %90 = math.exp %89 {DataUse} : tensor<64x128xf32>
        %91 = tt.addptr %53, %73 {MetaUse} : tensor<64x1x!tt.ptr<f16>>, tensor<64x1xi32>
        %92 = tt.broadcast %91 {MetaUse} : tensor<64x1x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
        %93 = tt.addptr %92, %5 {MetaUse} : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
        %94 = tt.load %93 {DataUse} : tensor<64x128x!tt.ptr<f16>>
        %95 = arith.truncf %90 {DataUse} : tensor<64x128xf32> to tensor<64x128xf16>
        %96 = tt.trans %95 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf16> -> tensor<128x64xf16>
        %97 = tt.dot %96, %94, %arg15 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %98 = tt.dot %94, %54, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf16> * tensor<128x128xf16> -> tensor<64x128xf32>
        %99 = tt.addptr %55, %68 {MetaUse} : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
        %100 = tt.load %99 {DataUse} : tensor<64x!tt.ptr<f32>>
        %101 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %102 = tt.broadcast %101 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
        %103 = arith.subf %98, %102 {DataUse} : tensor<64x128xf32>
        %104 = arith.mulf %90, %103 {DataUse} : tensor<64x128xf32>
        %105 = arith.mulf %104, %6 {DataUse} : tensor<64x128xf32>
        %106 = math.absf %105 {DataUse} : tensor<64x128xf32>
        %107 = tt.reshape %106 allow_reorder {DataUse} : tensor<64x128xf32> -> tensor<8192xf32>
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
        %111 = tt.splat %110 {DataUse} : f32 -> tensor<64x128xf32>
        %112 = arith.mulf %105, %111 {DataUse} : tensor<64x128xf32>
        %113 = tt.fp_to_fp %112 {DataUse}, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
        %114 = tt.dot %113, %40, %cst_1 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<64x128xf32>
        %115 = arith.divf %48, %110 : f32
        %116 = tt.splat %115 {DataUse} : f32 -> tensor<64x128xf32>
        %117 = arith.mulf %114, %116 {DataUse} : tensor<64x128xf32>
        %118 = tt.trans %113 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
        %119 = tt.dot %118, %77, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xf8E4M3FN> * tensor<64x128xf8E4M3FN> -> tensor<128x128xf32>
        %120 = arith.divf %71, %110 : f32
        %121 = tt.splat %120 {DataUse} : f32 -> tensor<128x128xf32>
        %122 = arith.mulf %119, %121 {DataUse} : tensor<128x128xf32>
        %123 = arith.addf %arg16, %122 {DataUse} : tensor<128x128xf32>
        %124 = tt.addptr %56, %73 {MetaUse} : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
        %125 = tt.broadcast %124 {MetaUse} : tensor<64x1x!tt.ptr<f32>> -> tensor<64x128x!tt.ptr<f32>>
        %126 = tt.addptr %125, %5 {MetaUse} : tensor<64x128x!tt.ptr<f32>>, tensor<64x128xi32>
        %127 = tt.atomic_rmw fadd, acq_rel, gpu, %126, %117, %cst {Undefined} : (tensor<64x128x!tt.ptr<f32>>, tensor<64x128xf32>, tensor<64x128xi1>) -> tensor<64x128xf32>
        scf.yield %97, %123 : tensor<128x128xf32>, tensor<128x128xf32>
      } {DataUse}
      %58 = tt.splat %23 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %59 = tt.addptr %58, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %60 = tt.broadcast %59 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %61 = tt.addptr %60, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %61, %57#1 : tensor<128x128x!tt.ptr<f32>>
      %62 = tt.splat %24 {MetaUse} : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %63 = tt.addptr %62, %35 {MetaUse} : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %65 = tt.addptr %64, %3 {MetaUse} : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %65, %57#0 : tensor<128x128x!tt.ptr<f32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_fwd/sep_md_false.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c8388608_i64 = arith.constant 8388608 : i64
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c8192_i64 = arith.constant 8192 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c65536_i32 = arith.constant 65536 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<5.000000e-01> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:     %6 = tt.get_program_id x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg8 = %6 to %c65536_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %10 = arith.divsi %8, %c8_i32 : i32
// CHECK-NEXT:       %11 = arith.remsi %8, %c8_i32 : i32
// CHECK-NEXT:       %12 = arith.muli %11, %arg7 : i32
// CHECK-NEXT:       %13 = arith.divsi %12, %c8_i32 : i32
// CHECK-NEXT:       %14 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %15 = arith.muli %14, %c8388608_i64 : i64
// CHECK-NEXT:       %16 = arith.extsi %11 : i32 to i64
// CHECK-NEXT:       %17 = arith.muli %16, %c1048576_i64 : i64
// CHECK-NEXT:       %18 = arith.addi %15, %17 : i64
// CHECK-NEXT:       %19 = arith.extsi %13 : i32 to i64
// CHECK-NEXT:       %20 = arith.muli %19, %c1048576_i64 : i64
// CHECK-NEXT:       %21 = arith.addi %15, %20 : i64
// CHECK-NEXT:       %22 = tt.addptr %arg0, %18 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %23 = arith.muli %9, %c128_i32 : i32
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %22, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
// CHECK-NEXT:       %25 = tt.addptr %arg2, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %26 = tt.make_tensor_ptr %25, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
// CHECK-NEXT:       %27 = tt.addptr %arg1, %21 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %28 = tt.make_tensor_ptr %27, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16>>
// CHECK-NEXT:       %29 = tt.addptr %arg5, %18 : !tt.ptr<f16>, i64
// CHECK-NEXT:       %30 = arith.addi %23, %5 : i32
// CHECK-NEXT:       %31 = tt.make_tensor_ptr %29, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%30, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<64x128xf16>>
// CHECK-NEXT:       %32 = tt.splat %23 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %33 = arith.addi %32, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %34 = tt.load %24 {DataUse} : !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:       %35:5 = scf.for %arg9 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg10 = %cst_2, %arg11 = %cst, %arg12 = %cst_1, %arg13 = %26, %arg14 = %28) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>)  : i32 {
// CHECK-NEXT:         %46 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:         %47 = tt.trans %46 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
// CHECK-NEXT:         %48 = tt.dot %34, %47, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %49 = arith.mulf %48, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %50 = "tt.reduce"(%49) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %71 = arith.maximumf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %71 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %51 = arith.maximumf %arg12, %50 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %52 = tt.expand_dims %51 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %53 = tt.broadcast %52 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %54 = arith.subf %49, %53 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %55 = math.exp %54 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %56 = arith.truncf %55 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:         %57 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:         %58 = "tt.reduce"(%55) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %71 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %71 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %59 = arith.subf %arg12, %51 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %60 = math.exp %59 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %61 = arith.mulf %arg10, %60 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %62 = arith.addf %61, %58 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %63 = tt.expand_dims %60 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %64 = tt.broadcast %63 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %65 = arith.mulf %arg11, %64 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %extracted_slice_5 = tensor.extract_slice %56[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:         %66 = tensor.empty() : tensor<128x128xf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_5 into %66[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf16> into tensor<128x128xf16>
// CHECK-NEXT:         %67 = tt.dot %inserted_slice, %57, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %68 = arith.addf %67, %65 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         %69 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
// CHECK-NEXT:         %70 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
// CHECK-NEXT:         scf.yield %62, %68, %51, %69, %70 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %36 = math.log %35#0 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %37 = arith.addf %35#2, %36 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %38 = tt.expand_dims %35#0 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %39 = tt.broadcast %38 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %40 = arith.divf %35#1, %39 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %41 = arith.muli %8, %c8192_i32 : i32
// CHECK-NEXT:       %42 = tt.addptr %arg4, %41 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %43 = tt.splat %42 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %44 = tt.addptr %43, %33 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %37[%3] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %44[%3] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_3, %extracted_slice {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %45 = arith.truncf %40 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
// CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %45[%4, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf16> to tensor<64x128xf16>
// CHECK-NEXT:       tt.store %31, %extracted_slice_4 {tiled_op} : !tt.ptr<tensor<64x128xf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant {DataUse} dense<5.000000e-01> : tensor<128x128xf32>
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
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
    %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
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
      %25 = tt.splat %17 {MetaUse} : i32 -> tensor<128xi32>
      %26 = arith.addi %25, %1 {MetaUse} : tensor<128xi32>
      %27 = tt.load %18 {DataUse} : !tt.ptr<tensor<128x128xf16>>
      %28:5 = scf.for %arg9 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %cst_2, %arg12 = %cst_0, %arg13 = %20, %arg14 = %22) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>)  : i32 {
        %39 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xf16>>
        %40 = tt.trans %39 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %41 = tt.dot %27, %40, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
        %42 = arith.mulf %41, %cst_1 {DataUse} : tensor<128x128xf32>
        %43 = "tt.reduce"(%42) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %63 = arith.maximumf %arg15, %arg16 : f32
          tt.reduce.return %63 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %44 = arith.maximumf %arg12, %43 {DataUse} : tensor<128xf32>
        %45 = tt.expand_dims %44 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %46 = tt.broadcast %45 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %47 = arith.subf %42, %46 {DataUse} : tensor<128x128xf32>
        %48 = math.exp %47 {DataUse} : tensor<128x128xf32>
        %49 = arith.truncf %48 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
        %50 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<128x128xf16>>
        %51 = "tt.reduce"(%48) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %63 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %63 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %52 = arith.subf %arg12, %44 {DataUse} : tensor<128xf32>
        %53 = math.exp %52 {DataUse} : tensor<128xf32>
        %54 = arith.mulf %arg10, %53 {DataUse} : tensor<128xf32>
        %55 = arith.addf %54, %51 {DataUse} : tensor<128xf32>
        %56 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %57 = tt.broadcast %56 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %58 = arith.mulf %arg11, %57 {DataUse} : tensor<128x128xf32>
        %59 = tt.dot %49, %50, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
        %60 = arith.addf %59, %58 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
        %61 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
        %62 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xf16>>
        scf.yield %55, %60, %44, %61, %62 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf16>>, !tt.ptr<tensor<128x128xf16>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %29 = math.log %28#0 {DataUse} : tensor<128xf32>
      %30 = arith.addf %28#2, %29 {DataUse} : tensor<128xf32>
      %31 = tt.expand_dims %28#0 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %32 = tt.broadcast %31 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %33 = arith.divf %28#1, %32 {DataUse} : tensor<128x128xf32>
      %34 = arith.muli %2, %c8192_i32 : i32
      %35 = tt.addptr %arg4, %34 : !tt.ptr<f32>, i32
      %36 = tt.splat %35 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %37 = tt.addptr %36, %26 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %37, %30 : tensor<128x!tt.ptr<f32>>
      %38 = arith.truncf %33 {DataUse} : tensor<128x128xf32> to tensor<128x128xf16>
      tt.store %24, %38 : !tt.ptr<tensor<128x128xf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_fwd/sep_md_true.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c8388608_i64 = arith.constant 8388608 : i64
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c8192_i64 = arith.constant 8192 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<5.000000e-01> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = arith.index_cast %5 : index to i32
// CHECK-NEXT:     %7 = tt.get_program_id x : i32
// CHECK-NEXT:     %8 = tt.addptr %arg6, %7 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %9 = tt.load %8 : !tt.ptr<i32>
// CHECK-NEXT:     %10 = tt.addptr %8, %c1_i32 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %11 = tt.load %10 : !tt.ptr<i32>
// CHECK-NEXT:     %12 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg8 = %9 to %11 step %c1_i32  : i32 {
// CHECK-NEXT:       %13 = arith.divsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %14 = arith.remsi %arg8, %c64_i32 : i32
// CHECK-NEXT:       %15 = arith.divsi %13, %c8_i32 : i32
// CHECK-NEXT:       %16 = arith.remsi %13, %c8_i32 : i32
// CHECK-NEXT:       %17 = arith.muli %16, %arg7 : i32
// CHECK-NEXT:       %18 = arith.divsi %17, %c8_i32 : i32
// CHECK-NEXT:       %19 = arith.extsi %15 : i32 to i64
// CHECK-NEXT:       %20 = arith.muli %19, %c8388608_i64 : i64
// CHECK-NEXT:       %21 = arith.extsi %16 : i32 to i64
// CHECK-NEXT:       %22 = arith.muli %21, %c1048576_i64 : i64
// CHECK-NEXT:       %23 = arith.addi %20, %22 : i64
// CHECK-NEXT:       %24 = arith.extsi %18 : i32 to i64
// CHECK-NEXT:       %25 = arith.muli %24, %c1048576_i64 : i64
// CHECK-NEXT:       %26 = arith.addi %20, %25 : i64
// CHECK-NEXT:       %27 = tt.addptr %arg0, %23 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %28 = arith.muli %14, %c128_i32 : i32
// CHECK-NEXT:       %29 = tt.make_tensor_ptr %27, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:       %30 = tt.addptr %arg2, %26 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %31 = tt.make_tensor_ptr %30, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:       %32 = tt.addptr %arg1, %26 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %33 = tt.make_tensor_ptr %32, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xbf16>>
// CHECK-NEXT:       %34 = tt.addptr %arg5, %23 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %35 = arith.addi %28, %6 : i32
// CHECK-NEXT:       %36 = tt.make_tensor_ptr %34, [%c8192_i64, %c128_i64], [%c128_i64, %c1_i64], [%35, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %37 = tt.splat %28 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %38 = arith.addi %37, %12 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %39 = tt.load %29 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:       %40:5 = scf.for %arg9 = %c0_i32 to %28 step %c128_i32 iter_args(%arg10 = %cst_3, %arg11 = %cst, %arg12 = %cst_2, %arg13 = %31, %arg14 = %33) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
// CHECK-NEXT:         %58 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %60 = tt.dot %39, %59, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %61 = arith.mulf %60, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %62 = "tt.reduce"(%61) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %83 = arith.maximumf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %83 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %63 = arith.maximumf %arg12, %62 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %64 = tt.expand_dims %63 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %65 = tt.broadcast %64 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %66 = arith.subf %61, %65 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %67 = math.exp %66 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %68 = arith.truncf %67 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %69 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %70 = "tt.reduce"(%67) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:           %83 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:           tt.reduce.return %83 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %71 = arith.subf %arg12, %63 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %72 = math.exp %71 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %73 = arith.mulf %arg10, %72 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %74 = arith.addf %73, %70 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %75 = tt.expand_dims %72 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %76 = tt.broadcast %75 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %77 = arith.mulf %arg11, %76 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %extracted_slice_6 = tensor.extract_slice %68[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:         %78 = tensor.empty() : tensor<128x128xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_6 into %78[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xbf16> into tensor<128x128xbf16>
// CHECK-NEXT:         %79 = tt.dot %inserted_slice, %69, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %80 = arith.addf %79, %77 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         %81 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         %82 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         scf.yield %74, %80, %63, %81, %82 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %41 = arith.muli %14, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
// CHECK-NEXT:       %42 = arith.addi %14, %c1_i32 : i32
// CHECK-NEXT:       %43 = arith.muli %42, %c128_i32 : i32
// CHECK-NEXT:       %44 = tt.make_tensor_ptr %arg3, [%c8192_i64, %c8192_i64], [%c8192_i64, %c1_i64], [%28, %41] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
// CHECK-NEXT:       %45 = tt.advance %33, [%41, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:       %46 = tt.advance %31, [%41, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:       %47:6 = scf.for %arg9 = %41 to %43 step %c128_i32 iter_args(%arg10 = %44, %arg11 = %40#0, %arg12 = %40#1, %arg13 = %40#2, %arg14 = %46, %arg15 = %45) -> (!tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
// CHECK-NEXT:         %58 = tt.load %arg15 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %60 = tt.dot %39, %59, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %61 = tt.load %arg10 {DataUse} : !tt.ptr<tensor<128x128xf32>>
// CHECK-NEXT:         %62 = arith.mulf %60, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %63 = arith.cmpf une, %61, %cst {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %64 = arith.select %63, %cst_1, %cst {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
// CHECK-NEXT:         %65 = arith.addf %62, %64 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %88 = arith.maximumf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %88 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %67 = arith.maximumf %arg13, %66 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %68 = tt.expand_dims %67 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %69 = tt.broadcast %68 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %70 = arith.subf %65, %69 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.advance %arg10, [%c0_i32, %c128_i32] : <tensor<128x128xf32>>
// CHECK-NEXT:         %72 = math.exp %70 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %73 = arith.truncf %72 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %74 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:         %75 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:           %88 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:           tt.reduce.return %88 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %76 = arith.subf %arg13, %67 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %77 = math.exp %76 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %78 = arith.mulf %arg11, %77 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %79 = arith.addf %78, %75 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %80 = tt.expand_dims %77 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %81 = tt.broadcast %80 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %82 = arith.mulf %arg12, %81 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %extracted_slice_6 = tensor.extract_slice %73[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:         %83 = tensor.empty() : tensor<128x128xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_6 into %83[%3, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xbf16> into tensor<128x128xbf16>
// CHECK-NEXT:         %84 = tt.dot %inserted_slice, %74, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %85 = arith.addf %84, %82 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         %86 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         %87 = tt.advance %arg15, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
// CHECK-NEXT:         scf.yield %71, %79, %85, %67, %86, %87 : !tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %48 = math.log %47#1 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %49 = arith.addf %47#3, %48 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %50 = tt.expand_dims %47#1 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %51 = tt.broadcast %50 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %52 = arith.divf %47#2, %51 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %53 = arith.muli %13, %c8192_i32 : i32
// CHECK-NEXT:       %54 = tt.addptr %arg4, %53 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %55 = tt.splat %54 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %56 = tt.addptr %55, %38 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %49[%4] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %56[%4] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_4, %extracted_slice {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %57 = arith.truncf %52 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %57[%5, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %36, %extracted_slice_5 {tiled_op} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
    %cst_2 = arith.constant {DataUse} dense<5.000000e-01> : tensor<128x128xf32>
    %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
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
    %5 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
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
      %29 = tt.splat %21 {MetaUse} : i32 -> tensor<128xi32>
      %30 = arith.addi %29, %5 {MetaUse} : tensor<128xi32>
      %31 = tt.load %22 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
      %32:5 = scf.for %arg9 = %c0_i32 to %21 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %cst_3, %arg12 = %cst_0, %arg13 = %24, %arg14 = %26) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
        %50 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
        %51 = tt.trans %50 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %52 = tt.dot %31, %51, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %53 = arith.mulf %52, %cst_2 {DataUse} : tensor<128x128xf32>
        %54 = "tt.reduce"(%53) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %74 = arith.maximumf %arg15, %arg16 : f32
          tt.reduce.return %74 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %55 = arith.maximumf %arg12, %54 {DataUse} : tensor<128xf32>
        %56 = tt.expand_dims %55 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %57 = tt.broadcast %56 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %58 = arith.subf %53, %57 {DataUse} : tensor<128x128xf32>
        %59 = math.exp %58 {DataUse} : tensor<128x128xf32>
        %60 = arith.truncf %59 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
        %61 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
        %62 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
        ^bb0(%arg15: f32, %arg16: f32):
          %74 = arith.addf %arg15, %arg16 : f32
          tt.reduce.return %74 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %63 = arith.subf %arg12, %55 {DataUse} : tensor<128xf32>
        %64 = math.exp %63 {DataUse} : tensor<128xf32>
        %65 = arith.mulf %arg10, %64 {DataUse} : tensor<128xf32>
        %66 = arith.addf %65, %62 {DataUse} : tensor<128xf32>
        %67 = tt.expand_dims %64 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %68 = tt.broadcast %67 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %69 = arith.mulf %arg11, %68 {DataUse} : tensor<128x128xf32>
        %70 = tt.dot %60, %61, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %71 = arith.addf %70, %69 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
        %72 = tt.advance %arg13, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        %73 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        scf.yield %66, %71, %55, %72, %73 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %33 = arith.muli %7, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %34 = arith.addi %7, %c1_i32 : i32
      %35 = arith.muli %34, %c128_i32 : i32
      %36 = tt.make_tensor_ptr %arg3, [%c8192_i64, %c8192_i64], [%c8192_i64, %c1_i64], [%21, %33] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
      %37 = tt.advance %26, [%33, %c0_i32] : <tensor<128x128xbf16>>
      %38 = tt.advance %24, [%33, %c0_i32] : <tensor<128x128xbf16>>
      %39:6 = scf.for %arg9 = %33 to %35 step %c128_i32 iter_args(%arg10 = %36, %arg11 = %32#0, %arg12 = %32#1, %arg13 = %32#2, %arg14 = %38, %arg15 = %37) -> (!tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>)  : i32 {
        %50 = tt.load %arg15 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
        %51 = tt.trans %50 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %52 = tt.dot %31, %51, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %53 = tt.load %arg10 {DataUse} : !tt.ptr<tensor<128x128xf32>>
        %54 = arith.mulf %52, %cst_2 {DataUse} : tensor<128x128xf32>
        %55 = arith.cmpf une, %53, %cst_3 {DataUse} : tensor<128x128xf32>
        %56 = arith.select %55, %cst_1, %cst_3 {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
        %57 = arith.addf %54, %56 {DataUse} : tensor<128x128xf32>
        %58 = "tt.reduce"(%57) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %79 = arith.maximumf %arg16, %arg17 : f32
          tt.reduce.return %79 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %59 = arith.maximumf %arg13, %58 {DataUse} : tensor<128xf32>
        %60 = tt.expand_dims %59 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %61 = tt.broadcast %60 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %62 = arith.subf %57, %61 {DataUse} : tensor<128x128xf32>
        %63 = tt.advance %arg10, [%c0_i32, %c128_i32] : <tensor<128x128xf32>>
        %64 = math.exp %62 {DataUse} : tensor<128x128xf32>
        %65 = arith.truncf %64 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
        %66 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<128x128xbf16>>
        %67 = "tt.reduce"(%64) <{axis = 1 : i32}> ({
        ^bb0(%arg16: f32, %arg17: f32):
          %79 = arith.addf %arg16, %arg17 : f32
          tt.reduce.return %79 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %68 = arith.subf %arg13, %59 {DataUse} : tensor<128xf32>
        %69 = math.exp %68 {DataUse} : tensor<128xf32>
        %70 = arith.mulf %arg11, %69 {DataUse} : tensor<128xf32>
        %71 = arith.addf %70, %67 {DataUse} : tensor<128xf32>
        %72 = tt.expand_dims %69 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.mulf %arg12, %73 {DataUse} : tensor<128x128xf32>
        %75 = tt.dot %65, %66, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %76 = arith.addf %75, %74 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
        %77 = tt.advance %arg14, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        %78 = tt.advance %arg15, [%c128_i32, %c0_i32] : <tensor<128x128xbf16>>
        scf.yield %63, %71, %76, %59, %77, %78 : !tt.ptr<tensor<128x128xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xbf16>>, !tt.ptr<tensor<128x128xbf16>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %40 = math.log %39#1 {DataUse} : tensor<128xf32>
      %41 = arith.addf %39#3, %40 {DataUse} : tensor<128xf32>
      %42 = tt.expand_dims %39#1 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %43 = tt.broadcast %42 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %44 = arith.divf %39#2, %43 {DataUse} : tensor<128x128xf32>
      %45 = arith.muli %6, %c8192_i32 : i32
      %46 = tt.addptr %arg4, %45 : !tt.ptr<f32>, i32
      %47 = tt.splat %46 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %48 = tt.addptr %47, %30 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %48, %41 : tensor<128x!tt.ptr<f32>>
      %49 = arith.truncf %44 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %28, %49 : !tt.ptr<tensor<128x128xbf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_fwd_fp8/sep_md_false.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:     %6 = tt.get_program_id x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg9 = %6 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %10 = arith.divsi %8, %c8_i32 : i32
// CHECK-NEXT:       %11 = arith.remsi %8, %c8_i32 : i32
// CHECK-NEXT:       %12 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %13 = arith.muli %12, %c1048576_i64 : i64
// CHECK-NEXT:       %14 = arith.extsi %11 : i32 to i64
// CHECK-NEXT:       %15 = arith.muli %14, %c131072_i64 : i64
// CHECK-NEXT:       %16 = arith.addi %13, %15 : i64
// CHECK-NEXT:       %17 = arith.muli %12, %c64_i64 : i64
// CHECK-NEXT:       %18 = arith.muli %14, %c8_i64 : i64
// CHECK-NEXT:       %19 = arith.addi %17, %18 : i64
// CHECK-NEXT:       %20 = tt.addptr %arg0, %16 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %21 = arith.muli %9, %c128_i32 : i32
// CHECK-NEXT:       %22 = tt.make_tensor_ptr %20, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %23 = tt.addptr %arg2, %16 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %23, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %25 = tt.addptr %arg1, %16 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %26 = tt.make_tensor_ptr %25, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %27 = tt.addptr %arg6, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %28 = arith.divsi %21, %c128_i32 : i32
// CHECK-NEXT:       %29 = tt.make_tensor_ptr %27, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %30 = tt.addptr %arg7, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %31 = tt.make_tensor_ptr %30, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %32 = tt.addptr %arg8, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %33 = tt.make_tensor_ptr %32, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %34 = tt.addptr %arg4, %16 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %35 = arith.addi %21, %5 : i32
// CHECK-NEXT:       %36 = tt.make_tensor_ptr %34, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%35, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<64x128xf32>>
// CHECK-NEXT:       %37 = tt.splat %21 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %38 = arith.addi %37, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %39 = tt.load %22 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %40 = tt.load %29 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %41 = tt.broadcast %40 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %42:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %31, %arg12 = %33, %arg13 = %cst_2, %arg14 = %cst, %arg15 = %cst_1, %arg16 = %24, %arg17 = %26) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %52 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %53 = tt.trans %52 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %54 = tt.dot %39, %53, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %55 = tt.load %arg11 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %56 = arith.mulf %54, %41 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %57 = tt.broadcast %55 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %58 = arith.mulf %56, %57 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %59 = arith.mulf %58, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %60 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %86 = arith.maximumf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %86 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %61 = arith.maximumf %arg15, %60 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %62 = tt.expand_dims %61 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %63 = tt.broadcast %62 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %64 = arith.subf %59, %63 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %65 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %66 = math.exp %64 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %67 = tt.fp_to_fp %66 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %68 = tt.load %arg16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %extracted_slice_5 = tensor.extract_slice %67[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %69 = tensor.empty() : tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_5 into %69[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf8E4M3FN> into tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %70 = tt.dot %inserted_slice, %68, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %72 = tt.broadcast %71 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %73 = arith.mulf %70, %72 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %74 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %75 = "tt.reduce"(%66) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %86 = arith.addf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %86 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %76 = arith.subf %arg15, %61 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %77 = math.exp %76 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %78 = arith.mulf %arg13, %77 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %79 = arith.addf %78, %75 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %80 = tt.expand_dims %77 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %81 = tt.broadcast %80 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %82 = arith.mulf %arg14, %81 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %83 = arith.addf %82, %73 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %84 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %85 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %65, %74, %79, %83, %61, %84, %85 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %43 = math.log %42#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %44 = arith.addf %42#4, %43 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %45 = tt.expand_dims %42#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %46 = tt.broadcast %45 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %47 = arith.divf %42#3, %46 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %48 = arith.muli %8, %c1024_i32 : i32
// CHECK-NEXT:       %49 = tt.addptr %arg3, %48 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %50 = tt.splat %49 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %51 = tt.addptr %50, %38 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %44[%3] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %51[%3] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_3, %extracted_slice {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %47[%4, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:       tt.store %36, %extracted_slice_4 {tiled_op} : !tt.ptr<tensor<64x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
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
    %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
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
      %30 = tt.splat %15 {MetaUse} : i32 -> tensor<128xi32>
      %31 = arith.addi %30, %1 {MetaUse} : tensor<128xi32>
      %32 = tt.load %16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %33 = tt.load %23 {DataUse} : !tt.ptr<tensor<1x1xf32>>
      %34 = tt.broadcast %33 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
      %35:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %25, %arg12 = %27, %arg13 = %cst, %arg14 = %cst_2, %arg15 = %cst_0, %arg16 = %18, %arg17 = %20) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %45 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %46 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %47 = tt.dot %32, %46, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %48 = tt.load %arg11 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %49 = arith.mulf %47, %34 {DataUse} : tensor<128x128xf32>
        %50 = tt.broadcast %48 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %51 = arith.mulf %49, %50 {DataUse} : tensor<128x128xf32>
        %52 = arith.mulf %51, %cst_1 {DataUse} : tensor<128x128xf32>
        %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.maximumf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %54 = arith.maximumf %arg15, %53 {DataUse} : tensor<128xf32>
        %55 = tt.expand_dims %54 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %56 = tt.broadcast %55 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %57 = arith.subf %52, %56 {DataUse} : tensor<128x128xf32>
        %58 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %59 = math.exp %57 {DataUse} : tensor<128x128xf32>
        %60 = tt.fp_to_fp %59 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %61 = tt.load %arg16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %62 = tt.dot %60, %61, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %63 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %64 = tt.broadcast %63 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %65 = arith.mulf %62, %64 {DataUse} : tensor<128x128xf32>
        %66 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %67 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.addf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %68 = arith.subf %arg15, %54 {DataUse} : tensor<128xf32>
        %69 = math.exp %68 {DataUse} : tensor<128xf32>
        %70 = arith.mulf %arg13, %69 {DataUse} : tensor<128xf32>
        %71 = arith.addf %70, %67 {DataUse} : tensor<128xf32>
        %72 = tt.expand_dims %69 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.mulf %arg14, %73 {DataUse} : tensor<128x128xf32>
        %75 = arith.addf %74, %65 {DataUse} : tensor<128x128xf32>
        %76 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %77 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %58, %66, %71, %75, %54, %76, %77 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %36 = math.log %35#2 {DataUse} : tensor<128xf32>
      %37 = arith.addf %35#4, %36 {DataUse} : tensor<128xf32>
      %38 = tt.expand_dims %35#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %39 = tt.broadcast %38 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %40 = arith.divf %35#3, %39 {DataUse} : tensor<128x128xf32>
      %41 = arith.muli %2, %c1024_i32 : i32
      %42 = tt.addptr %arg3, %41 : !tt.ptr<f32>, i32
      %43 = tt.splat %42 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %44 = tt.addptr %43, %31 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %44, %37 : tensor<128x!tt.ptr<f32>>
      tt.store %29, %40 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_fwd_fp8/sep_md_true.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = arith.index_cast %5 : index to i32
// CHECK-NEXT:     %7 = tt.get_program_id x : i32
// CHECK-NEXT:     %8 = tt.addptr %arg6, %7 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %9 = tt.load %8 : !tt.ptr<i32>
// CHECK-NEXT:     %10 = tt.addptr %8, %c1_i32 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %11 = tt.load %10 : !tt.ptr<i32>
// CHECK-NEXT:     %12 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %13 = tt.bitcast %arg3 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK-NEXT:     scf.for %arg10 = %9 to %11 step %c1_i32  : i32 {
// CHECK-NEXT:       %14 = arith.divsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %15 = arith.remsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %16 = arith.divsi %14, %c8_i32 : i32
// CHECK-NEXT:       %17 = arith.remsi %14, %c8_i32 : i32
// CHECK-NEXT:       %18 = arith.extsi %16 : i32 to i64
// CHECK-NEXT:       %19 = arith.muli %18, %c1048576_i64 : i64
// CHECK-NEXT:       %20 = arith.extsi %17 : i32 to i64
// CHECK-NEXT:       %21 = arith.muli %20, %c131072_i64 : i64
// CHECK-NEXT:       %22 = arith.addi %19, %21 : i64
// CHECK-NEXT:       %23 = arith.muli %18, %c64_i64 : i64
// CHECK-NEXT:       %24 = arith.muli %20, %c8_i64 : i64
// CHECK-NEXT:       %25 = arith.addi %23, %24 : i64
// CHECK-NEXT:       %26 = tt.addptr %arg0, %22 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %27 = arith.muli %15, %c128_i32 : i32
// CHECK-NEXT:       %28 = tt.make_tensor_ptr %26, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %29 = tt.addptr %arg2, %22 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %30 = tt.make_tensor_ptr %29, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %31 = tt.addptr %arg1, %22 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %32 = tt.make_tensor_ptr %31, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %33 = tt.addptr %arg7, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %34 = arith.divsi %27, %c128_i32 : i32
// CHECK-NEXT:       %35 = tt.make_tensor_ptr %33, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%34, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %36 = tt.addptr %arg8, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %37 = tt.make_tensor_ptr %36, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %38 = tt.addptr %arg9, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %39 = tt.make_tensor_ptr %38, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %40 = tt.addptr %arg5, %22 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %41 = arith.addi %27, %6 : i32
// CHECK-NEXT:       %42 = tt.make_tensor_ptr %40, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%41, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<64x128xf32>>
// CHECK-NEXT:       %43 = tt.splat %27 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %44 = arith.addi %43, %12 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %45 = tt.load %28 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %46 = tt.load %35 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %47 = tt.broadcast %46 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %48:7 = scf.for %arg11 = %c0_i32 to %27 step %c128_i32 iter_args(%arg12 = %37, %arg13 = %39, %arg14 = %cst_4, %arg15 = %cst, %arg16 = %cst_3, %arg17 = %30, %arg18 = %32) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %66 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %67 = tt.trans %66 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %68 = tt.dot %45, %67, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %69 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %70 = arith.mulf %68, %47 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.broadcast %69 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %72 = arith.mulf %70, %71 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %73 = arith.mulf %72, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %74 = "tt.reduce"(%73) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %100 = arith.maximumf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %100 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %75 = arith.maximumf %arg16, %74 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %76 = tt.expand_dims %75 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %77 = tt.broadcast %76 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %78 = arith.subf %73, %77 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %79 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %80 = math.exp %78 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %81 = tt.fp_to_fp %80 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %82 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %81[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %83 = tensor.empty() : tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_7 into %83[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf8E4M3FN> into tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %84 = tt.dot %inserted_slice, %82, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %85 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %86 = tt.broadcast %85 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %87 = arith.mulf %84, %86 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %88 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %89 = "tt.reduce"(%80) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %100 = arith.addf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %100 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %90 = arith.subf %arg16, %75 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %91 = math.exp %90 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %92 = arith.mulf %arg14, %91 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %93 = arith.addf %92, %89 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %94 = tt.expand_dims %91 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %95 = tt.broadcast %94 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %96 = arith.mulf %arg15, %95 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %97 = arith.addf %96, %87 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %98 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %99 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %79, %88, %93, %97, %75, %98, %99 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %49 = arith.muli %15, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
// CHECK-NEXT:       %50 = arith.addi %15, %c1_i32 : i32
// CHECK-NEXT:       %51 = arith.muli %50, %c128_i32 : i32
// CHECK-NEXT:       %52 = tt.make_tensor_ptr %13, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%27, %49] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
// CHECK-NEXT:       %53 = tt.advance %32, [%49, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %54 = tt.advance %30, [%49, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %55 = tt.broadcast %46 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %56:8 = scf.for %arg11 = %49 to %51 step %c128_i32 iter_args(%arg12 = %52, %arg13 = %37, %arg14 = %39, %arg15 = %48#2, %arg16 = %48#3, %arg17 = %48#4, %arg18 = %54, %arg19 = %53) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %66 = tt.load %arg19 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %67 = tt.trans %66 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %68 = tt.dot %45, %67, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %69 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %70 = arith.mulf %68, %55 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.broadcast %69 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %72 = arith.mulf %70, %71 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %73 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<128x128xi8>>
// CHECK-NEXT:         %74 = arith.mulf %72, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %75 = arith.cmpi ne, %73, %cst_2 {DataUse} : tensor<128x128xi8>
// CHECK-NEXT:         %76 = arith.select %75, %cst_1, %cst {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
// CHECK-NEXT:         %77 = arith.addf %74, %76 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %78 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %105 = arith.maximumf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %79 = arith.maximumf %arg17, %78 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %80 = tt.expand_dims %79 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %81 = tt.broadcast %80 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %82 = arith.subf %77, %81 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %83 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
// CHECK-NEXT:         %84 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %85 = math.exp %82 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %86 = tt.fp_to_fp %85 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %87 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %86[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %88 = tensor.empty() : tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_7 into %88[%3, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf8E4M3FN> into tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %89 = tt.dot %inserted_slice, %87, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %90 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %91 = tt.broadcast %90 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.mulf %89, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %94 = "tt.reduce"(%85) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %105 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %95 = arith.subf %arg17, %79 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %96 = math.exp %95 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %97 = arith.mulf %arg15, %96 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %98 = arith.addf %97, %94 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %99 = tt.expand_dims %96 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %101 = arith.mulf %arg16, %100 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %102 = arith.addf %101, %92 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %103 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %104 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %83, %84, %93, %98, %102, %79, %103, %104 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %57 = math.log %56#3 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %58 = arith.addf %56#5, %57 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %59 = tt.expand_dims %56#3 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %60 = tt.broadcast %59 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %61 = arith.divf %56#4, %60 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %62 = arith.muli %14, %c1024_i32 : i32
// CHECK-NEXT:       %63 = tt.addptr %arg4, %62 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %64 = tt.splat %63 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %65 = tt.addptr %64, %44 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %58[%4] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %65[%4] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_5, %extracted_slice {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %extracted_slice_6 = tensor.extract_slice %61[%5, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:       tt.store %42, %extracted_slice_6 {tiled_op} : !tt.ptr<tensor<64x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant {DataUse} dense<0> : tensor<128x128xi8>
    %cst_2 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
    %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
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
    %5 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
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
      %35 = tt.splat %20 {MetaUse} : i32 -> tensor<128xi32>
      %36 = arith.addi %35, %5 {MetaUse} : tensor<128xi32>
      %37 = tt.load %21 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %38 = tt.load %28 {DataUse} : !tt.ptr<tensor<1x1xf32>>
      %39 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
      %40:7 = scf.for %arg11 = %c0_i32 to %20 step %c128_i32 iter_args(%arg12 = %30, %arg13 = %32, %arg14 = %cst, %arg15 = %cst_4, %arg16 = %cst_0, %arg17 = %23, %arg18 = %25) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %39 {DataUse} : tensor<128x128xf32>
        %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
        %65 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
        %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.maximumf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %67 = arith.maximumf %arg16, %66 {DataUse} : tensor<128xf32>
        %68 = tt.expand_dims %67 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %69 = tt.broadcast %68 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %70 = arith.subf %65, %69 {DataUse} : tensor<128x128xf32>
        %71 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %72 = math.exp %70 {DataUse} : tensor<128x128xf32>
        %73 = tt.fp_to_fp %72 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %74 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %75 = tt.dot %73, %74, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %76 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %77 = tt.broadcast %76 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %78 = arith.mulf %75, %77 {DataUse} : tensor<128x128xf32>
        %79 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %80 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.addf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %81 = arith.subf %arg16, %67 {DataUse} : tensor<128xf32>
        %82 = math.exp %81 {DataUse} : tensor<128xf32>
        %83 = arith.mulf %arg14, %82 {DataUse} : tensor<128xf32>
        %84 = arith.addf %83, %80 {DataUse} : tensor<128xf32>
        %85 = tt.expand_dims %82 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %86 = tt.broadcast %85 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %87 = arith.mulf %arg15, %86 {DataUse} : tensor<128x128xf32>
        %88 = arith.addf %87, %78 {DataUse} : tensor<128x128xf32>
        %89 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %90 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %71, %79, %84, %88, %67, %89, %90 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %41 = arith.muli %8, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %42 = arith.addi %8, %c1_i32 : i32
      %43 = arith.muli %42, %c128_i32 : i32
      %44 = tt.make_tensor_ptr %6, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%20, %41] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
      %45 = tt.advance %25, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %46 = tt.advance %23, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %47 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
      %48:8 = scf.for %arg11 = %41 to %43 step %c128_i32 iter_args(%arg12 = %44, %arg13 = %30, %arg14 = %32, %arg15 = %40#2, %arg16 = %40#3, %arg17 = %40#4, %arg18 = %46, %arg19 = %45) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg19 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %47 {DataUse} : tensor<128x128xf32>
        %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
        %65 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<128x128xi8>>
        %66 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
        %67 = arith.cmpi ne, %65, %cst_1 {DataUse} : tensor<128x128xi8>
        %68 = arith.select %67, %cst_2, %cst_4 {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
        %69 = arith.addf %66, %68 {DataUse} : tensor<128x128xf32>
        %70 = "tt.reduce"(%69) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.maximumf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %71 = arith.maximumf %arg17, %70 {DataUse} : tensor<128xf32>
        %72 = tt.expand_dims %71 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.subf %69, %73 {DataUse} : tensor<128x128xf32>
        %75 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
        %76 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %77 = math.exp %74 {DataUse} : tensor<128x128xf32>
        %78 = tt.fp_to_fp %77 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %79 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %80 = tt.dot %78, %79, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %81 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %82 = tt.broadcast %81 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %83 = arith.mulf %80, %82 {DataUse} : tensor<128x128xf32>
        %84 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %85 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.addf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %86 = arith.subf %arg17, %71 {DataUse} : tensor<128xf32>
        %87 = math.exp %86 {DataUse} : tensor<128xf32>
        %88 = arith.mulf %arg15, %87 {DataUse} : tensor<128xf32>
        %89 = arith.addf %88, %85 {DataUse} : tensor<128xf32>
        %90 = tt.expand_dims %87 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.mulf %arg16, %91 {DataUse} : tensor<128x128xf32>
        %93 = arith.addf %92, %83 {DataUse} : tensor<128x128xf32>
        %94 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %95 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %75, %76, %84, %89, %93, %71, %94, %95 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %49 = math.log %48#3 {DataUse} : tensor<128xf32>
      %50 = arith.addf %48#5, %49 {DataUse} : tensor<128xf32>
      %51 = tt.expand_dims %48#3 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %52 = tt.broadcast %51 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %53 = arith.divf %48#4, %52 {DataUse} : tensor<128x128xf32>
      %54 = arith.muli %7, %c1024_i32 : i32
      %55 = tt.addptr %arg4, %54 : !tt.ptr<f32>, i32
      %56 = tt.splat %55 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %57 = tt.addptr %56, %36 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %57, %50 : tensor<128x!tt.ptr<f32>>
      tt.store %34, %53 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_fwd_fp8_Q2/sep_md_false.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c8192_i32 = arith.constant 8192 : i32
// CHECK-NEXT:     %c28_i32 = arith.constant 28 : i32
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:     %6 = tt.get_program_id x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     scf.for %arg9 = %6 to %c8192_i32 step %c28_i32  : i32 {
// CHECK-NEXT:       %8 = arith.divsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %9 = arith.remsi %arg9, %c8_i32 : i32
// CHECK-NEXT:       %10 = arith.divsi %8, %c8_i32 : i32
// CHECK-NEXT:       %11 = arith.remsi %8, %c8_i32 : i32
// CHECK-NEXT:       %12 = arith.extsi %10 : i32 to i64
// CHECK-NEXT:       %13 = arith.muli %12, %c1048576_i64 : i64
// CHECK-NEXT:       %14 = arith.extsi %11 : i32 to i64
// CHECK-NEXT:       %15 = arith.muli %14, %c131072_i64 : i64
// CHECK-NEXT:       %16 = arith.addi %13, %15 : i64
// CHECK-NEXT:       %17 = arith.muli %12, %c64_i64 : i64
// CHECK-NEXT:       %18 = arith.muli %14, %c8_i64 : i64
// CHECK-NEXT:       %19 = arith.addi %17, %18 : i64
// CHECK-NEXT:       %20 = tt.addptr %arg0, %16 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %21 = arith.muli %9, %c128_i32 : i32
// CHECK-NEXT:       %22 = tt.make_tensor_ptr %20, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %23 = tt.addptr %arg2, %16 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %24 = tt.make_tensor_ptr %23, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %25 = tt.addptr %arg1, %16 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %26 = tt.make_tensor_ptr %25, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %27 = tt.addptr %arg6, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %28 = arith.divsi %21, %c128_i32 : i32
// CHECK-NEXT:       %29 = tt.make_tensor_ptr %27, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%28, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %30 = tt.addptr %arg7, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %31 = tt.make_tensor_ptr %30, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %32 = tt.addptr %arg8, %19 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %33 = tt.make_tensor_ptr %32, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %34 = tt.addptr %arg4, %16 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %35 = arith.addi %21, %5 : i32
// CHECK-NEXT:       %36 = tt.make_tensor_ptr %34, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%35, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<64x128xf32>>
// CHECK-NEXT:       %37 = tt.splat %21 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %38 = arith.addi %37, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %39 = tt.load %22 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %40 = tt.load %29 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %41 = tt.broadcast %40 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %42:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %31, %arg12 = %33, %arg13 = %cst_2, %arg14 = %cst, %arg15 = %cst_1, %arg16 = %24, %arg17 = %26) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %52 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %53 = tt.trans %52 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %54 = tt.dot %39, %53, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %55 = tt.load %arg11 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %56 = arith.mulf %54, %41 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %57 = tt.broadcast %55 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %58 = arith.mulf %56, %57 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %59 = arith.mulf %58, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %60 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %86 = arith.maximumf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %86 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %61 = arith.maximumf %arg15, %60 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %62 = tt.expand_dims %61 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %63 = tt.broadcast %62 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %64 = arith.subf %59, %63 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %65 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %66 = math.exp %64 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %67 = tt.fp_to_fp %66 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %68 = tt.load %arg16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %extracted_slice_5 = tensor.extract_slice %67[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %69 = tensor.empty() : tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_5 into %69[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf8E4M3FN> into tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %70 = tt.dot %inserted_slice, %68, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %72 = tt.broadcast %71 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %73 = arith.mulf %70, %72 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %74 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %75 = "tt.reduce"(%66) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg18: f32, %arg19: f32):
// CHECK-NEXT:           %86 = arith.addf %arg18, %arg19 : f32
// CHECK-NEXT:           tt.reduce.return %86 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %76 = arith.subf %arg15, %61 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %77 = math.exp %76 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %78 = arith.mulf %arg13, %77 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %79 = arith.addf %78, %75 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %80 = tt.expand_dims %77 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %81 = tt.broadcast %80 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %82 = arith.mulf %arg14, %81 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %83 = arith.addf %82, %73 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %84 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %85 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %65, %74, %79, %83, %61, %84, %85 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %43 = math.log %42#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %44 = arith.addf %42#4, %43 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %45 = tt.expand_dims %42#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %46 = tt.broadcast %45 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %47 = arith.divf %42#3, %46 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %48 = arith.muli %8, %c1024_i32 : i32
// CHECK-NEXT:       %49 = tt.addptr %arg3, %48 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %50 = tt.splat %49 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %51 = tt.addptr %50, %38 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %44[%3] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_3 = tensor.extract_slice %51[%3] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_3, %extracted_slice {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %extracted_slice_4 = tensor.extract_slice %47[%4, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:       tt.store %36, %extracted_slice_4 {tiled_op} : !tt.ptr<tensor<64x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
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
    %1 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
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
      %30 = tt.splat %15 {MetaUse} : i32 -> tensor<128xi32>
      %31 = arith.addi %30, %1 {MetaUse} : tensor<128xi32>
      %32 = tt.load %16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %33 = tt.load %23 {DataUse} : !tt.ptr<tensor<1x1xf32>>
      %34 = tt.broadcast %33 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
      %35:7 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %25, %arg12 = %27, %arg13 = %cst, %arg14 = %cst_2, %arg15 = %cst_0, %arg16 = %18, %arg17 = %20) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %45 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %46 = tt.trans %45 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %47 = tt.dot %32, %46, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %48 = tt.load %arg11 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %49 = arith.mulf %47, %34 {DataUse} : tensor<128x128xf32>
        %50 = tt.broadcast %48 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %51 = arith.mulf %49, %50 {DataUse} : tensor<128x128xf32>
        %52 = arith.mulf %51, %cst_1 {DataUse} : tensor<128x128xf32>
        %53 = "tt.reduce"(%52) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.maximumf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %54 = arith.maximumf %arg15, %53 {DataUse} : tensor<128xf32>
        %55 = tt.expand_dims %54 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %56 = tt.broadcast %55 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %57 = arith.subf %52, %56 {DataUse} : tensor<128x128xf32>
        %58 = tt.advance %arg11, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %59 = math.exp %57 {DataUse} : tensor<128x128xf32>
        %60 = tt.fp_to_fp %59 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %61 = tt.load %arg16 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %62 = tt.dot %60, %61, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %63 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %64 = tt.broadcast %63 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %65 = arith.mulf %62, %64 {DataUse} : tensor<128x128xf32>
        %66 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %67 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
        ^bb0(%arg18: f32, %arg19: f32):
          %78 = arith.addf %arg18, %arg19 : f32
          tt.reduce.return %78 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %68 = arith.subf %arg15, %54 {DataUse} : tensor<128xf32>
        %69 = math.exp %68 {DataUse} : tensor<128xf32>
        %70 = arith.mulf %arg13, %69 {DataUse} : tensor<128xf32>
        %71 = arith.addf %70, %67 {DataUse} : tensor<128xf32>
        %72 = tt.expand_dims %69 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.mulf %arg14, %73 {DataUse} : tensor<128x128xf32>
        %75 = arith.addf %74, %65 {DataUse} : tensor<128x128xf32>
        %76 = tt.advance %arg16, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %77 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %58, %66, %71, %75, %54, %76, %77 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %36 = math.log %35#2 {DataUse} : tensor<128xf32>
      %37 = arith.addf %35#4, %36 {DataUse} : tensor<128xf32>
      %38 = tt.expand_dims %35#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %39 = tt.broadcast %38 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %40 = arith.divf %35#3, %39 {DataUse} : tensor<128x128xf32>
      %41 = arith.muli %2, %c1024_i32 : i32
      %42 = tt.addptr %arg3, %41 : !tt.ptr<f32>, i32
      %43 = tt.splat %42 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %44 = tt.addptr %43, %31 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %44, %37 : tensor<128x!tt.ptr<f32>>
      tt.store %29, %40 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/fa_fwd_fp8_Q2/sep_md_true.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c1024_i32 = arith.constant 1024 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:     %c1048576_i64 = arith.constant 1048576 : i64
// CHECK-NEXT:     %c131072_i64 = arith.constant 131072 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c1024_i64 = arith.constant 1024 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = arith.index_cast %5 : index to i32
// CHECK-NEXT:     %7 = tt.get_program_id x : i32
// CHECK-NEXT:     %8 = tt.addptr %arg6, %7 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %9 = tt.load %8 : !tt.ptr<i32>
// CHECK-NEXT:     %10 = tt.addptr %8, %c1_i32 : !tt.ptr<i32>, i32
// CHECK-NEXT:     %11 = tt.load %10 : !tt.ptr<i32>
// CHECK-NEXT:     %12 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %13 = tt.bitcast %arg3 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK-NEXT:     scf.for %arg10 = %9 to %11 step %c1_i32  : i32 {
// CHECK-NEXT:       %14 = arith.divsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %15 = arith.remsi %arg10, %c8_i32 : i32
// CHECK-NEXT:       %16 = arith.divsi %14, %c8_i32 : i32
// CHECK-NEXT:       %17 = arith.remsi %14, %c8_i32 : i32
// CHECK-NEXT:       %18 = arith.extsi %16 : i32 to i64
// CHECK-NEXT:       %19 = arith.muli %18, %c1048576_i64 : i64
// CHECK-NEXT:       %20 = arith.extsi %17 : i32 to i64
// CHECK-NEXT:       %21 = arith.muli %20, %c131072_i64 : i64
// CHECK-NEXT:       %22 = arith.addi %19, %21 : i64
// CHECK-NEXT:       %23 = arith.muli %18, %c64_i64 : i64
// CHECK-NEXT:       %24 = arith.muli %20, %c8_i64 : i64
// CHECK-NEXT:       %25 = arith.addi %23, %24 : i64
// CHECK-NEXT:       %26 = tt.addptr %arg0, %22 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %27 = arith.muli %15, %c128_i32 : i32
// CHECK-NEXT:       %28 = tt.make_tensor_ptr %26, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%27, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %29 = tt.addptr %arg2, %22 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %30 = tt.make_tensor_ptr %29, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %31 = tt.addptr %arg1, %22 : !tt.ptr<f8E4M3FN>, i64
// CHECK-NEXT:       %32 = tt.make_tensor_ptr %31, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %33 = tt.addptr %arg7, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %34 = arith.divsi %27, %c128_i32 : i32
// CHECK-NEXT:       %35 = tt.make_tensor_ptr %33, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%34, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %36 = tt.addptr %arg8, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %37 = tt.make_tensor_ptr %36, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %38 = tt.addptr %arg9, %25 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %39 = tt.make_tensor_ptr %38, [%c8_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1x1xf32>>
// CHECK-NEXT:       %40 = tt.addptr %arg5, %22 : !tt.ptr<f32>, i64
// CHECK-NEXT:       %41 = arith.addi %27, %6 : i32
// CHECK-NEXT:       %42 = tt.make_tensor_ptr %40, [%c1024_i64, %c128_i64], [%c128_i64, %c1_i64], [%41, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<64x128xf32>>
// CHECK-NEXT:       %43 = tt.splat %27 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %44 = arith.addi %43, %12 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %45 = tt.load %28 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %46 = tt.load %35 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:       %47 = tt.broadcast %46 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %48:7 = scf.for %arg11 = %c0_i32 to %27 step %c128_i32 iter_args(%arg12 = %37, %arg13 = %39, %arg14 = %cst_4, %arg15 = %cst, %arg16 = %cst_3, %arg17 = %30, %arg18 = %32) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %66 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %67 = tt.trans %66 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %68 = tt.dot %45, %67, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %69 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %70 = arith.mulf %68, %47 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.broadcast %69 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %72 = arith.mulf %70, %71 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %73 = arith.mulf %72, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %74 = "tt.reduce"(%73) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %100 = arith.maximumf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %100 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %75 = arith.maximumf %arg16, %74 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %76 = tt.expand_dims %75 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %77 = tt.broadcast %76 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %78 = arith.subf %73, %77 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %79 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %80 = math.exp %78 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %81 = tt.fp_to_fp %80 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %82 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %81[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %83 = tensor.empty() : tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_7 into %83[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf8E4M3FN> into tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %84 = tt.dot %inserted_slice, %82, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %85 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %86 = tt.broadcast %85 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %87 = arith.mulf %84, %86 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %88 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %89 = "tt.reduce"(%80) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg19: f32, %arg20: f32):
// CHECK-NEXT:           %100 = arith.addf %arg19, %arg20 : f32
// CHECK-NEXT:           tt.reduce.return %100 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %90 = arith.subf %arg16, %75 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %91 = math.exp %90 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %92 = arith.mulf %arg14, %91 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %93 = arith.addf %92, %89 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %94 = tt.expand_dims %91 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %95 = tt.broadcast %94 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %96 = arith.mulf %arg15, %95 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %97 = arith.addf %96, %87 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %98 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %99 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %79, %88, %93, %97, %75, %98, %99 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %49 = arith.muli %15, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
// CHECK-NEXT:       %50 = arith.addi %15, %c1_i32 : i32
// CHECK-NEXT:       %51 = arith.muli %50, %c128_i32 : i32
// CHECK-NEXT:       %52 = tt.make_tensor_ptr %13, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%27, %49] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
// CHECK-NEXT:       %53 = tt.advance %32, [%49, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %54 = tt.advance %30, [%49, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       %55 = tt.broadcast %46 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %56:8 = scf.for %arg11 = %49 to %51 step %c128_i32 iter_args(%arg12 = %52, %arg13 = %37, %arg14 = %39, %arg15 = %48#2, %arg16 = %48#3, %arg17 = %48#4, %arg18 = %54, %arg19 = %53) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
// CHECK-NEXT:         %66 = tt.load %arg19 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %67 = tt.trans %66 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %68 = tt.dot %45, %67, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %69 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %70 = arith.mulf %68, %55 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %71 = tt.broadcast %69 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %72 = arith.mulf %70, %71 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %73 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<128x128xi8>>
// CHECK-NEXT:         %74 = arith.mulf %72, %cst_0 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %75 = arith.cmpi ne, %73, %cst_2 {DataUse} : tensor<128x128xi8>
// CHECK-NEXT:         %76 = arith.select %75, %cst_1, %cst {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
// CHECK-NEXT:         %77 = arith.addf %74, %76 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %78 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %105 = arith.maximumf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %79 = arith.maximumf %arg17, %78 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %80 = tt.expand_dims %79 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %81 = tt.broadcast %80 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %82 = arith.subf %77, %81 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %83 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
// CHECK-NEXT:         %84 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %85 = math.exp %82 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %86 = tt.fp_to_fp %85 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %87 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %86[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf8E4M3FN> to tensor<64x128xf8E4M3FN>
// CHECK-NEXT:         %88 = tensor.empty() : tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_7 into %88[%3, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xf8E4M3FN> into tensor<128x128xf8E4M3FN>
// CHECK-NEXT:         %89 = tt.dot %inserted_slice, %87, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
// CHECK-NEXT:         %90 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<1x1xf32>>
// CHECK-NEXT:         %91 = tt.broadcast %90 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.mulf %89, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
// CHECK-NEXT:         %94 = "tt.reduce"(%85) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg20: f32, %arg21: f32):
// CHECK-NEXT:           %105 = arith.addf %arg20, %arg21 : f32
// CHECK-NEXT:           tt.reduce.return %105 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %95 = arith.subf %arg17, %79 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %96 = math.exp %95 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %97 = arith.mulf %arg15, %96 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %98 = arith.addf %97, %94 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %99 = tt.expand_dims %96 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %100 = tt.broadcast %99 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %101 = arith.mulf %arg16, %100 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %102 = arith.addf %101, %92 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %103 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         %104 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:         scf.yield %83, %84, %93, %98, %102, %79, %103, %104 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
// CHECK-NEXT:       } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
// CHECK-NEXT:       %57 = math.log %56#3 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %58 = arith.addf %56#5, %57 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %59 = tt.expand_dims %56#3 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %60 = tt.broadcast %59 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %61 = arith.divf %56#4, %60 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %62 = arith.muli %14, %c1024_i32 : i32
// CHECK-NEXT:       %63 = tt.addptr %arg4, %62 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %64 = tt.splat %63 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %65 = tt.addptr %64, %44 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %58[%4] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %65[%4] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       tt.store %extracted_slice_5, %extracted_slice {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %extracted_slice_6 = tensor.extract_slice %61[%5, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xf32> to tensor<64x128xf32>
// CHECK-NEXT:       tt.store %42, %extracted_slice_6 {tiled_op} : !tt.ptr<tensor<64x128xf32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_attn_fwd_fp8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<128xf32>
    %cst_1 = arith.constant {DataUse} dense<0> : tensor<128x128xi8>
    %cst_2 = arith.constant {DataUse} dense<-1.000000e+04> : tensor<128x128xf32>
    %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
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
    %5 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
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
      %35 = tt.splat %20 {MetaUse} : i32 -> tensor<128xi32>
      %36 = arith.addi %35, %5 {MetaUse} : tensor<128xi32>
      %37 = tt.load %21 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
      %38 = tt.load %28 {DataUse} : !tt.ptr<tensor<1x1xf32>>
      %39 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
      %40:7 = scf.for %arg11 = %c0_i32 to %20 step %c128_i32 iter_args(%arg12 = %30, %arg13 = %32, %arg14 = %cst, %arg15 = %cst_4, %arg16 = %cst_0, %arg17 = %23, %arg18 = %25) -> (!tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %39 {DataUse} : tensor<128x128xf32>
        %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
        %65 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
        %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.maximumf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %67 = arith.maximumf %arg16, %66 {DataUse} : tensor<128xf32>
        %68 = tt.expand_dims %67 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %69 = tt.broadcast %68 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %70 = arith.subf %65, %69 {DataUse} : tensor<128x128xf32>
        %71 = tt.advance %arg12, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %72 = math.exp %70 {DataUse} : tensor<128x128xf32>
        %73 = tt.fp_to_fp %72 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %74 = tt.load %arg17 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %75 = tt.dot %73, %74, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %76 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %77 = tt.broadcast %76 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %78 = arith.mulf %75, %77 {DataUse} : tensor<128x128xf32>
        %79 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %80 = "tt.reduce"(%72) <{axis = 1 : i32}> ({
        ^bb0(%arg19: f32, %arg20: f32):
          %91 = arith.addf %arg19, %arg20 : f32
          tt.reduce.return %91 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %81 = arith.subf %arg16, %67 {DataUse} : tensor<128xf32>
        %82 = math.exp %81 {DataUse} : tensor<128xf32>
        %83 = arith.mulf %arg14, %82 {DataUse} : tensor<128xf32>
        %84 = arith.addf %83, %80 {DataUse} : tensor<128xf32>
        %85 = tt.expand_dims %82 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %86 = tt.broadcast %85 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %87 = arith.mulf %arg15, %86 {DataUse} : tensor<128x128xf32>
        %88 = arith.addf %87, %78 {DataUse} : tensor<128x128xf32>
        %89 = tt.advance %arg17, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %90 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %71, %79, %84, %88, %67, %89, %90 : !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %41 = arith.muli %8, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
      %42 = arith.addi %8, %c1_i32 : i32
      %43 = arith.muli %42, %c128_i32 : i32
      %44 = tt.make_tensor_ptr %6, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%20, %41] {order = array<i32: 1, 0>} : <tensor<128x128xi8>>
      %45 = tt.advance %25, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %46 = tt.advance %23, [%41, %c0_i32] : <tensor<128x128xf8E4M3FN>>
      %47 = tt.broadcast %38 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
      %48:8 = scf.for %arg11 = %41 to %43 step %c128_i32 iter_args(%arg12 = %44, %arg13 = %30, %arg14 = %32, %arg15 = %40#2, %arg16 = %40#3, %arg17 = %40#4, %arg18 = %46, %arg19 = %45) -> (!tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>)  : i32 {
        %58 = tt.load %arg19 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %59 = tt.trans %58 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xf8E4M3FN> -> tensor<128x128xf8E4M3FN>
        %60 = tt.dot %37, %59, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %61 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %62 = arith.mulf %60, %47 {DataUse} : tensor<128x128xf32>
        %63 = tt.broadcast %61 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %64 = arith.mulf %62, %63 {DataUse} : tensor<128x128xf32>
        %65 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<128x128xi8>>
        %66 = arith.mulf %64, %cst_3 {DataUse} : tensor<128x128xf32>
        %67 = arith.cmpi ne, %65, %cst_1 {DataUse} : tensor<128x128xi8>
        %68 = arith.select %67, %cst_2, %cst_4 {DataUse} : tensor<128x128xi1>, tensor<128x128xf32>
        %69 = arith.addf %66, %68 {DataUse} : tensor<128x128xf32>
        %70 = "tt.reduce"(%69) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.maximumf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %71 = arith.maximumf %arg17, %70 {DataUse} : tensor<128xf32>
        %72 = tt.expand_dims %71 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %73 = tt.broadcast %72 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %74 = arith.subf %69, %73 {DataUse} : tensor<128x128xf32>
        %75 = tt.advance %arg12, [%c0_i32, %c128_i32] : <tensor<128x128xi8>>
        %76 = tt.advance %arg13, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %77 = math.exp %74 {DataUse} : tensor<128x128xf32>
        %78 = tt.fp_to_fp %77 {DataUse}, rounding = rtne : tensor<128x128xf32> -> tensor<128x128xf8E4M3FN>
        %79 = tt.load %arg18 {DataUse} : !tt.ptr<tensor<128x128xf8E4M3FN>>
        %80 = tt.dot %78, %79, %cst_4 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xf8E4M3FN> * tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
        %81 = tt.load %arg14 {DataUse} : !tt.ptr<tensor<1x1xf32>>
        %82 = tt.broadcast %81 {DataUse} : tensor<1x1xf32> -> tensor<128x128xf32>
        %83 = arith.mulf %80, %82 {DataUse} : tensor<128x128xf32>
        %84 = tt.advance %arg14, [%c1_i32, %c0_i32] : <tensor<1x1xf32>>
        %85 = "tt.reduce"(%77) <{axis = 1 : i32}> ({
        ^bb0(%arg20: f32, %arg21: f32):
          %96 = arith.addf %arg20, %arg21 : f32
          tt.reduce.return %96 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %86 = arith.subf %arg17, %71 {DataUse} : tensor<128xf32>
        %87 = math.exp %86 {DataUse} : tensor<128xf32>
        %88 = arith.mulf %arg15, %87 {DataUse} : tensor<128xf32>
        %89 = arith.addf %88, %85 {DataUse} : tensor<128xf32>
        %90 = tt.expand_dims %87 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.mulf %arg16, %91 {DataUse} : tensor<128x128xf32>
        %93 = arith.addf %92, %83 {DataUse} : tensor<128x128xf32>
        %94 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        %95 = tt.advance %arg19, [%c128_i32, %c0_i32] : <tensor<128x128xf8E4M3FN>>
        scf.yield %75, %76, %84, %89, %93, %71, %94, %95 : !tt.ptr<tensor<128x128xi8>>, !tt.ptr<tensor<1x1xf32>>, !tt.ptr<tensor<1x1xf32>>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, !tt.ptr<tensor<128x128xf8E4M3FN>>, !tt.ptr<tensor<128x128xf8E4M3FN>>
      } {DataUse, tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
      %49 = math.log %48#3 {DataUse} : tensor<128xf32>
      %50 = arith.addf %48#5, %49 {DataUse} : tensor<128xf32>
      %51 = tt.expand_dims %48#3 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %52 = tt.broadcast %51 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %53 = arith.divf %48#4, %52 {DataUse} : tensor<128x128xf32>
      %54 = arith.muli %7, %c1024_i32 : i32
      %55 = tt.addptr %arg4, %54 : !tt.ptr<f32>, i32
      %56 = tt.splat %55 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %57 = tt.addptr %56, %36 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      tt.store %57, %50 : tensor<128x!tt.ptr<f32>>
      tt.store %34, %53 : !tt.ptr<tensor<128x128xf32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/prepare_wy/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @prepare_wy_repr_bwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = arith.index_cast %3 : index to i32
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map()[%1]
// CHECK-NEXT:     %7 = arith.index_cast %6 : index to i32
// CHECK-NEXT:     %8 = affine.apply #map()[%1]
// CHECK-NEXT:     %9 = affine.apply #map()[%1]
// CHECK-NEXT:     %10 = affine.apply #map()[%1]
// CHECK-NEXT:     %11 = affine.apply #map()[%1]
// CHECK-NEXT:     %12 = affine.apply #map()[%1]
// CHECK-NEXT:     %13 = arith.index_cast %12 : index to i32
// CHECK-NEXT:     %14 = affine.apply #map()[%1]
// CHECK-NEXT:     %15 = arith.index_cast %14 : index to i32
// CHECK-NEXT:     %16 = affine.apply #map()[%1]
// CHECK-NEXT:     %17 = affine.apply #map()[%1]
// CHECK-NEXT:     %18 = arith.index_cast %17 : index to i32
// CHECK-NEXT:     %19 = tt.get_program_id x : i32
// CHECK-NEXT:     %20 = tt.get_program_id y : i32
// CHECK-NEXT:     %21 = arith.divsi %20, %c32_i32 : i32
// CHECK-NEXT:     %22 = arith.remsi %20, %c32_i32 : i32
// CHECK-NEXT:     %23 = arith.muli %21, %arg11 : i32
// CHECK-NEXT:     %24 = arith.muli %23, %c32_i32 : i32
// CHECK-NEXT:     %25 = arith.addi %24, %22 : i32
// CHECK-NEXT:     %26 = tt.addptr %arg2, %25 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %27 = arith.muli %19, %c64_i32 : i32
// CHECK-NEXT:     %28 = arith.extsi %arg11 : i32 to i64
// CHECK-NEXT:     %29 = tt.make_tensor_ptr %26, [%28], [%c32_i64], [%27] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %30 = tt.addptr %arg9, %25 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %31 = arith.addi %27, %15 : i32
// CHECK-NEXT:     %32 = tt.make_tensor_ptr %30, [%28], [%c32_i64], [%31] {order = array<i32: 0>, tiled_op} : <tensor<32xbf16>>
// CHECK-NEXT:     %33 = arith.muli %25, %c64_i32 : i32
// CHECK-NEXT:     %34 = tt.addptr %arg4, %33 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %35 = tt.make_tensor_ptr %34, [%c64_i64, %28], [%c1_i64, %c2048_i64], [%c0_i32, %27] {order = array<i32: 0, 1>} : <tensor<64x64xbf16>>
// CHECK-NEXT:     %36 = tt.load %29 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %37 = tt.load %35 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:     %38 = tt.addptr %arg3, %25 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %39 = tt.make_tensor_ptr %38, [%28], [%c32_i64], [%27] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %40 = tt.load %39 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %41 = math.exp %40 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:     %42 = arith.muli %25, %c128_i32 : i32
// CHECK-NEXT:     %43 = tt.addptr %arg0, %42 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %44 = tt.addptr %arg7, %42 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %45 = tt.addptr %arg5, %42 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %46 = arith.mulf %36, %41 {DataUse} : tensor<64xbf16>
// CHECK-NEXT:     %47 = tt.expand_dims %46 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %48 = tt.broadcast %47 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %49 = arith.extf %47 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:     %50 = tt.broadcast %49 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:     %51 = tt.expand_dims %41 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %52 = arith.extf %51 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:     %53 = tt.broadcast %52 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:     %54:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %cst_0, %arg15 = %cst_0) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
// CHECK-NEXT:       %122 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:       %123 = tt.make_tensor_ptr %43, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %122] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %124 = arith.addi %27, %4 : i32
// CHECK-NEXT:       %125 = tt.make_tensor_ptr %44, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%124, %122] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:       %126 = tt.make_tensor_ptr %45, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %122] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %127 = tt.load %123 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %128 = arith.mulf %127, %48 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:       %129 = tt.load %126 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %128[%2, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       %130 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:       %inserted_slice_6 = tensor.insert_slice %extracted_slice_5 into %130[%2, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:       %131 = tt.trans %inserted_slice_6 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %132 = tt.dot %129, %131, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %133 = tt.dot %37, %129, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %134 = arith.mulf %133, %50 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %135 = arith.extf %127 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %136 = arith.mulf %133, %135 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %137 = arith.mulf %136, %53 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %138 = "tt.reduce"(%137) <{axis = 1 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:         %145 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:         tt.reduce.return %145 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:       %139 = arith.addf %arg14, %138 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %140 = arith.extf %128 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %141 = arith.mulf %133, %140 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %142 = "tt.reduce"(%141) <{axis = 1 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg16: f32, %arg17: f32):
// CHECK-NEXT:         %145 = arith.addf %arg16, %arg17 : f32
// CHECK-NEXT:         tt.reduce.return %145 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:       %143 = arith.addf %arg15, %142 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %144 = arith.truncf %134 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %extracted_slice_7 = tensor.extract_slice %144[%3, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       tt.store %125, %extracted_slice_7 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:       scf.yield %132, %139, %143 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %55 = arith.muli %25, %c128_i32 : i32
// CHECK-NEXT:     %56 = tt.addptr %arg1, %55 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %57 = tt.addptr %arg8, %55 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %58 = tt.addptr %arg6, %55 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %59 = tt.expand_dims %36 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %60 = tt.broadcast %59 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %61 = arith.extf %59 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:     %62 = tt.broadcast %61 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:     %63:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %54#0, %arg14 = %54#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
// CHECK-NEXT:       %122 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:       %123 = tt.make_tensor_ptr %56, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %122] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %124 = arith.addi %27, %7 : i32
// CHECK-NEXT:       %125 = tt.make_tensor_ptr %57, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%124, %122] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:       %126 = tt.make_tensor_ptr %58, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %122] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %127 = tt.load %123 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %128 = arith.mulf %127, %60 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:       %129 = tt.load %126 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %128[%5, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       %130 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:       %inserted_slice_6 = tensor.insert_slice %extracted_slice_5 into %130[%5, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:       %131 = tt.trans %inserted_slice_6 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %132 = tt.dot %129, %131, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %133 = tt.dot %37, %129, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %134 = arith.mulf %133, %62 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %135 = arith.extf %127 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %136 = arith.mulf %133, %135 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %137 = "tt.reduce"(%136) <{axis = 1 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:         %140 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:         tt.reduce.return %140 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:       %138 = arith.addf %arg14, %137 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %139 = arith.truncf %134 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %extracted_slice_7 = tensor.extract_slice %139[%6, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       tt.store %125, %extracted_slice_7 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:       scf.yield %132, %138 : tensor<64x64xf32>, tensor<64xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %64 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %65 = tt.splat %27 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %66 = arith.addi %65, %64 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %67 = tt.splat %arg11 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %68 = arith.cmpi slt, %66, %67 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %69 = tt.expand_dims %66 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %70 = tt.expand_dims %66 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %71 = tt.broadcast %69 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %72 = tt.broadcast %70 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %73 = arith.cmpi sgt, %71, %72 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %74 = tt.expand_dims %68 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %75 = tt.expand_dims %68 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %76 = tt.broadcast %74 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %77 = tt.broadcast %75 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %78 = arith.andi %76, %77 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %79 = arith.andi %73, %78 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %80 = arith.select %79, %63#0, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %81 = arith.truncf %80 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %81[%8, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:     %82 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:     %inserted_slice = tensor.insert_slice %extracted_slice into %82[%8, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:     %83 = tt.dot %inserted_slice, %37, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %84 = arith.truncf %83 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     %85 = tt.dot %37, %84, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %86 = tt.expand_dims %40 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %87 = tt.expand_dims %40 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:     %88 = tt.broadcast %86 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %89 = tt.broadcast %87 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %90 = arith.subf %88, %89 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:     %91 = arith.extf %90 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %92 = math.exp %91 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %93 = arith.mulf %85, %92 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %94 = arith.subf %cst, %93 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %95 = arith.select %79, %94, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %96 = arith.truncf %95 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     gpu.barrier
// CHECK-NEXT:     %97 = arith.muli %25, %c128_i32 : i32
// CHECK-NEXT:     %98 = tt.addptr %arg0, %97 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %99 = tt.addptr %arg7, %97 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %100 = tt.expand_dims %36 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %101 = tt.broadcast %100 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %102 = arith.extf %100 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:     %103 = tt.broadcast %102 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:     %104:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst, %arg14 = %63#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
// CHECK-NEXT:       %122 = arith.muli %arg12, %c64_i32 : i32
// CHECK-NEXT:       %123 = tt.make_tensor_ptr %98, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %122] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %124 = tt.make_tensor_ptr %99, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%27, %122] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:       %125 = arith.addi %27, %13 : i32
// CHECK-NEXT:       %126 = tt.make_tensor_ptr %99, [%28, %c128_i64], [%c4096_i64, %c1_i64], [%125, %122] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:       %127 = tt.load %123 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %128 = tt.trans %127 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %129 = arith.mulf %127, %101 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:       %130 = tt.dot %127, %128, %arg13 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %extracted_slice_5 = tensor.extract_slice %96[%9, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       %131 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:       %inserted_slice_6 = tensor.insert_slice %extracted_slice_5 into %131[%9, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:       %132 = tt.dot %inserted_slice_6, %127, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %133 = arith.extf %127 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %134 = arith.mulf %132, %133 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %135 = "tt.reduce"(%134) <{axis = 1 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg15: f32, %arg16: f32):
// CHECK-NEXT:         %148 = arith.addf %arg15, %arg16 : f32
// CHECK-NEXT:         tt.reduce.return %148 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:       %136 = arith.addf %arg14, %135 {DataUse} : tensor<64xf32>
// CHECK-NEXT:       %137 = arith.mulf %132, %103 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %extracted_slice_7 = tensor.extract_slice %129[%11, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       %138 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:       %inserted_slice_8 = tensor.insert_slice %extracted_slice_7 into %138[%11, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:       %139 = tt.trans %inserted_slice_8 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:       %extracted_slice_9 = tensor.extract_slice %96[%10, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       %140 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:       %inserted_slice_10 = tensor.insert_slice %extracted_slice_9 into %140[%10, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:       %141 = tt.dot %139, %inserted_slice_10, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:       %142 = tt.trans %141 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK-NEXT:       %143 = arith.addf %137, %142 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %144 = tt.load %124 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:       %145 = arith.extf %144 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:       %146 = arith.addf %143, %145 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:       %147 = arith.truncf %146 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:       %extracted_slice_11 = tensor.extract_slice %147[%12, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:       tt.store %126, %extracted_slice_11 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:       scf.yield %130, %136 : tensor<64x64xf32>, tensor<64xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %105 = arith.truncf %104#1 {DataUse} : tensor<64xf32> to tensor<64xbf16>
// CHECK-NEXT:     %extracted_slice_1 = tensor.extract_slice %105[%14] [32] [1] {to_be_bubbled_slice} : tensor<64xbf16> to tensor<32xbf16>
// CHECK-NEXT:     tt.store %32, %extracted_slice_1 {boundaryCheck = array<i32: 0>, tiled_op} : !tt.ptr<tensor<32xbf16>>
// CHECK-NEXT:     %106 = tt.expand_dims %36 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %107 = arith.extf %106 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:     %108 = tt.broadcast %107 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:     %109 = arith.mulf %104#0, %108 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %110 = arith.extf %96 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %111 = arith.mulf %110, %109 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %112 = tt.addptr %arg10, %25 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %113 = arith.addi %27, %18 : i32
// CHECK-NEXT:     %114 = tt.make_tensor_ptr %112, [%28], [%c32_i64], [%113] {order = array<i32: 0>, tiled_op} : <tensor<32xbf16>>
// CHECK-NEXT:     %115 = "tt.reduce"(%111) <{axis = 1 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %122 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %122 : f32
// CHECK-NEXT:     }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %extracted_slice_2 = tensor.extract_slice %111[%16, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xf32> to tensor<32x64xf32>
// CHECK-NEXT:     %116 = "tt.reduce"(%extracted_slice_2) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %122 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %122 : f32
// CHECK-NEXT:     }) {DataUse, tiled_op} : (tensor<32x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %117 = tensor.empty() : tensor<2x64xf32>
// CHECK-NEXT:     %inserted_slice_3 = tensor.insert_slice %116 into %117[%1, 0] [1, 64] [1, 1] {vv_communication} : tensor<64xf32> into tensor<2x64xf32>
// CHECK-NEXT:     %118 = "tt.reduce"(%inserted_slice_3) <{axis = 0 : i32}> ({
// CHECK-NEXT:     ^bb0(%arg12: f32, %arg13: f32):
// CHECK-NEXT:       %122 = arith.addf %arg12, %arg13 : f32
// CHECK-NEXT:       tt.reduce.return %122 : f32
// CHECK-NEXT:     }) {DataUse, tiled_op} : (tensor<2x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:     %119 = arith.subf %115, %118 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %120 = arith.addf %54#2, %119 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %121 = arith.truncf %120 {DataUse} : tensor<64xf32> to tensor<64xbf16>
// CHECK-NEXT:     %extracted_slice_4 = tensor.extract_slice %121[%17] [32] [1] {to_be_bubbled_slice} : tensor<64xbf16> to tensor<32xbf16>
// CHECK-NEXT:     tt.store %114, %extracted_slice_4 {boundaryCheck = array<i32: 0>, tiled_op} : !tt.ptr<tensor<32xbf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @prepare_wy_repr_bwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg11: i32) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
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
    %16 = tt.load %10 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %17 = tt.load %15 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
    %18 = tt.addptr %arg3, %6 : !tt.ptr<bf16>, i32
    %19 = tt.make_tensor_ptr %18, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
    %20 = tt.load %19 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %21 = math.exp %20 {DataUse} : tensor<64xbf16>
    %22 = arith.muli %6, %c128_i32 : i32
    %23 = tt.addptr %arg0, %22 : !tt.ptr<bf16>, i32
    %24 = tt.addptr %arg7, %22 : !tt.ptr<bf16>, i32
    %25 = tt.addptr %arg5, %22 : !tt.ptr<bf16>, i32
    %26 = arith.mulf %16, %21 {DataUse} : tensor<64xbf16>
    %27 = tt.expand_dims %26 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %28 = tt.broadcast %27 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %29 = arith.extf %27 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
    %30 = tt.broadcast %29 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
    %31 = tt.expand_dims %21 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %32 = arith.extf %31 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
    %33 = tt.broadcast %32 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
    %34:3 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_0, %arg14 = %cst, %arg15 = %cst) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
      %98 = arith.muli %arg12, %c64_i32 : i32
      %99 = tt.make_tensor_ptr %23, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %100 = tt.make_tensor_ptr %24, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %101 = tt.make_tensor_ptr %25, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %102 = tt.load %99 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %103 = arith.mulf %102, %28 {DataUse} : tensor<64x64xbf16>
      %104 = tt.load %101 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %105 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
      %106 = tt.dot %104, %105, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %107 = tt.dot %17, %104, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %108 = arith.mulf %107, %30 {DataUse} : tensor<64x64xf32>
      %109 = arith.extf %102 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
      %110 = arith.mulf %107, %109 {DataUse} : tensor<64x64xf32>
      %111 = arith.mulf %110, %33 {DataUse} : tensor<64x64xf32>
      %112 = "tt.reduce"(%111) <{axis = 1 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %119 = arith.addf %arg16, %arg17 : f32
        tt.reduce.return %119 : f32
      }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
      %113 = arith.addf %arg14, %112 {DataUse} : tensor<64xf32>
      %114 = arith.extf %103 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
      %115 = arith.mulf %107, %114 {DataUse} : tensor<64x64xf32>
      %116 = "tt.reduce"(%115) <{axis = 1 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %119 = arith.addf %arg16, %arg17 : f32
        tt.reduce.return %119 : f32
      }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
      %117 = arith.addf %arg15, %116 {DataUse} : tensor<64xf32>
      %118 = arith.truncf %108 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
      tt.store %100, %118 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
      scf.yield %106, %113, %117 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
    } {DataUse}
    %35 = arith.muli %6, %c128_i32 : i32
    %36 = tt.addptr %arg1, %35 : !tt.ptr<bf16>, i32
    %37 = tt.addptr %arg8, %35 : !tt.ptr<bf16>, i32
    %38 = tt.addptr %arg6, %35 : !tt.ptr<bf16>, i32
    %39 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %40 = tt.broadcast %39 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %41 = arith.extf %39 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
    %42 = tt.broadcast %41 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
    %43:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %34#0, %arg14 = %34#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
      %98 = arith.muli %arg12, %c64_i32 : i32
      %99 = tt.make_tensor_ptr %36, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %100 = tt.make_tensor_ptr %37, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %101 = tt.make_tensor_ptr %38, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %102 = tt.load %99 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %103 = arith.mulf %102, %40 {DataUse} : tensor<64x64xbf16>
      %104 = tt.load %101 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %105 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
      %106 = tt.dot %104, %105, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %107 = tt.dot %17, %104, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %108 = arith.mulf %107, %42 {DataUse} : tensor<64x64xf32>
      %109 = arith.extf %102 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
      %110 = arith.mulf %107, %109 {DataUse} : tensor<64x64xf32>
      %111 = "tt.reduce"(%110) <{axis = 1 : i32}> ({
      ^bb0(%arg15: f32, %arg16: f32):
        %114 = arith.addf %arg15, %arg16 : f32
        tt.reduce.return %114 : f32
      }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
      %112 = arith.addf %arg14, %111 {DataUse} : tensor<64xf32>
      %113 = arith.truncf %108 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
      tt.store %100, %113 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
      scf.yield %106, %112 : tensor<64x64xf32>, tensor<64xf32>
    } {DataUse}
    %44 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %45 = tt.splat %8 {DataUse} : i32 -> tensor<64xi32>
    %46 = arith.addi %45, %44 {DataUse} : tensor<64xi32>
    %47 = tt.splat %arg11 {DataUse} : i32 -> tensor<64xi32>
    %48 = arith.cmpi slt, %46, %47 {DataUse} : tensor<64xi32>
    %49 = tt.expand_dims %46 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %50 = tt.expand_dims %46 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %51 = tt.broadcast %49 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
    %52 = tt.broadcast %50 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %53 = arith.cmpi sgt, %51, %52 {DataUse} : tensor<64x64xi32>
    %54 = tt.expand_dims %48 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %55 = tt.expand_dims %48 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %56 = tt.broadcast %54 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
    %57 = tt.broadcast %55 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
    %58 = arith.andi %56, %57 {DataUse} : tensor<64x64xi1>
    %59 = arith.andi %53, %58 {DataUse} : tensor<64x64xi1>
    %60 = arith.select %59, %43#0, %cst_0 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
    %61 = arith.truncf %60 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
    %62 = tt.dot %61, %17, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
    %63 = arith.truncf %62 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
    %64 = tt.dot %17, %63, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
    %65 = tt.expand_dims %20 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %66 = tt.expand_dims %20 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
    %67 = tt.broadcast %65 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %68 = tt.broadcast %66 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
    %69 = arith.subf %67, %68 {DataUse} : tensor<64x64xbf16>
    %70 = arith.extf %69 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
    %71 = math.exp %70 {DataUse} : tensor<64x64xf32>
    %72 = arith.mulf %64, %71 {DataUse} : tensor<64x64xf32>
    %73 = arith.subf %cst_0, %72 {DataUse} : tensor<64x64xf32>
    %74 = arith.select %59, %73, %cst_0 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
    %75 = arith.truncf %74 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
    gpu.barrier
    %76 = arith.muli %6, %c128_i32 : i32
    %77 = tt.addptr %arg0, %76 : !tt.ptr<bf16>, i32
    %78 = tt.addptr %arg7, %76 : !tt.ptr<bf16>, i32
    %79 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %80 = tt.broadcast %79 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %81 = arith.extf %79 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
    %82 = tt.broadcast %81 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
    %83:2 = scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg13 = %cst_0, %arg14 = %43#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
      %98 = arith.muli %arg12, %c64_i32 : i32
      %99 = tt.make_tensor_ptr %77, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %100 = tt.make_tensor_ptr %78, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %98] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
      %101 = tt.load %99 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %102 = tt.trans %101 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
      %103 = arith.mulf %101, %80 {DataUse} : tensor<64x64xbf16>
      %104 = tt.dot %101, %102, %arg13 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %105 = tt.dot %75, %101, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %106 = arith.extf %101 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
      %107 = arith.mulf %105, %106 {DataUse} : tensor<64x64xf32>
      %108 = "tt.reduce"(%107) <{axis = 1 : i32}> ({
      ^bb0(%arg15: f32, %arg16: f32):
        %119 = arith.addf %arg15, %arg16 : f32
        tt.reduce.return %119 : f32
      }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
      %109 = arith.addf %arg14, %108 {DataUse} : tensor<64xf32>
      %110 = arith.mulf %105, %82 {DataUse} : tensor<64x64xf32>
      %111 = tt.trans %103 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xbf16> -> tensor<64x64xbf16>
      %112 = tt.dot %111, %75, %cst_0 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      %113 = tt.trans %112 {DataUse, order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
      %114 = arith.addf %110, %113 {DataUse} : tensor<64x64xf32>
      %115 = tt.load %100 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
      %116 = arith.extf %115 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
      %117 = arith.addf %114, %116 {DataUse} : tensor<64x64xf32>
      %118 = arith.truncf %117 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
      tt.store %100, %118 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
      scf.yield %104, %109 : tensor<64x64xf32>, tensor<64xf32>
    } {DataUse}
    %84 = arith.truncf %83#1 {DataUse} : tensor<64xf32> to tensor<64xbf16>
    tt.store %12, %84 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xbf16>>
    %85 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %86 = arith.extf %85 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
    %87 = tt.broadcast %86 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
    %88 = arith.mulf %83#0, %87 {DataUse} : tensor<64x64xf32>
    %89 = arith.extf %75 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
    %90 = arith.mulf %89, %88 {DataUse} : tensor<64x64xf32>
    %91 = tt.addptr %arg10, %6 : !tt.ptr<bf16>, i32
    %92 = tt.make_tensor_ptr %91, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
    %93 = "tt.reduce"(%90) <{axis = 1 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %98 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %98 : f32
    }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
    %94 = "tt.reduce"(%90) <{axis = 0 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %98 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %98 : f32
    }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
    %95 = arith.subf %93, %94 {DataUse} : tensor<64xf32>
    %96 = arith.addf %34#2, %95 {DataUse} : tensor<64xf32>
    %97 = arith.truncf %96 {DataUse} : tensor<64xf32> to tensor<64xbf16>
    tt.store %92, %97 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xbf16>>
    tt.return
  }
}


// -----
// Source: gaoyou/0701/recompute_w_u/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @recompute_w_u_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = tt.get_program_id x : i32
// CHECK-NEXT:     %5 = tt.get_program_id y : i32
// CHECK-NEXT:     %6 = arith.divsi %5, %c32_i32 : i32
// CHECK-NEXT:     %7 = arith.remsi %5, %c32_i32 : i32
// CHECK-NEXT:     %8 = arith.muli %6, %arg7 : i32
// CHECK-NEXT:     %9 = arith.muli %8, %c32_i32 : i32
// CHECK-NEXT:     %10 = tt.addptr %arg2, %9 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %11 = tt.addptr %10, %7 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %12 = arith.muli %4, %c64_i32 : i32
// CHECK-NEXT:     %13 = arith.extsi %arg7 : i32 to i64
// CHECK-NEXT:     %14 = tt.make_tensor_ptr %11, [%13], [%c32_i64], [%12] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %15 = tt.load %14 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %16 = arith.addi %9, %7 : i32
// CHECK-NEXT:     %17 = arith.muli %16, %c64_i32 : i32
// CHECK-NEXT:     %18 = tt.addptr %arg5, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %19 = tt.make_tensor_ptr %18, [%13, %c64_i64], [%c2048_i64, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
// CHECK-NEXT:     %20 = tt.load %19 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
// CHECK-NEXT:     %21 = arith.muli %16, %c128_i32 : i32
// CHECK-NEXT:     %22 = tt.addptr %arg1, %21 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %23 = tt.make_tensor_ptr %22, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %24 = tt.addptr %arg4, %21 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %25 = tt.make_tensor_ptr %24, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %26 = tt.load %23 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %27 = tt.expand_dims %15 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %28 = tt.broadcast %27 {DataUse} : tensor<64x1xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:     %29 = arith.mulf %26, %28 {DataUse} : tensor<64x128xbf16>
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %29[%2, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:     %30 = tensor.empty() : tensor<64x128xbf16>
// CHECK-NEXT:     %inserted_slice = tensor.insert_slice %extracted_slice into %30[%2, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xbf16> into tensor<64x128xbf16>
// CHECK-NEXT:     %31 = tt.dot %20, %inserted_slice, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %32 = arith.truncf %31 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     tt.store %25, %32 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %33 = tt.addptr %arg6, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %34 = tt.make_tensor_ptr %33, [%13], [%c32_i64], [%12] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %35 = tt.load %34 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %36 = arith.extf %35 {DataUse} : tensor<64xbf16> to tensor<64xf32>
// CHECK-NEXT:     %37 = math.exp %36 {DataUse} : tensor<64xf32>
// CHECK-NEXT:     %38 = tt.addptr %arg0, %21 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %39 = tt.make_tensor_ptr %38, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %40 = tt.addptr %arg3, %21 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %41 = tt.make_tensor_ptr %40, [%13, %c128_i64], [%c4096_i64, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %42 = tt.load %39 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %43 = arith.mulf %42, %28 {DataUse} : tensor<64x128xbf16>
// CHECK-NEXT:     %44 = tt.expand_dims %37 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:     %45 = arith.extf %43 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
// CHECK-NEXT:     %46 = tt.broadcast %44 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:     %47 = arith.mulf %45, %46 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:     %48 = arith.truncf %47 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     %extracted_slice_0 = tensor.extract_slice %48[%3, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:     %49 = tensor.empty() : tensor<64x128xbf16>
// CHECK-NEXT:     %inserted_slice_1 = tensor.insert_slice %extracted_slice_0 into %49[%3, 0] [32, 128] [1, 1] {cv_communication_slice} : tensor<32x128xbf16> into tensor<64x128xbf16>
// CHECK-NEXT:     %50 = tt.dot %20, %inserted_slice_1, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:     %51 = arith.truncf %50 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:     tt.store %41, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @recompute_w_u_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
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
    %11 = tt.load %10 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %12 = arith.addi %5, %3 : i32
    %13 = arith.muli %12, %c64_i32 : i32
    %14 = tt.addptr %arg5, %13 : !tt.ptr<bf16>, i32
    %15 = tt.make_tensor_ptr %14, [%9, %c64_i64], [%c2048_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
    %16 = tt.load %15 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xbf16>>
    %17 = arith.muli %12, %c128_i32 : i32
    %18 = tt.addptr %arg1, %17 : !tt.ptr<bf16>, i32
    %19 = tt.make_tensor_ptr %18, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %20 = tt.addptr %arg4, %17 : !tt.ptr<bf16>, i32
    %21 = tt.make_tensor_ptr %20, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %22 = tt.load %19 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %23 = tt.expand_dims %11 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %24 = tt.broadcast %23 {DataUse} : tensor<64x1xbf16> -> tensor<64x128xbf16>
    %25 = arith.mulf %22, %24 {DataUse} : tensor<64x128xbf16>
    %26 = tt.dot %16, %25, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %27 = arith.truncf %26 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %21, %27 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    %28 = tt.addptr %arg6, %12 : !tt.ptr<bf16>, i32
    %29 = tt.make_tensor_ptr %28, [%9], [%c32_i64], [%8] {order = array<i32: 0>} : <tensor<64xbf16>>
    %30 = tt.load %29 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %31 = arith.extf %30 {DataUse} : tensor<64xbf16> to tensor<64xf32>
    %32 = math.exp %31 {DataUse} : tensor<64xf32>
    %33 = tt.addptr %arg0, %17 : !tt.ptr<bf16>, i32
    %34 = tt.make_tensor_ptr %33, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %35 = tt.addptr %arg3, %17 : !tt.ptr<bf16>, i32
    %36 = tt.make_tensor_ptr %35, [%9, %c128_i64], [%c4096_i64, %c1_i64], [%8, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %37 = tt.load %34 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %38 = arith.mulf %37, %24 {DataUse} : tensor<64x128xbf16>
    %39 = tt.expand_dims %32 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %40 = arith.extf %38 {DataUse} : tensor<64x128xbf16> to tensor<64x128xf32>
    %41 = tt.broadcast %39 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
    %42 = arith.mulf %40, %41 {DataUse} : tensor<64x128xf32>
    %43 = arith.truncf %42 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
    %44 = tt.dot %16, %43, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %45 = arith.truncf %44 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
    tt.store %36, %45 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x128xbf16>>
    tt.return
  }
}


// -----
// Source: gaoyou/0701/scaled_dot_kkt/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @chunk_scaled_dot_kkt_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = arith.index_cast %2 : index to i32
// CHECK-NEXT:     %4 = tt.get_program_id x : i32
// CHECK-NEXT:     %5 = tt.get_program_id y : i32
// CHECK-NEXT:     %6 = arith.divsi %5, %c32_i32 : i32
// CHECK-NEXT:     %7 = arith.remsi %5, %c32_i32 : i32
// CHECK-NEXT:     %8 = arith.muli %6, %arg4 : i32
// CHECK-NEXT:     %9 = arith.muli %4, %c64_i32 : i32
// CHECK-NEXT:     %10 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %11 = tt.splat %9 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %12 = arith.addi %11, %10 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %13 = tt.splat %arg4 {DataUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:     %14 = arith.cmpi slt, %12, %13 {DataUse} : tensor<64xi32>
// CHECK-NEXT:     %15 = arith.muli %8, %c32_i32 : i32
// CHECK-NEXT:     %16 = tt.addptr %arg2, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %17 = tt.addptr %16, %7 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %18 = arith.extsi %arg4 : i32 to i64
// CHECK-NEXT:     %19 = tt.make_tensor_ptr %17, [%18], [%c32_i64], [%9] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %20 = tt.load %19 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %21 = arith.addi %15, %7 : i32
// CHECK-NEXT:     %22 = arith.muli %21, %c128_i32 : i32
// CHECK-NEXT:     %23 = tt.addptr %arg0, %22 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %24 = tt.make_tensor_ptr %23, [%18, %c128_i64], [%c4096_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:     %25 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:     %26 = tt.trans %25 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:     %27 = tt.dot %25, %26, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:     %28 = tt.addptr %arg1, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %29 = tt.addptr %28, %7 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %30 = tt.make_tensor_ptr %29, [%18], [%c32_i64], [%9] {order = array<i32: 0>} : <tensor<64xbf16>>
// CHECK-NEXT:     %31 = tt.load %30 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
// CHECK-NEXT:     %32 = tt.expand_dims %31 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %33 = tt.expand_dims %31 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
// CHECK-NEXT:     %34 = tt.broadcast %32 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %35 = tt.broadcast %33 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
// CHECK-NEXT:     %36 = arith.subf %34, %35 {DataUse} : tensor<64x64xbf16>
// CHECK-NEXT:     %37 = arith.extf %36 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
// CHECK-NEXT:     %38 = math.exp %37 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %39 = arith.mulf %27, %38 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %40 = tt.expand_dims %20 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
// CHECK-NEXT:     %41 = arith.extf %40 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
// CHECK-NEXT:     %42 = tt.broadcast %41 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:     %43 = arith.mulf %39, %42 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:     %44 = tt.expand_dims %12 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %45 = tt.expand_dims %12 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %46 = tt.broadcast %44 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %47 = tt.broadcast %45 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %48 = arith.cmpi sgt, %46, %47 {DataUse} : tensor<64x64xi32>
// CHECK-NEXT:     %49 = tt.expand_dims %14 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
// CHECK-NEXT:     %50 = tt.expand_dims %14 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
// CHECK-NEXT:     %51 = tt.broadcast %49 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %52 = tt.broadcast %50 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
// CHECK-NEXT:     %53 = arith.andi %51, %52 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %54 = arith.andi %48, %53 {DataUse} : tensor<64x64xi1>
// CHECK-NEXT:     %55 = arith.select %54, %43, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:     %56 = arith.muli %21, %c64_i32 : i32
// CHECK-NEXT:     %57 = tt.addptr %arg3, %56 : !tt.ptr<bf16>, i32
// CHECK-NEXT:     %58 = arith.addi %9, %3 : i32
// CHECK-NEXT:     %59 = tt.make_tensor_ptr %57, [%18, %c64_i64], [%c2048_i64, %c1_i64], [%58, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x64xbf16>>
// CHECK-NEXT:     %60 = arith.truncf %55 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %60[%2, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:     tt.store %59, %extracted_slice {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<32x64xbf16>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @chunk_scaled_dot_kkt_fwd_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %c2048_i64 = arith.constant 2048 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
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
    %6 = tt.make_range {DataUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %7 = tt.splat %5 {DataUse} : i32 -> tensor<64xi32>
    %8 = arith.addi %7, %6 {DataUse} : tensor<64xi32>
    %9 = tt.splat %arg4 {DataUse} : i32 -> tensor<64xi32>
    %10 = arith.cmpi slt, %8, %9 {DataUse} : tensor<64xi32>
    %11 = arith.muli %4, %c32_i32 : i32
    %12 = tt.addptr %arg2, %11 : !tt.ptr<bf16>, i32
    %13 = tt.addptr %12, %3 : !tt.ptr<bf16>, i32
    %14 = arith.extsi %arg4 : i32 to i64
    %15 = tt.make_tensor_ptr %13, [%14], [%c32_i64], [%5] {order = array<i32: 0>} : <tensor<64xbf16>>
    %16 = tt.load %15 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %17 = arith.addi %11, %3 : i32
    %18 = arith.muli %17, %c128_i32 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<bf16>, i32
    %20 = tt.make_tensor_ptr %19, [%14, %c128_i64], [%c4096_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
    %21 = tt.load %20 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<64x128xbf16>>
    %22 = tt.trans %21 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
    %23 = tt.dot %21, %22, %cst {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
    %24 = tt.addptr %arg1, %11 : !tt.ptr<bf16>, i32
    %25 = tt.addptr %24, %3 : !tt.ptr<bf16>, i32
    %26 = tt.make_tensor_ptr %25, [%14], [%c32_i64], [%5] {order = array<i32: 0>} : <tensor<64xbf16>>
    %27 = tt.load %26 {DataUse, boundaryCheck = array<i32: 0>, padding = 1 : i32} : !tt.ptr<tensor<64xbf16>>
    %28 = tt.expand_dims %27 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %29 = tt.expand_dims %27 {DataUse, axis = 0 : i32} : tensor<64xbf16> -> tensor<1x64xbf16>
    %30 = tt.broadcast %28 {DataUse} : tensor<64x1xbf16> -> tensor<64x64xbf16>
    %31 = tt.broadcast %29 {DataUse} : tensor<1x64xbf16> -> tensor<64x64xbf16>
    %32 = arith.subf %30, %31 {DataUse} : tensor<64x64xbf16>
    %33 = arith.extf %32 {DataUse} : tensor<64x64xbf16> to tensor<64x64xf32>
    %34 = math.exp %33 {DataUse} : tensor<64x64xf32>
    %35 = arith.mulf %23, %34 {DataUse} : tensor<64x64xf32>
    %36 = tt.expand_dims %16 {DataUse, axis = 1 : i32} : tensor<64xbf16> -> tensor<64x1xbf16>
    %37 = arith.extf %36 {DataUse} : tensor<64x1xbf16> to tensor<64x1xf32>
    %38 = tt.broadcast %37 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
    %39 = arith.mulf %35, %38 {DataUse} : tensor<64x64xf32>
    %40 = tt.expand_dims %8 {DataUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %41 = tt.expand_dims %8 {DataUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %42 = tt.broadcast %40 {DataUse} : tensor<64x1xi32> -> tensor<64x64xi32>
    %43 = tt.broadcast %41 {DataUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %44 = arith.cmpi sgt, %42, %43 {DataUse} : tensor<64x64xi32>
    %45 = tt.expand_dims %10 {DataUse, axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %46 = tt.expand_dims %10 {DataUse, axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %47 = tt.broadcast %45 {DataUse} : tensor<64x1xi1> -> tensor<64x64xi1>
    %48 = tt.broadcast %46 {DataUse} : tensor<1x64xi1> -> tensor<64x64xi1>
    %49 = arith.andi %47, %48 {DataUse} : tensor<64x64xi1>
    %50 = arith.andi %44, %49 {DataUse} : tensor<64x64xi1>
    %51 = arith.select %50, %39, %cst {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
    %52 = arith.muli %17, %c64_i32 : i32
    %53 = tt.addptr %arg3, %52 : !tt.ptr<bf16>, i32
    %54 = tt.make_tensor_ptr %53, [%14, %c64_i64], [%c2048_i64, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xbf16>>
    %55 = arith.truncf %51 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
    tt.store %54, %55 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xbf16>>
    tt.return
  }
}


// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_kv_sep_md_false.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_6 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_10 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.0883883461> : tensor<64x128xf32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map1()[%1]
// CHECK-NEXT:     %5 = tt.get_program_id x : i32
// CHECK-NEXT:     %6 = tt.get_num_programs x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %8 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %9 = tt.expand_dims %7 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %10 = tt.broadcast %9 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %11 = tt.broadcast %9 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %12 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg9 = %5 to %c64_i32 step %6  : i32 {
// CHECK-NEXT:       %13 = arith.divsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %14 = arith.remsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %15 = arith.muli %13, %c524288_i32 : i32
// CHECK-NEXT:       %16 = tt.addptr %arg1, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %17 = tt.addptr %16, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %18 = arith.muli %14, %c64_i32 : i32
// CHECK-NEXT:       %19 = tt.splat %18 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %20 = arith.addi %19, %8 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %21 = tt.expand_dims %20 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:       %22 = arith.muli %21, %cst {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %23 = tt.splat %17 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %24 = tt.addptr %23, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %25 = tt.broadcast %24 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %26 = tt.addptr %25, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %27 = tt.addptr %arg2, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %28 = tt.addptr %27, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %29 = tt.splat %28 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %30 = tt.addptr %29, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %31 = tt.broadcast %30 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %32 = tt.addptr %31, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %33 = tt.addptr %arg5, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %34 = tt.addptr %33, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %35 = tt.splat %34 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %36 = tt.addptr %35, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %37 = tt.broadcast %36 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %38 = tt.addptr %37, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %39 = tt.addptr %arg6, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %40 = tt.addptr %39, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %41 = tt.splat %40 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %42 = tt.addptr %41, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %43 = tt.broadcast %42 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %44 = tt.addptr %43, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %45 = arith.cmpi slt, %21, %cst_0 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %46 = tt.broadcast %45 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:       %47 = tt.load %26, %46, %cst_12 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %48 = tt.load %32, %46, %cst_12 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %49 = arith.muli %13, %c2621440_i32 : i32
// CHECK-NEXT:       %50 = tt.addptr %arg0, %49 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %51 = tt.addptr %arg3, %49 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %52 = arith.muli %13, %c20480_i32 : i32
// CHECK-NEXT:       %53 = tt.addptr %arg4, %52 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %54 = tt.addptr %arg7, %52 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %55 = tt.expand_dims %20 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:       %56 = tt.broadcast %55 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:       %57 = arith.cmpi slt, %55, %cst_5 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:       %58 = tt.broadcast %57 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %59 = tt.trans %47 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %60 = tt.trans %48 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %61:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_1, %arg12 = %cst_1) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:         %64 = arith.muli %arg10, %c524288_i32 : i32
// CHECK-NEXT:         %65 = tt.addptr %50, %64 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %66 = tt.splat %65 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %67 = tt.addptr %51, %64 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %68 = tt.splat %67 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %69 = arith.muli %arg10, %c4096_i32 : i32
// CHECK-NEXT:         %70 = tt.addptr %53, %69 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %71 = tt.splat %70 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %72 = tt.addptr %54, %69 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %73 = tt.splat %72 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %74:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:           %75 = arith.muli %arg13, %c128_i32 : i32
// CHECK-NEXT:           %76 = tt.splat %75 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %77 = arith.addi %76, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %78 = tt.expand_dims %77 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %79 = arith.muli %78, %cst_2 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %80 = tt.addptr %66, %79 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %81 = tt.broadcast %80 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %82 = tt.addptr %81, %11 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %83 = tt.addptr %68, %79 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %84 = tt.broadcast %83 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %85 = tt.addptr %84, %11 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %86 = tt.addptr %71, %77 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %87 = tt.addptr %73, %77 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %88 = arith.muli %78, %cst_3 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %89 = tt.addptr %12, %88 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:           %90 = tt.broadcast %89 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:           %91 = tt.addptr %90, %56 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:           %92 = arith.cmpi slt, %78, %cst_3 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %93 = arith.cmpi slt, %77, %cst_4 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %94 = tt.broadcast %92 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:           %95 = arith.andi %94, %58 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:           %96 = tt.broadcast %92 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %97 = tt.load %82, %96, %cst_13 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %98 = tt.load %85, %96, %cst_13 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %99 = tt.load %87, %93, %cst_6 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %100 = tt.load %86, %93, %cst_6 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %101 = tt.bitcast %91 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %102 = tt.load %101, %95, %cst_14 {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %103 = tt.dot %97, %59, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %104 = arith.mulf %103, %cst_8 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %105 = arith.sitofp %102 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:           %106 = arith.subf %cst_9, %105 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %107 = arith.mulf %106, %cst_10 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %108 = arith.subf %104, %107 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %109 = tt.expand_dims %99 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %110 = tt.broadcast %109 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %111 = arith.subf %108, %110 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %112 = math.exp %111 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %113 = arith.truncf %112 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %extracted_slice_17 = tensor.extract_slice %113[%2, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:           %114 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:           %inserted_slice = tensor.insert_slice %extracted_slice_17 into %114[%2, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:           %115 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %116 = tt.dot %115, %98, %arg14 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %117 = tt.dot %98, %60, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %118 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %119 = tt.broadcast %118 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %120 = arith.subf %117, %119 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %121 = arith.mulf %112, %120 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %122 = arith.truncf %121 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %extracted_slice_18 = tensor.extract_slice %122[%3, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:           %123 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:           %inserted_slice_19 = tensor.insert_slice %extracted_slice_18 into %123[%3, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:           %124 = tt.trans %inserted_slice_19 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %125 = tt.dot %124, %97, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %126 = arith.mulf %125, %cst_11 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %127 = arith.addf %arg15, %126 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           scf.yield %116, %127 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         scf.yield %74#0, %74#1 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %62 = arith.truncf %61#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %62[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:       %extracted_slice_15 = tensor.extract_slice %38[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128x!tt.ptr<bf16>> to tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_16 = tensor.extract_slice %46[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xi1> to tensor<32x128xi1>
// CHECK-NEXT:       tt.store %extracted_slice_15, %extracted_slice, %extracted_slice_16 {tiled_op} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %63 = arith.truncf %61#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %44, %63, %46 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<64x128xf32>
    %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
    %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
    %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
    %cst_10 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
    %c20480_i32 = arith.constant 20480 : i32
    %cst_11 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c2621440_i32 = arith.constant 2621440 : i32
    %cst_12 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %cst_14 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %5 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
    %6 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %7 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg9 = %0 to %c64_i32 step %1  : i32 {
      %8 = arith.divsi %arg9, %c64_i32 : i32
      %9 = arith.remsi %arg9, %c64_i32 : i32
      %10 = arith.muli %8, %c524288_i32 : i32
      %11 = tt.addptr %arg1, %10 : !tt.ptr<bf16>, i32
      %12 = tt.addptr %11, %c0_i32 : !tt.ptr<bf16>, i32
      %13 = arith.muli %9, %c64_i32 : i32
      %14 = tt.splat %13 {MetaUse} : i32 -> tensor<64xi32>
      %15 = arith.addi %14, %3 {MetaUse} : tensor<64xi32>
      %16 = tt.expand_dims %15 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %17 = arith.muli %16, %cst_14 {MetaUse} : tensor<64x1xi32>
      %18 = tt.splat %12 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %19 = tt.addptr %18, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %20 = tt.broadcast %19 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %21 = tt.addptr %20, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %22 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
      %23 = tt.addptr %22, %c0_i32 : !tt.ptr<bf16>, i32
      %24 = tt.splat %23 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %25 = tt.addptr %24, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %26 = tt.broadcast %25 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %27 = tt.addptr %26, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %28 = tt.addptr %arg5, %10 : !tt.ptr<bf16>, i32
      %29 = tt.addptr %28, %c0_i32 : !tt.ptr<bf16>, i32
      %30 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %31 = tt.addptr %30, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %32 = tt.broadcast %31 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %33 = tt.addptr %32, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %34 = tt.addptr %arg6, %10 : !tt.ptr<bf16>, i32
      %35 = tt.addptr %34, %c0_i32 : !tt.ptr<bf16>, i32
      %36 = tt.splat %35 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %37 = tt.addptr %36, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %38 = tt.broadcast %37 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %39 = tt.addptr %38, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %40 = arith.cmpi slt, %16, %cst_13 {MetaUse} : tensor<64x1xi32>
      %41 = tt.broadcast %40 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
      %42 = tt.load %21, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
      %43 = tt.load %27, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
      %44 = arith.muli %8, %c2621440_i32 : i32
      %45 = tt.addptr %arg0, %44 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %arg3, %44 : !tt.ptr<bf16>, i32
      %47 = arith.muli %8, %c20480_i32 : i32
      %48 = tt.addptr %arg4, %47 : !tt.ptr<f32>, i32
      %49 = tt.addptr %arg7, %47 : !tt.ptr<f32>, i32
      %50 = tt.expand_dims %15 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %51 = tt.broadcast %50 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
      %52 = arith.cmpi slt, %50, %cst_8 {MetaUse} : tensor<1x64xi32>
      %53 = tt.broadcast %52 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
      %54 = tt.trans %42 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %55 = tt.trans %43 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %56:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_12, %arg12 = %cst_12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
        %59 = arith.muli %arg10, %c524288_i32 : i32
        %60 = tt.addptr %45, %59 : !tt.ptr<bf16>, i32
        %61 = tt.splat %60 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %62 = tt.addptr %46, %59 : !tt.ptr<bf16>, i32
        %63 = tt.splat %62 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %64 = arith.muli %arg10, %c4096_i32 : i32
        %65 = tt.addptr %48, %64 : !tt.ptr<f32>, i32
        %66 = tt.splat %65 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %67 = tt.addptr %49, %64 : !tt.ptr<f32>, i32
        %68 = tt.splat %67 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %69:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
          %70 = arith.muli %arg13, %c128_i32 : i32
          %71 = tt.splat %70 {MetaUse} : i32 -> tensor<128xi32>
          %72 = arith.addi %71, %2 {MetaUse} : tensor<128xi32>
          %73 = tt.expand_dims %72 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %74 = arith.muli %73, %cst_11 {MetaUse} : tensor<128x1xi32>
          %75 = tt.addptr %61, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %76 = tt.broadcast %75 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %77 = tt.addptr %76, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %78 = tt.addptr %63, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %79 = tt.broadcast %78 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %80 = tt.addptr %79, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %81 = tt.addptr %66, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %82 = tt.addptr %68, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %83 = arith.muli %73, %cst_10 {MetaUse} : tensor<128x1xi32>
          %84 = tt.addptr %7, %83 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
          %85 = tt.broadcast %84 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
          %86 = tt.addptr %85, %51 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
          %87 = arith.cmpi slt, %73, %cst_10 {MetaUse} : tensor<128x1xi32>
          %88 = arith.cmpi slt, %72, %cst_9 {MetaUse} : tensor<128xi32>
          %89 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
          %90 = arith.andi %89, %53 {MetaUse} : tensor<128x64xi1>
          %91 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
          %92 = tt.load %77, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %93 = tt.load %80, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %94 = tt.load %82, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
          %95 = tt.load %81, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
          %96 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
          %97 = tt.load %96, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
          %98 = tt.dot %92, %54, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %99 = arith.mulf %98, %cst_5 {DataUse} : tensor<128x64xf32>
          %100 = arith.sitofp %97 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
          %101 = arith.subf %cst_4, %100 {DataUse} : tensor<128x64xf32>
          %102 = arith.mulf %101, %cst_3 {DataUse} : tensor<128x64xf32>
          %103 = arith.subf %99, %102 {DataUse} : tensor<128x64xf32>
          %104 = tt.expand_dims %94 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %105 = tt.broadcast %104 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
          %106 = arith.subf %103, %105 {DataUse} : tensor<128x64xf32>
          %107 = math.exp %106 {DataUse} : tensor<128x64xf32>
          %108 = arith.truncf %107 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
          %109 = tt.trans %108 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %110 = tt.dot %109, %93, %arg14 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %111 = tt.dot %93, %55, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %112 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %113 = tt.broadcast %112 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
          %114 = arith.subf %111, %113 {DataUse} : tensor<128x64xf32>
          %115 = arith.mulf %107, %114 {DataUse} : tensor<128x64xf32>
          %116 = arith.truncf %115 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
          %117 = tt.trans %116 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %118 = tt.dot %117, %92, %cst_12 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %119 = arith.mulf %118, %cst_2 {DataUse} : tensor<64x128xf32>
          %120 = arith.addf %arg15, %119 {DataUse} : tensor<64x128xf32>
          scf.yield %110, %120 : tensor<64x128xf32>, tensor<64x128xf32>
        } {DataUse}
        scf.yield %69#0, %69#1 : tensor<64x128xf32>, tensor<64x128xf32>
      } {DataUse}
      %57 = arith.truncf %56#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %33, %57, %41 : tensor<64x128x!tt.ptr<bf16>>
      %58 = arith.truncf %56#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %39, %58, %41 : tensor<64x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_kv_sep_md_true.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: #map1 = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_6 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_10 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.0883883461> : tensor<64x128xf32>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map1()[%1]
// CHECK-NEXT:     %5 = tt.get_program_id x : i32
// CHECK-NEXT:     %6 = tt.get_num_programs x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %8 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %9 = tt.expand_dims %7 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %10 = tt.broadcast %9 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %11 = tt.broadcast %9 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %12 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg9 = %5 to %c64_i32 step %6  : i32 {
// CHECK-NEXT:       %13 = arith.divsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %14 = arith.remsi %arg9, %c64_i32 : i32
// CHECK-NEXT:       %15 = arith.muli %13, %c524288_i32 : i32
// CHECK-NEXT:       %16 = tt.addptr %arg1, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %17 = tt.addptr %16, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %18 = arith.muli %14, %c64_i32 : i32
// CHECK-NEXT:       %19 = tt.splat %18 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:       %20 = arith.addi %19, %8 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:       %21 = tt.expand_dims %20 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:       %22 = arith.muli %21, %cst {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %23 = tt.splat %17 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %24 = tt.addptr %23, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %25 = tt.broadcast %24 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %26 = tt.addptr %25, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %27 = tt.addptr %arg2, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %28 = tt.addptr %27, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %29 = tt.splat %28 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %30 = tt.addptr %29, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %31 = tt.broadcast %30 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %32 = tt.addptr %31, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %33 = tt.addptr %arg5, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %34 = tt.addptr %33, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %35 = tt.splat %34 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %36 = tt.addptr %35, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %37 = tt.broadcast %36 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %38 = tt.addptr %37, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %39 = tt.addptr %arg6, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %40 = tt.addptr %39, %c0_i32 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %41 = tt.splat %40 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %42 = tt.addptr %41, %22 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:       %43 = tt.broadcast %42 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %44 = tt.addptr %43, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:       %45 = arith.cmpi slt, %21, %cst_0 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:       %46 = tt.broadcast %45 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:       %47 = tt.load %26, %46, %cst_12 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %48 = tt.load %32, %46, %cst_12 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %49 = arith.muli %13, %c2621440_i32 : i32
// CHECK-NEXT:       %50 = tt.addptr %arg0, %49 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %51 = tt.addptr %arg3, %49 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %52 = arith.muli %13, %c20480_i32 : i32
// CHECK-NEXT:       %53 = tt.addptr %arg4, %52 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %54 = tt.addptr %arg7, %52 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %55 = tt.expand_dims %20 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:       %56 = tt.broadcast %55 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:       %57 = arith.cmpi slt, %55, %cst_5 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:       %58 = tt.broadcast %57 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %59 = tt.trans %47 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %60 = tt.trans %48 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:       %61:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_1, %arg12 = %cst_1) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:         %64 = arith.muli %arg10, %c524288_i32 : i32
// CHECK-NEXT:         %65 = tt.addptr %50, %64 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %66 = tt.splat %65 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %67 = tt.addptr %51, %64 : !tt.ptr<bf16>, i32
// CHECK-NEXT:         %68 = tt.splat %67 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:         %69 = arith.muli %arg10, %c4096_i32 : i32
// CHECK-NEXT:         %70 = tt.addptr %53, %69 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %71 = tt.splat %70 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %72 = tt.addptr %54, %69 : !tt.ptr<f32>, i32
// CHECK-NEXT:         %73 = tt.splat %72 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:         %74:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
// CHECK-NEXT:           %75 = arith.muli %arg13, %c128_i32 : i32
// CHECK-NEXT:           %76 = tt.splat %75 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:           %77 = arith.addi %76, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %78 = tt.expand_dims %77 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:           %79 = arith.muli %78, %cst_2 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %80 = tt.addptr %66, %79 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %81 = tt.broadcast %80 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %82 = tt.addptr %81, %11 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %83 = tt.addptr %68, %79 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:           %84 = tt.broadcast %83 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %85 = tt.addptr %84, %11 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:           %86 = tt.addptr %71, %77 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %87 = tt.addptr %73, %77 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:           %88 = arith.muli %78, %cst_3 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %89 = tt.addptr %12, %88 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:           %90 = tt.broadcast %89 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:           %91 = tt.addptr %90, %56 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:           %92 = arith.cmpi slt, %78, %cst_3 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:           %93 = arith.cmpi slt, %77, %cst_4 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:           %94 = tt.broadcast %92 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:           %95 = arith.andi %94, %58 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:           %96 = tt.broadcast %92 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:           %97 = tt.load %82, %96, %cst_13 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %98 = tt.load %85, %96, %cst_13 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:           %99 = tt.load %87, %93, %cst_6 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %100 = tt.load %86, %93, %cst_6 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:           %101 = tt.bitcast %91 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %102 = tt.load %101, %95, %cst_14 {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:           %103 = tt.dot %97, %59, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %104 = arith.mulf %103, %cst_8 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %105 = arith.sitofp %102 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:           %106 = arith.subf %cst_9, %105 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %107 = arith.mulf %106, %cst_10 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %108 = arith.subf %104, %107 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %109 = tt.expand_dims %99 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %110 = tt.broadcast %109 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %111 = arith.subf %108, %110 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %112 = math.exp %111 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %113 = arith.truncf %112 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %extracted_slice_17 = tensor.extract_slice %113[%2, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:           %114 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:           %inserted_slice = tensor.insert_slice %extracted_slice_17 into %114[%2, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:           %115 = tt.trans %inserted_slice {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %116 = tt.dot %115, %98, %arg14 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %117 = tt.dot %98, %60, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:           %118 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:           %119 = tt.broadcast %118 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:           %120 = arith.subf %117, %119 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %121 = arith.mulf %112, %120 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:           %122 = arith.truncf %121 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:           %extracted_slice_18 = tensor.extract_slice %122[%3, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:           %123 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:           %inserted_slice_19 = tensor.insert_slice %extracted_slice_18 into %123[%3, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:           %124 = tt.trans %inserted_slice_19 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
// CHECK-NEXT:           %125 = tt.dot %124, %97, %cst_1 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:           %126 = arith.mulf %125, %cst_11 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           %127 = arith.addf %arg15, %126 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:           scf.yield %116, %127 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:         } {DataUse}
// CHECK-NEXT:         scf.yield %74#0, %74#1 : tensor<64x128xf32>, tensor<64x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %62 = arith.truncf %61#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %62[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:       %extracted_slice_15 = tensor.extract_slice %38[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128x!tt.ptr<bf16>> to tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_16 = tensor.extract_slice %46[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xi1> to tensor<32x128xi1>
// CHECK-NEXT:       tt.store %extracted_slice_15, %extracted_slice, %extracted_slice_16 {tiled_op} : tensor<32x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %63 = arith.truncf %61#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       tt.store %44, %63, %46 : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_kv(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<64x128xf32>
    %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
    %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
    %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
    %cst_10 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
    %c20480_i32 = arith.constant 20480 : i32
    %cst_11 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c2621440_i32 = arith.constant 2621440 : i32
    %cst_12 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
    %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %cst_14 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %5 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
    %6 = tt.broadcast %4 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %7 = tt.splat %arg8 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
    scf.for %arg9 = %0 to %c64_i32 step %1  : i32 {
      %8 = arith.divsi %arg9, %c64_i32 : i32
      %9 = arith.remsi %arg9, %c64_i32 : i32
      %10 = arith.muli %8, %c524288_i32 : i32
      %11 = tt.addptr %arg1, %10 : !tt.ptr<bf16>, i32
      %12 = tt.addptr %11, %c0_i32 : !tt.ptr<bf16>, i32
      %13 = arith.muli %9, %c64_i32 : i32
      %14 = tt.splat %13 {MetaUse} : i32 -> tensor<64xi32>
      %15 = arith.addi %14, %3 {MetaUse} : tensor<64xi32>
      %16 = tt.expand_dims %15 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %17 = arith.muli %16, %cst_14 {MetaUse} : tensor<64x1xi32>
      %18 = tt.splat %12 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %19 = tt.addptr %18, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %20 = tt.broadcast %19 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %21 = tt.addptr %20, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %22 = tt.addptr %arg2, %10 : !tt.ptr<bf16>, i32
      %23 = tt.addptr %22, %c0_i32 : !tt.ptr<bf16>, i32
      %24 = tt.splat %23 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %25 = tt.addptr %24, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %26 = tt.broadcast %25 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %27 = tt.addptr %26, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %28 = tt.addptr %arg5, %10 : !tt.ptr<bf16>, i32
      %29 = tt.addptr %28, %c0_i32 : !tt.ptr<bf16>, i32
      %30 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %31 = tt.addptr %30, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %32 = tt.broadcast %31 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %33 = tt.addptr %32, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %34 = tt.addptr %arg6, %10 : !tt.ptr<bf16>, i32
      %35 = tt.addptr %34, %c0_i32 : !tt.ptr<bf16>, i32
      %36 = tt.splat %35 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %37 = tt.addptr %36, %17 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
      %38 = tt.broadcast %37 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
      %39 = tt.addptr %38, %5 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      %40 = arith.cmpi slt, %16, %cst_13 {MetaUse} : tensor<64x1xi32>
      %41 = tt.broadcast %40 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
      %42 = tt.load %21, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
      %43 = tt.load %27, %41, %cst_1 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
      %44 = arith.muli %8, %c2621440_i32 : i32
      %45 = tt.addptr %arg0, %44 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %arg3, %44 : !tt.ptr<bf16>, i32
      %47 = arith.muli %8, %c20480_i32 : i32
      %48 = tt.addptr %arg4, %47 : !tt.ptr<f32>, i32
      %49 = tt.addptr %arg7, %47 : !tt.ptr<f32>, i32
      %50 = tt.expand_dims %15 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %51 = tt.broadcast %50 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
      %52 = arith.cmpi slt, %50, %cst_8 {MetaUse} : tensor<1x64xi32>
      %53 = tt.broadcast %52 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
      %54 = tt.trans %42 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %55 = tt.trans %43 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
      %56:2 = scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32 iter_args(%arg11 = %cst_12, %arg12 = %cst_12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
        %59 = arith.muli %arg10, %c524288_i32 : i32
        %60 = tt.addptr %45, %59 : !tt.ptr<bf16>, i32
        %61 = tt.splat %60 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %62 = tt.addptr %46, %59 : !tt.ptr<bf16>, i32
        %63 = tt.splat %62 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
        %64 = arith.muli %arg10, %c4096_i32 : i32
        %65 = tt.addptr %48, %64 : !tt.ptr<f32>, i32
        %66 = tt.splat %65 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %67 = tt.addptr %49, %64 : !tt.ptr<f32>, i32
        %68 = tt.splat %67 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
        %69:2 = scf.for %arg13 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
          %70 = arith.muli %arg13, %c128_i32 : i32
          %71 = tt.splat %70 {MetaUse} : i32 -> tensor<128xi32>
          %72 = arith.addi %71, %2 {MetaUse} : tensor<128xi32>
          %73 = tt.expand_dims %72 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %74 = arith.muli %73, %cst_11 {MetaUse} : tensor<128x1xi32>
          %75 = tt.addptr %61, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %76 = tt.broadcast %75 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %77 = tt.addptr %76, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %78 = tt.addptr %63, %74 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
          %79 = tt.broadcast %78 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
          %80 = tt.addptr %79, %6 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
          %81 = tt.addptr %66, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %82 = tt.addptr %68, %72 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
          %83 = arith.muli %73, %cst_10 {MetaUse} : tensor<128x1xi32>
          %84 = tt.addptr %7, %83 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
          %85 = tt.broadcast %84 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
          %86 = tt.addptr %85, %51 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
          %87 = arith.cmpi slt, %73, %cst_10 {MetaUse} : tensor<128x1xi32>
          %88 = arith.cmpi slt, %72, %cst_9 {MetaUse} : tensor<128xi32>
          %89 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
          %90 = arith.andi %89, %53 {MetaUse} : tensor<128x64xi1>
          %91 = tt.broadcast %87 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
          %92 = tt.load %77, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %93 = tt.load %80, %91, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
          %94 = tt.load %82, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
          %95 = tt.load %81, %88, %cst_7 {DataUse} : tensor<128x!tt.ptr<f32>>
          %96 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
          %97 = tt.load %96, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
          %98 = tt.dot %92, %54, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %99 = arith.mulf %98, %cst_5 {DataUse} : tensor<128x64xf32>
          %100 = arith.sitofp %97 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
          %101 = arith.subf %cst_4, %100 {DataUse} : tensor<128x64xf32>
          %102 = arith.mulf %101, %cst_3 {DataUse} : tensor<128x64xf32>
          %103 = arith.subf %99, %102 {DataUse} : tensor<128x64xf32>
          %104 = tt.expand_dims %94 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %105 = tt.broadcast %104 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
          %106 = arith.subf %103, %105 {DataUse} : tensor<128x64xf32>
          %107 = math.exp %106 {DataUse} : tensor<128x64xf32>
          %108 = arith.truncf %107 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
          %109 = tt.trans %108 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %110 = tt.dot %109, %93, %arg14 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %111 = tt.dot %93, %55, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
          %112 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
          %113 = tt.broadcast %112 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
          %114 = arith.subf %111, %113 {DataUse} : tensor<128x64xf32>
          %115 = arith.mulf %107, %114 {DataUse} : tensor<128x64xf32>
          %116 = arith.truncf %115 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
          %117 = tt.trans %116 {DataUse, order = array<i32: 1, 0>} : tensor<128x64xbf16> -> tensor<64x128xbf16>
          %118 = tt.dot %117, %92, %cst_12 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x128xbf16> -> tensor<64x128xf32>
          %119 = arith.mulf %118, %cst_2 {DataUse} : tensor<64x128xf32>
          %120 = arith.addf %arg15, %119 {DataUse} : tensor<64x128xf32>
          scf.yield %110, %120 : tensor<64x128xf32>, tensor<64x128xf32>
        } {DataUse}
        scf.yield %69#0, %69#1 : tensor<64x128xf32>, tensor<64x128xf32>
      } {DataUse}
      %57 = arith.truncf %56#1 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %33, %57, %41 : tensor<64x128x!tt.ptr<bf16>>
      %58 = arith.truncf %56#0 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %39, %58, %41 : tensor<64x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_q_sep_md_false.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_6 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_10 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = tt.get_program_id x : i32
// CHECK-NEXT:     %5 = tt.get_num_programs x : i32
// CHECK-NEXT:     %6 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %7 = tt.expand_dims %6 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %8 = tt.broadcast %7 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %9 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %10 = tt.broadcast %7 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %11 = tt.splat %arg7 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg8 = %4 to %c160_i32 step %5  : i32 {
// CHECK-NEXT:       %12 = arith.divsi %arg8, %c160_i32 : i32
// CHECK-NEXT:       %13 = arith.divsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %14 = arith.remsi %13, %c5_i32 : i32
// CHECK-NEXT:       %15 = arith.remsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %16 = arith.muli %12, %c2621440_i32 : i32
// CHECK-NEXT:       %17 = tt.addptr %arg0, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %18 = arith.muli %14, %c524288_i32 : i32
// CHECK-NEXT:       %19 = tt.addptr %17, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %20 = arith.muli %15, %c128_i32 : i32
// CHECK-NEXT:       %21 = tt.splat %20 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %22 = arith.addi %21, %6 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %23 = tt.expand_dims %22 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %24 = arith.muli %23, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %25 = tt.splat %19 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %26 = tt.addptr %25, %24 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %27 = tt.broadcast %26 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %28 = tt.addptr %27, %8 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %29 = tt.addptr %arg3, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %30 = tt.addptr %29, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %31 = tt.splat %30 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %32 = tt.addptr %31, %24 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %33 = tt.broadcast %32 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %34 = tt.addptr %33, %8 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %35 = tt.addptr %arg5, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %36 = tt.addptr %35, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %37 = tt.splat %36 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %38 = tt.addptr %37, %24 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %39 = tt.broadcast %38 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %40 = tt.addptr %39, %8 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %41 = arith.muli %12, %c20480_i32 : i32
// CHECK-NEXT:       %42 = tt.addptr %arg4, %41 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %43 = arith.muli %14, %c4096_i32 : i32
// CHECK-NEXT:       %44 = tt.addptr %42, %43 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %45 = tt.splat %44 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %46 = tt.addptr %45, %22 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %47 = tt.addptr %arg6, %41 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %48 = tt.addptr %47, %43 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %49 = tt.splat %48 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %50 = tt.addptr %49, %22 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %51 = arith.cmpi slt, %23, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %52 = arith.cmpi slt, %22, %cst_1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %53 = tt.broadcast %51 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %54 = tt.load %28, %53, %cst_12 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %55 = tt.load %34, %53, %cst_12 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %56 = tt.load %50, %52, %cst_3 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %57 = tt.load %46, %52, %cst_3 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %58 = arith.muli %12, %c524288_i32 : i32
// CHECK-NEXT:       %59 = tt.addptr %arg1, %58 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %60 = arith.divsi %14, %c5_i32 : i32
// CHECK-NEXT:       %61 = arith.muli %60, %c524288_i32 : i32
// CHECK-NEXT:       %62 = tt.addptr %59, %61 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %63 = tt.splat %62 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %64 = tt.addptr %arg2, %58 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %65 = tt.addptr %64, %61 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %66 = tt.splat %65 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %67 = arith.muli %23, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %68 = tt.addptr %11, %67 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %69 = tt.broadcast %68 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:       %70 = tt.broadcast %51 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %71 = tt.expand_dims %56 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %72 = tt.broadcast %71 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %73 = tt.expand_dims %57 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %74 = tt.broadcast %73 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %75 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_2) -> (tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %77 = arith.muli %arg9, %c64_i32 : i32
// CHECK-NEXT:         %78 = tt.splat %77 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %79 = arith.addi %78, %9 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %80 = tt.expand_dims %79 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %81 = arith.muli %80, %cst_4 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %82 = tt.addptr %63, %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %83 = tt.broadcast %82 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %84 = tt.addptr %83, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %85 = tt.addptr %66, %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %86 = tt.broadcast %85 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %87 = tt.addptr %86, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %88 = tt.expand_dims %79 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:         %89 = tt.broadcast %88 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:         %90 = tt.addptr %69, %89 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:         %91 = arith.cmpi slt, %80, %cst_5 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %92 = arith.cmpi slt, %88, %cst_6 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:         %93 = tt.broadcast %92 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:         %94 = arith.andi %70, %93 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:         %95 = tt.broadcast %91 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:         %96 = tt.load %84, %95, %cst_13 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %97 = tt.load %87, %95, %cst_13 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %98 = tt.bitcast %90 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %99 = tt.load %98, %94, %cst_14 {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %100 = tt.trans %96 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %101 = tt.dot %54, %100, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %102 = arith.mulf %101, %cst_8 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %103 = arith.sitofp %99 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:         %104 = arith.subf %cst_9, %103 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %105 = arith.mulf %104, %cst_10 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %106 = arith.subf %102, %105 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %107 = arith.subf %106, %72 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %108 = math.exp %107 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %109 = tt.trans %97 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %110 = tt.dot %55, %109, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %111 = arith.subf %110, %74 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %112 = arith.mulf %108, %111 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %113 = arith.truncf %112 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %113[%2, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:         %114 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_17 into %114[%2, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:         %115 = tt.dot %inserted_slice, %96, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %116 = arith.mulf %115, %cst_11 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %117 = arith.addf %arg10, %116 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %117 : tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %76 = arith.truncf %75 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %76[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:       %extracted_slice_15 = tensor.extract_slice %40[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128x!tt.ptr<bf16>> to tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_16 = tensor.extract_slice %53[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xi1> to tensor<64x128xi1>
// CHECK-NEXT:       tt.store %extracted_slice_15, %extracted_slice, %extracted_slice_16 {tiled_op} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
    %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
    %cst_9 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
    %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_12 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
    %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_14 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
    %7 = tt.splat %arg7 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
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
      %17 = tt.splat %16 {MetaUse} : i32 -> tensor<128xi32>
      %18 = arith.addi %17, %2 {MetaUse} : tensor<128xi32>
      %19 = tt.expand_dims %18 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %20 = arith.muli %19, %cst_14 {MetaUse} : tensor<128x1xi32>
      %21 = tt.splat %15 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %23 = tt.broadcast %22 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %24 = tt.addptr %23, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %25 = tt.addptr %arg3, %12 : !tt.ptr<bf16>, i32
      %26 = tt.addptr %25, %14 : !tt.ptr<bf16>, i32
      %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %29 = tt.broadcast %28 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %30 = tt.addptr %29, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %31 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
      %32 = tt.addptr %31, %14 : !tt.ptr<bf16>, i32
      %33 = tt.splat %32 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %34 = tt.addptr %33, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %35 = tt.broadcast %34 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %36 = tt.addptr %35, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %37 = arith.muli %8, %c20480_i32 : i32
      %38 = tt.addptr %arg4, %37 : !tt.ptr<f32>, i32
      %39 = arith.muli %10, %c4096_i32 : i32
      %40 = tt.addptr %38, %39 : !tt.ptr<f32>, i32
      %41 = tt.splat %40 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %42 = tt.addptr %41, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %43 = tt.addptr %arg6, %37 : !tt.ptr<f32>, i32
      %44 = tt.addptr %43, %39 : !tt.ptr<f32>, i32
      %45 = tt.splat %44 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %46 = tt.addptr %45, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %47 = arith.cmpi slt, %19, %cst_13 {MetaUse} : tensor<128x1xi32>
      %48 = arith.cmpi slt, %18, %cst_12 {MetaUse} : tensor<128xi32>
      %49 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
      %50 = tt.load %24, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
      %51 = tt.load %30, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
      %52 = tt.load %46, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
      %53 = tt.load %42, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
      %54 = arith.muli %8, %c524288_i32 : i32
      %55 = tt.addptr %arg1, %54 : !tt.ptr<bf16>, i32
      %56 = arith.divsi %10, %c5_i32 : i32
      %57 = arith.muli %56, %c524288_i32 : i32
      %58 = tt.addptr %55, %57 : !tt.ptr<bf16>, i32
      %59 = tt.splat %58 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %60 = tt.addptr %arg2, %54 : !tt.ptr<bf16>, i32
      %61 = tt.addptr %60, %57 : !tt.ptr<bf16>, i32
      %62 = tt.splat %61 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %63 = arith.muli %19, %cst_13 {MetaUse} : tensor<128x1xi32>
      %64 = tt.addptr %7, %63 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %65 = tt.broadcast %64 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
      %66 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
      %67 = tt.expand_dims %52 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %68 = tt.broadcast %67 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
      %69 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %70 = tt.broadcast %69 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
      %71 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_11) -> (tensor<128x128xf32>)  : i32 {
        %73 = arith.muli %arg9, %c64_i32 : i32
        %74 = tt.splat %73 {MetaUse} : i32 -> tensor<64xi32>
        %75 = arith.addi %74, %5 {MetaUse} : tensor<64xi32>
        %76 = tt.expand_dims %75 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %77 = arith.muli %76, %cst_9 {MetaUse} : tensor<64x1xi32>
        %78 = tt.addptr %59, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %79 = tt.broadcast %78 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %80 = tt.addptr %79, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %81 = tt.addptr %62, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %82 = tt.broadcast %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %83 = tt.addptr %82, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %84 = tt.expand_dims %75 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %85 = tt.broadcast %84 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
        %86 = tt.addptr %65, %85 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
        %87 = arith.cmpi slt, %76, %cst_8 {MetaUse} : tensor<64x1xi32>
        %88 = arith.cmpi slt, %84, %cst_7 {MetaUse} : tensor<1x64xi32>
        %89 = tt.broadcast %88 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
        %90 = arith.andi %66, %89 {MetaUse} : tensor<128x64xi1>
        %91 = tt.broadcast %87 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
        %92 = tt.load %80, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
        %93 = tt.load %83, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
        %94 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
        %95 = tt.load %94, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
        %96 = tt.trans %92 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %97 = tt.dot %50, %96, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %98 = arith.mulf %97, %cst_5 {DataUse} : tensor<128x64xf32>
        %99 = arith.sitofp %95 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
        %100 = arith.subf %cst_4, %99 {DataUse} : tensor<128x64xf32>
        %101 = arith.mulf %100, %cst_3 {DataUse} : tensor<128x64xf32>
        %102 = arith.subf %98, %101 {DataUse} : tensor<128x64xf32>
        %103 = arith.subf %102, %68 {DataUse} : tensor<128x64xf32>
        %104 = math.exp %103 {DataUse} : tensor<128x64xf32>
        %105 = tt.trans %93 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %106 = tt.dot %51, %105, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %107 = arith.subf %106, %70 {DataUse} : tensor<128x64xf32>
        %108 = arith.mulf %104, %107 {DataUse} : tensor<128x64xf32>
        %109 = arith.truncf %108 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
        %110 = tt.dot %109, %92, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
        %111 = arith.mulf %110, %cst_2 {DataUse} : tensor<128x128xf32>
        %112 = arith.addf %arg10, %111 {DataUse} : tensor<128x128xf32>
        scf.yield %112 : tensor<128x128xf32>
      } {DataUse}
      %72 = arith.truncf %71 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %36, %72, %49 : tensor<128x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/sdpa_bwd/bwd_q_sep_md_true.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %cst_4 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %cst_6 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_10 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
// CHECK-NEXT:     %cst_11 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_12 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_13 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
// CHECK-NEXT:     %cst_14 = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = tt.get_program_id x : i32
// CHECK-NEXT:     %5 = tt.get_num_programs x : i32
// CHECK-NEXT:     %6 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %7 = tt.expand_dims %6 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %8 = tt.broadcast %7 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %9 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %10 = tt.broadcast %7 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
// CHECK-NEXT:     %11 = tt.splat %arg7 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg8 = %4 to %c160_i32 step %5  : i32 {
// CHECK-NEXT:       %12 = arith.divsi %arg8, %c160_i32 : i32
// CHECK-NEXT:       %13 = arith.divsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %14 = arith.remsi %13, %c5_i32 : i32
// CHECK-NEXT:       %15 = arith.remsi %arg8, %c32_i32 : i32
// CHECK-NEXT:       %16 = arith.muli %12, %c2621440_i32 : i32
// CHECK-NEXT:       %17 = tt.addptr %arg0, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %18 = arith.muli %14, %c524288_i32 : i32
// CHECK-NEXT:       %19 = tt.addptr %17, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %20 = arith.muli %15, %c128_i32 : i32
// CHECK-NEXT:       %21 = tt.splat %20 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %22 = arith.addi %21, %6 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %23 = tt.expand_dims %22 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %24 = arith.muli %23, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %25 = tt.splat %19 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %26 = tt.addptr %25, %24 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %27 = tt.broadcast %26 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %28 = tt.addptr %27, %8 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %29 = tt.addptr %arg3, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %30 = tt.addptr %29, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %31 = tt.splat %30 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %32 = tt.addptr %31, %24 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %33 = tt.broadcast %32 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %34 = tt.addptr %33, %8 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %35 = tt.addptr %arg5, %16 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %36 = tt.addptr %35, %18 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %37 = tt.splat %36 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %38 = tt.addptr %37, %24 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %39 = tt.broadcast %38 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %40 = tt.addptr %39, %8 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %41 = arith.muli %12, %c20480_i32 : i32
// CHECK-NEXT:       %42 = tt.addptr %arg4, %41 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %43 = arith.muli %14, %c4096_i32 : i32
// CHECK-NEXT:       %44 = tt.addptr %42, %43 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %45 = tt.splat %44 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %46 = tt.addptr %45, %22 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %47 = tt.addptr %arg6, %41 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %48 = tt.addptr %47, %43 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %49 = tt.splat %48 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %50 = tt.addptr %49, %22 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %51 = arith.cmpi slt, %23, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %52 = arith.cmpi slt, %22, %cst_1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %53 = tt.broadcast %51 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %54 = tt.load %28, %53, %cst_12 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %55 = tt.load %34, %53, %cst_12 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %56 = tt.load %50, %52, %cst_3 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %57 = tt.load %46, %52, %cst_3 {DataUse} : tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %58 = arith.muli %12, %c524288_i32 : i32
// CHECK-NEXT:       %59 = tt.addptr %arg1, %58 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %60 = arith.divsi %14, %c5_i32 : i32
// CHECK-NEXT:       %61 = arith.muli %60, %c524288_i32 : i32
// CHECK-NEXT:       %62 = tt.addptr %59, %61 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %63 = tt.splat %62 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %64 = tt.addptr %arg2, %58 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %65 = tt.addptr %64, %61 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %66 = tt.splat %65 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %67 = arith.muli %23, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %68 = tt.addptr %11, %67 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %69 = tt.broadcast %68 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
// CHECK-NEXT:       %70 = tt.broadcast %51 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
// CHECK-NEXT:       %71 = tt.expand_dims %56 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %72 = tt.broadcast %71 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %73 = tt.expand_dims %57 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %74 = tt.broadcast %73 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
// CHECK-NEXT:       %75 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_2) -> (tensor<128x128xf32>)  : i32 {
// CHECK-NEXT:         %77 = arith.muli %arg9, %c64_i32 : i32
// CHECK-NEXT:         %78 = tt.splat %77 {MetaUse} : i32 -> tensor<64xi32>
// CHECK-NEXT:         %79 = arith.addi %78, %9 {MetaUse} : tensor<64xi32>
// CHECK-NEXT:         %80 = tt.expand_dims %79 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:         %81 = arith.muli %80, %cst_4 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %82 = tt.addptr %63, %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %83 = tt.broadcast %82 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %84 = tt.addptr %83, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %85 = tt.addptr %66, %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
// CHECK-NEXT:         %86 = tt.broadcast %85 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %87 = tt.addptr %86, %10 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
// CHECK-NEXT:         %88 = tt.expand_dims %79 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:         %89 = tt.broadcast %88 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
// CHECK-NEXT:         %90 = tt.addptr %69, %89 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
// CHECK-NEXT:         %91 = arith.cmpi slt, %80, %cst_5 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:         %92 = arith.cmpi slt, %88, %cst_6 {MetaUse} : tensor<1x64xi32>
// CHECK-NEXT:         %93 = tt.broadcast %92 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
// CHECK-NEXT:         %94 = arith.andi %70, %93 {MetaUse} : tensor<128x64xi1>
// CHECK-NEXT:         %95 = tt.broadcast %91 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
// CHECK-NEXT:         %96 = tt.load %84, %95, %cst_13 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %97 = tt.load %87, %95, %cst_13 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %98 = tt.bitcast %90 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %99 = tt.load %98, %94, %cst_14 {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
// CHECK-NEXT:         %100 = tt.trans %96 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %101 = tt.dot %54, %100, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %102 = arith.mulf %101, %cst_8 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %103 = arith.sitofp %99 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
// CHECK-NEXT:         %104 = arith.subf %cst_9, %103 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %105 = arith.mulf %104, %cst_10 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %106 = arith.subf %102, %105 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %107 = arith.subf %106, %72 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %108 = math.exp %107 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %109 = tt.trans %97 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %110 = tt.dot %55, %109, %cst_7 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
// CHECK-NEXT:         %111 = arith.subf %110, %74 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %112 = arith.mulf %108, %111 {DataUse} : tensor<128x64xf32>
// CHECK-NEXT:         %113 = arith.truncf %112 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
// CHECK-NEXT:         %extracted_slice_17 = tensor.extract_slice %113[%2, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : tensor<128x64xbf16> to tensor<64x64xbf16>
// CHECK-NEXT:         %114 = tensor.empty() : tensor<128x64xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_17 into %114[%2, 0] [64, 64] [1, 1] {cv_communication_slice} : tensor<64x64xbf16> into tensor<128x64xbf16>
// CHECK-NEXT:         %115 = tt.dot %inserted_slice, %96, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %116 = arith.mulf %115, %cst_11 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %117 = arith.addf %arg10, %116 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %117 : tensor<128x128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %76 = arith.truncf %75 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %76[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:       %extracted_slice_15 = tensor.extract_slice %40[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128x!tt.ptr<bf16>> to tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_16 = tensor.extract_slice %53[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xi1> to tensor<64x128xi1>
// CHECK-NEXT:       tt.store %extracted_slice_15, %extracted_slice, %extracted_slice_16 {tiled_op} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_bwd_q(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant {MetaUse} dense<0> : tensor<128x64xi8>
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<64x128xbf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_2 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %cst_3 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x64xf32>
    %cst_4 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x64xf32>
    %cst_5 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x64xf32>
    %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x64xf32>
    %cst_7 = arith.constant {MetaUse} dense<4096> : tensor<1x64xi32>
    %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
    %cst_9 = arith.constant {MetaUse} dense<128> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_10 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128xf32>
    %cst_11 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_12 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
    %cst_13 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_14 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<64x128xi32>
    %7 = tt.splat %arg7 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
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
      %17 = tt.splat %16 {MetaUse} : i32 -> tensor<128xi32>
      %18 = arith.addi %17, %2 {MetaUse} : tensor<128xi32>
      %19 = tt.expand_dims %18 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %20 = arith.muli %19, %cst_14 {MetaUse} : tensor<128x1xi32>
      %21 = tt.splat %15 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %23 = tt.broadcast %22 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %24 = tt.addptr %23, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %25 = tt.addptr %arg3, %12 : !tt.ptr<bf16>, i32
      %26 = tt.addptr %25, %14 : !tt.ptr<bf16>, i32
      %27 = tt.splat %26 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %29 = tt.broadcast %28 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %30 = tt.addptr %29, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %31 = tt.addptr %arg5, %12 : !tt.ptr<bf16>, i32
      %32 = tt.addptr %31, %14 : !tt.ptr<bf16>, i32
      %33 = tt.splat %32 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %34 = tt.addptr %33, %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %35 = tt.broadcast %34 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %36 = tt.addptr %35, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %37 = arith.muli %8, %c20480_i32 : i32
      %38 = tt.addptr %arg4, %37 : !tt.ptr<f32>, i32
      %39 = arith.muli %10, %c4096_i32 : i32
      %40 = tt.addptr %38, %39 : !tt.ptr<f32>, i32
      %41 = tt.splat %40 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %42 = tt.addptr %41, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %43 = tt.addptr %arg6, %37 : !tt.ptr<f32>, i32
      %44 = tt.addptr %43, %39 : !tt.ptr<f32>, i32
      %45 = tt.splat %44 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %46 = tt.addptr %45, %18 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %47 = arith.cmpi slt, %19, %cst_13 {MetaUse} : tensor<128x1xi32>
      %48 = arith.cmpi slt, %18, %cst_12 {MetaUse} : tensor<128xi32>
      %49 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
      %50 = tt.load %24, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
      %51 = tt.load %30, %49, %cst_1 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
      %52 = tt.load %46, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
      %53 = tt.load %42, %48, %cst_10 {DataUse} : tensor<128x!tt.ptr<f32>>
      %54 = arith.muli %8, %c524288_i32 : i32
      %55 = tt.addptr %arg1, %54 : !tt.ptr<bf16>, i32
      %56 = arith.divsi %10, %c5_i32 : i32
      %57 = arith.muli %56, %c524288_i32 : i32
      %58 = tt.addptr %55, %57 : !tt.ptr<bf16>, i32
      %59 = tt.splat %58 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %60 = tt.addptr %arg2, %54 : !tt.ptr<bf16>, i32
      %61 = tt.addptr %60, %57 : !tt.ptr<bf16>, i32
      %62 = tt.splat %61 {MetaUse} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
      %63 = arith.muli %19, %cst_13 {MetaUse} : tensor<128x1xi32>
      %64 = tt.addptr %7, %63 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %65 = tt.broadcast %64 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i1>>
      %66 = tt.broadcast %47 {MetaUse} : tensor<128x1xi1> -> tensor<128x64xi1>
      %67 = tt.expand_dims %52 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %68 = tt.broadcast %67 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
      %69 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %70 = tt.broadcast %69 {DataUse} : tensor<128x1xf32> -> tensor<128x64xf32>
      %71 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg10 = %cst_11) -> (tensor<128x128xf32>)  : i32 {
        %73 = arith.muli %arg9, %c64_i32 : i32
        %74 = tt.splat %73 {MetaUse} : i32 -> tensor<64xi32>
        %75 = arith.addi %74, %5 {MetaUse} : tensor<64xi32>
        %76 = tt.expand_dims %75 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %77 = arith.muli %76, %cst_9 {MetaUse} : tensor<64x1xi32>
        %78 = tt.addptr %59, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %79 = tt.broadcast %78 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %80 = tt.addptr %79, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %81 = tt.addptr %62, %77 {MetaUse} : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
        %82 = tt.broadcast %81 {MetaUse} : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
        %83 = tt.addptr %82, %6 {MetaUse} : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
        %84 = tt.expand_dims %75 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %85 = tt.broadcast %84 {MetaUse} : tensor<1x64xi32> -> tensor<128x64xi32>
        %86 = tt.addptr %65, %85 {MetaUse} : tensor<128x64x!tt.ptr<i1>>, tensor<128x64xi32>
        %87 = arith.cmpi slt, %76, %cst_8 {MetaUse} : tensor<64x1xi32>
        %88 = arith.cmpi slt, %84, %cst_7 {MetaUse} : tensor<1x64xi32>
        %89 = tt.broadcast %88 {MetaUse} : tensor<1x64xi1> -> tensor<128x64xi1>
        %90 = arith.andi %66, %89 {MetaUse} : tensor<128x64xi1>
        %91 = tt.broadcast %87 {MetaUse} : tensor<64x1xi1> -> tensor<64x128xi1>
        %92 = tt.load %80, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
        %93 = tt.load %83, %91, %cst_0 {DataUse} : tensor<64x128x!tt.ptr<bf16>>
        %94 = tt.bitcast %86 {MetaUse} : tensor<128x64x!tt.ptr<i1>> -> tensor<128x64x!tt.ptr<i8>>
        %95 = tt.load %94, %90, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x64x!tt.ptr<i8>>
        %96 = tt.trans %92 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %97 = tt.dot %50, %96, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %98 = arith.mulf %97, %cst_5 {DataUse} : tensor<128x64xf32>
        %99 = arith.sitofp %95 {DataUse} : tensor<128x64xi8> to tensor<128x64xf32>
        %100 = arith.subf %cst_4, %99 {DataUse} : tensor<128x64xf32>
        %101 = arith.mulf %100, %cst_3 {DataUse} : tensor<128x64xf32>
        %102 = arith.subf %98, %101 {DataUse} : tensor<128x64xf32>
        %103 = arith.subf %102, %68 {DataUse} : tensor<128x64xf32>
        %104 = math.exp %103 {DataUse} : tensor<128x64xf32>
        %105 = tt.trans %93 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %106 = tt.dot %51, %105, %cst_6 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x64xbf16> -> tensor<128x64xf32>
        %107 = arith.subf %106, %70 {DataUse} : tensor<128x64xf32>
        %108 = arith.mulf %104, %107 {DataUse} : tensor<128x64xf32>
        %109 = arith.truncf %108 {DataUse} : tensor<128x64xf32> to tensor<128x64xbf16>
        %110 = tt.dot %109, %92, %cst_11 {DataUse, triton_cv12.normalized_dot} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
        %111 = arith.mulf %110, %cst_2 {DataUse} : tensor<128x128xf32>
        %112 = arith.addf %arg10, %111 {DataUse} : tensor<128x128xf32>
        scf.yield %112 : tensor<128x128xf32>
      } {DataUse}
      %72 = arith.truncf %71 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %36, %72, %49 : tensor<128x128x!tt.ptr<bf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/sdpa_fwd/sep_md_false.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<128xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<4096> : tensor<1x128xi32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = tt.get_program_id x : i32
// CHECK-NEXT:     %6 = tt.get_num_programs x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %8 = tt.expand_dims %7 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %9 = tt.broadcast %8 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %10 = tt.splat %arg5 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg6 = %5 to %c160_i32 step %6  : i32 {
// CHECK-NEXT:       %11 = arith.divsi %arg6, %c160_i32 : i32
// CHECK-NEXT:       %12 = arith.divsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %13 = arith.remsi %12, %c5_i32 : i32
// CHECK-NEXT:       %14 = arith.remsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %15 = arith.muli %11, %c2621440_i32 : i32
// CHECK-NEXT:       %16 = tt.addptr %arg0, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %17 = arith.muli %13, %c524288_i32 : i32
// CHECK-NEXT:       %18 = tt.addptr %16, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %19 = arith.muli %14, %c128_i32 : i32
// CHECK-NEXT:       %20 = tt.splat %19 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %21 = arith.addi %20, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %22 = tt.expand_dims %21 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %23 = arith.muli %22, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %24 = tt.splat %18 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %25 = tt.addptr %24, %23 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %26 = tt.broadcast %25 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %27 = tt.addptr %26, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %28 = tt.addptr %arg3, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %29 = tt.addptr %28, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %30 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %31 = tt.addptr %30, %23 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %32 = tt.broadcast %31 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %33 = tt.addptr %32, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %34 = arith.muli %11, %c20480_i32 : i32
// CHECK-NEXT:       %35 = tt.addptr %arg4, %34 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %36 = arith.muli %13, %c4096_i32 : i32
// CHECK-NEXT:       %37 = tt.addptr %35, %36 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %38 = tt.splat %37 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %39 = tt.addptr %38, %21 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %40 = arith.cmpi slt, %22, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %41 = arith.cmpi slt, %21, %cst_1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %42 = tt.broadcast %40 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %43 = tt.load %27, %42, %cst_9 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %44 = arith.muli %11, %c524288_i32 : i32
// CHECK-NEXT:       %45 = tt.addptr %arg1, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %46 = arith.divsi %13, %c5_i32 : i32
// CHECK-NEXT:       %47 = arith.muli %46, %c524288_i32 : i32
// CHECK-NEXT:       %48 = tt.addptr %45, %47 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %49 = tt.splat %48 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %50 = tt.addptr %arg2, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %51 = tt.addptr %50, %47 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %52 = tt.splat %51 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %53 = arith.muli %22, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %54 = tt.addptr %10, %53 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %55 = tt.broadcast %54 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
// CHECK-NEXT:       %56:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_2, %arg9 = %cst_4, %arg10 = %cst_3) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
// CHECK-NEXT:         %63 = arith.muli %arg7, %c128_i32 : i32
// CHECK-NEXT:         %64 = tt.splat %63 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %65 = arith.addi %64, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %66 = tt.expand_dims %65 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %67 = arith.muli %66, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %68 = tt.addptr %49, %67 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %69 = tt.broadcast %68 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %70 = tt.addptr %69, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %71 = tt.addptr %52, %67 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %72 = tt.broadcast %71 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %73 = tt.addptr %72, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %74 = tt.expand_dims %65 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:         %75 = tt.broadcast %74 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:         %76 = tt.addptr %55, %75 {MetaUse} : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
// CHECK-NEXT:         %77 = arith.cmpi slt, %66, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %78 = arith.cmpi slt, %74, %cst_5 {MetaUse} : tensor<1x128xi32>
// CHECK-NEXT:         %79 = tt.broadcast %78 {MetaUse} : tensor<1x128xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %80 = arith.andi %42, %79 {MetaUse} : tensor<128x128xi1>
// CHECK-NEXT:         %81 = tt.bitcast %76 {MetaUse} : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %82 = tt.load %81, %80, %cst_10 {DataUse, was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %83 = tt.broadcast %77 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %84 = tt.load %70, %83, %cst_9 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %85 = tt.load %73, %83, %cst_9 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %86 = tt.trans %84 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %87 = tt.dot %43, %86, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %88 = arith.mulf %87, %cst_6 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %89 = arith.sitofp %82 {DataUse} : tensor<128x128xi8> to tensor<128x128xf32>
// CHECK-NEXT:         %90 = arith.subf %cst_7, %89 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %91 = arith.mulf %90, %cst_8 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.subf %88, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = "tt.reduce"(%92) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %111 = arith.maximumf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %111 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %94 = arith.maximumf %arg9, %93 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %95 = tt.expand_dims %94 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %96 = tt.broadcast %95 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %97 = arith.subf %92, %96 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %98 = math.exp %97 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %99 = arith.subf %arg9, %94 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %100 = math.exp %99 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %101 = arith.mulf %100, %arg10 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %102 = "tt.reduce"(%98) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %111 = arith.addf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %111 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %103 = arith.addf %101, %102 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %104 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %105 = tt.broadcast %104 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %106 = arith.mulf %105, %arg8 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %107 = arith.truncf %98 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %107[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:         %108 = tensor.empty() : tensor<128x128xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_16 into %108[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xbf16> into tensor<128x128xbf16>
// CHECK-NEXT:         %109 = tt.dot %inserted_slice, %85, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %110 = arith.addf %109, %106 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %110, %94, %103 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %57 = tt.expand_dims %56#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %58 = tt.broadcast %57 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %59 = arith.divf %56#0, %58 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %60 = math.log %56#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %61 = arith.addf %60, %56#1 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %62 = arith.truncf %59 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %62[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:       %extracted_slice_11 = tensor.extract_slice %33[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128x!tt.ptr<bf16>> to tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %42[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xi1> to tensor<64x128xi1>
// CHECK-NEXT:       tt.store %extracted_slice_11, %extracted_slice, %extracted_slice_12 {tiled_op} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_13 = tensor.extract_slice %61[%4] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_14 = tensor.extract_slice %39[%4] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %extracted_slice_15 = tensor.extract_slice %41[%4] [64] [1] {to_be_bubbled_slice} : tensor<128xi1> to tensor<64xi1>
// CHECK-NEXT:       tt.store %extracted_slice_14, %extracted_slice_13, %extracted_slice_15 {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant {MetaUse} dense<0> : tensor<128x128xi8>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_1 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x128xf32>
    %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x128xf32>
    %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant {MetaUse} dense<4096> : tensor<1x128xi32>
    %cst_5 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<128xf32>
    %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128xf32>
    %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
    %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_10 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.splat %arg5 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
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
      %15 = tt.splat %14 {MetaUse} : i32 -> tensor<128xi32>
      %16 = arith.addi %15, %2 {MetaUse} : tensor<128xi32>
      %17 = tt.expand_dims %16 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %18 = arith.muli %17, %cst_10 {MetaUse} : tensor<128x1xi32>
      %19 = tt.splat %13 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %20 = tt.addptr %19, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %21 = tt.broadcast %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
      %24 = tt.addptr %23, %12 : !tt.ptr<bf16>, i32
      %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %26 = tt.addptr %25, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %27 = tt.broadcast %26 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %29 = arith.muli %6, %c20480_i32 : i32
      %30 = tt.addptr %arg4, %29 : !tt.ptr<f32>, i32
      %31 = arith.muli %8, %c4096_i32 : i32
      %32 = tt.addptr %30, %31 : !tt.ptr<f32>, i32
      %33 = tt.splat %32 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %34 = tt.addptr %33, %16 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %35 = arith.cmpi slt, %17, %cst_9 {MetaUse} : tensor<128x1xi32>
      %36 = arith.cmpi slt, %16, %cst_8 {MetaUse} : tensor<128xi32>
      %37 = tt.broadcast %35 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
      %38 = tt.load %22, %37, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
      %39 = arith.muli %6, %c524288_i32 : i32
      %40 = tt.addptr %arg1, %39 : !tt.ptr<bf16>, i32
      %41 = arith.divsi %8, %c5_i32 : i32
      %42 = arith.muli %41, %c524288_i32 : i32
      %43 = tt.addptr %40, %42 : !tt.ptr<bf16>, i32
      %44 = tt.splat %43 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %45 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %45, %42 : !tt.ptr<bf16>, i32
      %47 = tt.splat %46 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %48 = arith.muli %17, %cst_9 {MetaUse} : tensor<128x1xi32>
      %49 = tt.addptr %5, %48 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %50 = tt.broadcast %49 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
      %51:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_7, %arg9 = %cst_5, %arg10 = %cst_6) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
        %58 = arith.muli %arg7, %c128_i32 : i32
        %59 = tt.splat %58 {MetaUse} : i32 -> tensor<128xi32>
        %60 = arith.addi %59, %2 {MetaUse} : tensor<128xi32>
        %61 = tt.expand_dims %60 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %62 = arith.muli %61, %cst_10 {MetaUse} : tensor<128x1xi32>
        %63 = tt.addptr %44, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %66 = tt.addptr %47, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %67 = tt.broadcast %66 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %68 = tt.addptr %67, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %69 = tt.expand_dims %60 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %70 = tt.broadcast %69 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
        %71 = tt.addptr %50, %70 {MetaUse} : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
        %72 = arith.cmpi slt, %61, %cst_9 {MetaUse} : tensor<128x1xi32>
        %73 = arith.cmpi slt, %69, %cst_4 {MetaUse} : tensor<1x128xi32>
        %74 = tt.broadcast %73 {MetaUse} : tensor<1x128xi1> -> tensor<128x128xi1>
        %75 = arith.andi %37, %74 {MetaUse} : tensor<128x128xi1>
        %76 = tt.bitcast %71 {MetaUse} : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
        %77 = tt.load %76, %75, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
        %78 = tt.broadcast %72 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
        %79 = tt.load %65, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
        %80 = tt.load %68, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
        %81 = tt.trans %79 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %82 = tt.dot %38, %81, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %83 = arith.mulf %82, %cst_3 {DataUse} : tensor<128x128xf32>
        %84 = arith.sitofp %77 {DataUse} : tensor<128x128xi8> to tensor<128x128xf32>
        %85 = arith.subf %cst_2, %84 {DataUse} : tensor<128x128xf32>
        %86 = arith.mulf %85, %cst_1 {DataUse} : tensor<128x128xf32>
        %87 = arith.subf %83, %86 {DataUse} : tensor<128x128xf32>
        %88 = "tt.reduce"(%87) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.maximumf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %89 = arith.maximumf %arg9, %88 {DataUse} : tensor<128xf32>
        %90 = tt.expand_dims %89 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.subf %87, %91 {DataUse} : tensor<128x128xf32>
        %93 = math.exp %92 {DataUse} : tensor<128x128xf32>
        %94 = arith.subf %arg9, %89 {DataUse} : tensor<128xf32>
        %95 = math.exp %94 {DataUse} : tensor<128xf32>
        %96 = arith.mulf %95, %arg10 {DataUse} : tensor<128xf32>
        %97 = "tt.reduce"(%93) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.addf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %98 = arith.addf %96, %97 {DataUse} : tensor<128xf32>
        %99 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %100 = tt.broadcast %99 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %101 = arith.mulf %100, %arg8 {DataUse} : tensor<128x128xf32>
        %102 = arith.truncf %93 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
        %103 = tt.dot %102, %80, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %104 = arith.addf %103, %101 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
        scf.yield %104, %89, %98 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
      } {DataUse}
      %52 = tt.expand_dims %51#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %53 = tt.broadcast %52 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %54 = arith.divf %51#0, %53 {DataUse} : tensor<128x128xf32>
      %55 = math.log %51#2 {DataUse} : tensor<128xf32>
      %56 = arith.addf %55, %51#1 {DataUse} : tensor<128xf32>
      %57 = arith.truncf %54 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %28, %57, %37 : tensor<128x128x!tt.ptr<bf16>>
      tt.store %34, %56, %36 : tensor<128x!tt.ptr<f32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/sdpa_fwd/sep_md_true.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c2621440_i32 = arith.constant 2621440 : i32
// CHECK-NEXT:     %c524288_i32 = arith.constant 524288 : i32
// CHECK-NEXT:     %c128_i32 = arith.constant 128 : i32
// CHECK-NEXT:     %cst = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
// CHECK-NEXT:     %c20480_i32 = arith.constant 20480 : i32
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128xf32>
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<128xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {MetaUse} dense<4096> : tensor<1x128xi32>
// CHECK-NEXT:     %cst_6 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_7 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_8 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x128xf32>
// CHECK-NEXT:     %cst_9 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %cst_10 = arith.constant {MetaUse} dense<0> : tensor<128x128xi8>
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c160_i32 = arith.constant 160 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = tt.get_program_id x : i32
// CHECK-NEXT:     %6 = tt.get_num_programs x : i32
// CHECK-NEXT:     %7 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK-NEXT:     %8 = tt.expand_dims %7 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:     %9 = tt.broadcast %8 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:     %10 = tt.splat %arg5 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
// CHECK-NEXT:     scf.for %arg6 = %5 to %c160_i32 step %6  : i32 {
// CHECK-NEXT:       %11 = arith.divsi %arg6, %c160_i32 : i32
// CHECK-NEXT:       %12 = arith.divsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %13 = arith.remsi %12, %c5_i32 : i32
// CHECK-NEXT:       %14 = arith.remsi %arg6, %c32_i32 : i32
// CHECK-NEXT:       %15 = arith.muli %11, %c2621440_i32 : i32
// CHECK-NEXT:       %16 = tt.addptr %arg0, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %17 = arith.muli %13, %c524288_i32 : i32
// CHECK-NEXT:       %18 = tt.addptr %16, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %19 = arith.muli %14, %c128_i32 : i32
// CHECK-NEXT:       %20 = tt.splat %19 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:       %21 = arith.addi %20, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %22 = tt.expand_dims %21 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:       %23 = arith.muli %22, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %24 = tt.splat %18 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %25 = tt.addptr %24, %23 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %26 = tt.broadcast %25 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %27 = tt.addptr %26, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %28 = tt.addptr %arg3, %15 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %29 = tt.addptr %28, %17 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %30 = tt.splat %29 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %31 = tt.addptr %30, %23 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:       %32 = tt.broadcast %31 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %33 = tt.addptr %32, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:       %34 = arith.muli %11, %c20480_i32 : i32
// CHECK-NEXT:       %35 = tt.addptr %arg4, %34 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %36 = arith.muli %13, %c4096_i32 : i32
// CHECK-NEXT:       %37 = tt.addptr %35, %36 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %38 = tt.splat %37 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
// CHECK-NEXT:       %39 = tt.addptr %38, %21 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK-NEXT:       %40 = arith.cmpi slt, %22, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %41 = arith.cmpi slt, %21, %cst_1 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:       %42 = tt.broadcast %40 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:       %43 = tt.load %27, %42, %cst_9 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %44 = arith.muli %11, %c524288_i32 : i32
// CHECK-NEXT:       %45 = tt.addptr %arg1, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %46 = arith.divsi %13, %c5_i32 : i32
// CHECK-NEXT:       %47 = arith.muli %46, %c524288_i32 : i32
// CHECK-NEXT:       %48 = tt.addptr %45, %47 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %49 = tt.splat %48 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %50 = tt.addptr %arg2, %44 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %51 = tt.addptr %50, %47 : !tt.ptr<bf16>, i32
// CHECK-NEXT:       %52 = tt.splat %51 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
// CHECK-NEXT:       %53 = arith.muli %22, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:       %54 = tt.addptr %10, %53 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
// CHECK-NEXT:       %55 = tt.broadcast %54 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
// CHECK-NEXT:       %56:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_2, %arg9 = %cst_4, %arg10 = %cst_3) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
// CHECK-NEXT:         %63 = arith.muli %arg7, %c128_i32 : i32
// CHECK-NEXT:         %64 = tt.splat %63 {MetaUse} : i32 -> tensor<128xi32>
// CHECK-NEXT:         %65 = arith.addi %64, %7 {MetaUse} : tensor<128xi32>
// CHECK-NEXT:         %66 = tt.expand_dims %65 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
// CHECK-NEXT:         %67 = arith.muli %66, %cst {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %68 = tt.addptr %49, %67 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %69 = tt.broadcast %68 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %70 = tt.addptr %69, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %71 = tt.addptr %52, %67 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
// CHECK-NEXT:         %72 = tt.broadcast %71 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %73 = tt.addptr %72, %9 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
// CHECK-NEXT:         %74 = tt.expand_dims %65 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
// CHECK-NEXT:         %75 = tt.broadcast %74 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
// CHECK-NEXT:         %76 = tt.addptr %55, %75 {MetaUse} : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
// CHECK-NEXT:         %77 = arith.cmpi slt, %66, %cst_0 {MetaUse} : tensor<128x1xi32>
// CHECK-NEXT:         %78 = arith.cmpi slt, %74, %cst_5 {MetaUse} : tensor<1x128xi32>
// CHECK-NEXT:         %79 = tt.broadcast %78 {MetaUse} : tensor<1x128xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %80 = arith.andi %42, %79 {MetaUse} : tensor<128x128xi1>
// CHECK-NEXT:         %81 = tt.bitcast %76 {MetaUse} : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %82 = tt.load %81, %80, %cst_10 {DataUse, was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
// CHECK-NEXT:         %83 = tt.broadcast %77 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
// CHECK-NEXT:         %84 = tt.load %70, %83, %cst_9 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %85 = tt.load %73, %83, %cst_9 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
// CHECK-NEXT:         %86 = tt.trans %84 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
// CHECK-NEXT:         %87 = tt.dot %43, %86, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %88 = arith.mulf %87, %cst_6 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %89 = arith.sitofp %82 {DataUse} : tensor<128x128xi8> to tensor<128x128xf32>
// CHECK-NEXT:         %90 = arith.subf %cst_7, %89 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %91 = arith.mulf %90, %cst_8 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %92 = arith.subf %88, %91 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %93 = "tt.reduce"(%92) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %111 = arith.maximumf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %111 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %94 = arith.maximumf %arg9, %93 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %95 = tt.expand_dims %94 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %96 = tt.broadcast %95 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %97 = arith.subf %92, %96 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %98 = math.exp %97 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %99 = arith.subf %arg9, %94 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %100 = math.exp %99 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %101 = arith.mulf %100, %arg10 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %102 = "tt.reduce"(%98) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg11: f32, %arg12: f32):
// CHECK-NEXT:           %111 = arith.addf %arg11, %arg12 : f32
// CHECK-NEXT:           tt.reduce.return %111 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
// CHECK-NEXT:         %103 = arith.addf %101, %102 {DataUse} : tensor<128xf32>
// CHECK-NEXT:         %104 = tt.expand_dims %100 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:         %105 = tt.broadcast %104 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:         %106 = arith.mulf %105, %arg8 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:         %107 = arith.truncf %98 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:         %extracted_slice_16 = tensor.extract_slice %107[%2, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:         %108 = tensor.empty() : tensor<128x128xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_16 into %108[%2, 0] [64, 128] [1, 1] {cv_communication_slice} : tensor<64x128xbf16> into tensor<128x128xbf16>
// CHECK-NEXT:         %109 = tt.dot %inserted_slice, %85, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
// CHECK-NEXT:         %110 = arith.addf %109, %106 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
// CHECK-NEXT:         scf.yield %110, %94, %103 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
// CHECK-NEXT:       } {DataUse}
// CHECK-NEXT:       %57 = tt.expand_dims %56#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
// CHECK-NEXT:       %58 = tt.broadcast %57 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
// CHECK-NEXT:       %59 = arith.divf %56#0, %58 {DataUse} : tensor<128x128xf32>
// CHECK-NEXT:       %60 = math.log %56#2 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %61 = arith.addf %60, %56#1 {DataUse} : tensor<128xf32>
// CHECK-NEXT:       %62 = arith.truncf %59 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %62[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xbf16> to tensor<64x128xbf16>
// CHECK-NEXT:       %extracted_slice_11 = tensor.extract_slice %33[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128x!tt.ptr<bf16>> to tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_12 = tensor.extract_slice %42[%3, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : tensor<128x128xi1> to tensor<64x128xi1>
// CHECK-NEXT:       tt.store %extracted_slice_11, %extracted_slice, %extracted_slice_12 {tiled_op} : tensor<64x128x!tt.ptr<bf16>>
// CHECK-NEXT:       %extracted_slice_13 = tensor.extract_slice %61[%4] [64] [1] {to_be_bubbled_slice} : tensor<128xf32> to tensor<64xf32>
// CHECK-NEXT:       %extracted_slice_14 = tensor.extract_slice %39[%4] [64] [1] {to_be_bubbled_slice} : tensor<128x!tt.ptr<f32>> to tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:       %extracted_slice_15 = tensor.extract_slice %41[%4] [64] [1] {to_be_bubbled_slice} : tensor<128xi1> to tensor<64xi1>
// CHECK-NEXT:       tt.store %extracted_slice_14, %extracted_slice_13, %extracted_slice_15 {tiled_op} : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @kernel_sdpa_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c160_i32 = arith.constant 160 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant {MetaUse} dense<0> : tensor<128x128xi8>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant {MetaUse} dense<0.000000e+00> : tensor<128x128xbf16>
    %cst_1 = arith.constant {DataUse} dense<1.000000e+06> : tensor<128x128xf32>
    %cst_2 = arith.constant {DataUse} dense<1.000000e+00> : tensor<128x128xf32>
    %cst_3 = arith.constant {DataUse} dense<0.0883883461> : tensor<128x128xf32>
    %cst_4 = arith.constant {MetaUse} dense<4096> : tensor<1x128xi32>
    %cst_5 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<128xf32>
    %cst_6 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128xf32>
    %cst_7 = arith.constant {DataUse} dense<0.000000e+00> : tensor<128x128xf32>
    %cst_8 = arith.constant {MetaUse} dense<4096> : tensor<128xi32>
    %cst_9 = arith.constant {MetaUse} dense<4096> : tensor<128x1xi32>
    %c4096_i32 = arith.constant 4096 : i32
    %c20480_i32 = arith.constant 20480 : i32
    %cst_10 = arith.constant {MetaUse} dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c2621440_i32 = arith.constant 2621440 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {MetaUse, end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %4 = tt.broadcast %3 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
    %5 = tt.splat %arg5 {MetaUse} : !tt.ptr<i1> -> tensor<128x1x!tt.ptr<i1>>
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
      %15 = tt.splat %14 {MetaUse} : i32 -> tensor<128xi32>
      %16 = arith.addi %15, %2 {MetaUse} : tensor<128xi32>
      %17 = tt.expand_dims %16 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %18 = arith.muli %17, %cst_10 {MetaUse} : tensor<128x1xi32>
      %19 = tt.splat %13 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %20 = tt.addptr %19, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %21 = tt.broadcast %20 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %22 = tt.addptr %21, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %23 = tt.addptr %arg3, %10 : !tt.ptr<bf16>, i32
      %24 = tt.addptr %23, %12 : !tt.ptr<bf16>, i32
      %25 = tt.splat %24 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %26 = tt.addptr %25, %18 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
      %27 = tt.broadcast %26 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
      %28 = tt.addptr %27, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
      %29 = arith.muli %6, %c20480_i32 : i32
      %30 = tt.addptr %arg4, %29 : !tt.ptr<f32>, i32
      %31 = arith.muli %8, %c4096_i32 : i32
      %32 = tt.addptr %30, %31 : !tt.ptr<f32>, i32
      %33 = tt.splat %32 {MetaUse} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %34 = tt.addptr %33, %16 {MetaUse} : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %35 = arith.cmpi slt, %17, %cst_9 {MetaUse} : tensor<128x1xi32>
      %36 = arith.cmpi slt, %16, %cst_8 {MetaUse} : tensor<128xi32>
      %37 = tt.broadcast %35 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
      %38 = tt.load %22, %37, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
      %39 = arith.muli %6, %c524288_i32 : i32
      %40 = tt.addptr %arg1, %39 : !tt.ptr<bf16>, i32
      %41 = arith.divsi %8, %c5_i32 : i32
      %42 = arith.muli %41, %c524288_i32 : i32
      %43 = tt.addptr %40, %42 : !tt.ptr<bf16>, i32
      %44 = tt.splat %43 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %45 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32
      %46 = tt.addptr %45, %42 : !tt.ptr<bf16>, i32
      %47 = tt.splat %46 {MetaUse} : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
      %48 = arith.muli %17, %cst_9 {MetaUse} : tensor<128x1xi32>
      %49 = tt.addptr %5, %48 {MetaUse} : tensor<128x1x!tt.ptr<i1>>, tensor<128x1xi32>
      %50 = tt.broadcast %49 {MetaUse} : tensor<128x1x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i1>>
      %51:3 = scf.for %arg7 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg8 = %cst_7, %arg9 = %cst_5, %arg10 = %cst_6) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
        %58 = arith.muli %arg7, %c128_i32 : i32
        %59 = tt.splat %58 {MetaUse} : i32 -> tensor<128xi32>
        %60 = arith.addi %59, %2 {MetaUse} : tensor<128xi32>
        %61 = tt.expand_dims %60 {MetaUse, axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %62 = arith.muli %61, %cst_10 {MetaUse} : tensor<128x1xi32>
        %63 = tt.addptr %44, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %64 = tt.broadcast %63 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %65 = tt.addptr %64, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %66 = tt.addptr %47, %62 {MetaUse} : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
        %67 = tt.broadcast %66 {MetaUse} : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x128x!tt.ptr<bf16>>
        %68 = tt.addptr %67, %4 {MetaUse} : tensor<128x128x!tt.ptr<bf16>>, tensor<128x128xi32>
        %69 = tt.expand_dims %60 {MetaUse, axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %70 = tt.broadcast %69 {MetaUse} : tensor<1x128xi32> -> tensor<128x128xi32>
        %71 = tt.addptr %50, %70 {MetaUse} : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
        %72 = arith.cmpi slt, %61, %cst_9 {MetaUse} : tensor<128x1xi32>
        %73 = arith.cmpi slt, %69, %cst_4 {MetaUse} : tensor<1x128xi32>
        %74 = tt.broadcast %73 {MetaUse} : tensor<1x128xi1> -> tensor<128x128xi1>
        %75 = arith.andi %37, %74 {MetaUse} : tensor<128x128xi1>
        %76 = tt.bitcast %71 {MetaUse} : tensor<128x128x!tt.ptr<i1>> -> tensor<128x128x!tt.ptr<i8>>
        %77 = tt.load %76, %75, %cst {DataUse, was_bool_to_int8 = true} : tensor<128x128x!tt.ptr<i8>>
        %78 = tt.broadcast %72 {MetaUse} : tensor<128x1xi1> -> tensor<128x128xi1>
        %79 = tt.load %65, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
        %80 = tt.load %68, %78, %cst_0 {DataUse} : tensor<128x128x!tt.ptr<bf16>>
        %81 = tt.trans %79 {DataUse, order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %82 = tt.dot %38, %81, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %83 = arith.mulf %82, %cst_3 {DataUse} : tensor<128x128xf32>
        %84 = arith.sitofp %77 {DataUse} : tensor<128x128xi8> to tensor<128x128xf32>
        %85 = arith.subf %cst_2, %84 {DataUse} : tensor<128x128xf32>
        %86 = arith.mulf %85, %cst_1 {DataUse} : tensor<128x128xf32>
        %87 = arith.subf %83, %86 {DataUse} : tensor<128x128xf32>
        %88 = "tt.reduce"(%87) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.maximumf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %89 = arith.maximumf %arg9, %88 {DataUse} : tensor<128xf32>
        %90 = tt.expand_dims %89 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %91 = tt.broadcast %90 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %92 = arith.subf %87, %91 {DataUse} : tensor<128x128xf32>
        %93 = math.exp %92 {DataUse} : tensor<128x128xf32>
        %94 = arith.subf %arg9, %89 {DataUse} : tensor<128xf32>
        %95 = math.exp %94 {DataUse} : tensor<128xf32>
        %96 = arith.mulf %95, %arg10 {DataUse} : tensor<128xf32>
        %97 = "tt.reduce"(%93) <{axis = 1 : i32}> ({
        ^bb0(%arg11: f32, %arg12: f32):
          %105 = arith.addf %arg11, %arg12 : f32
          tt.reduce.return %105 : f32
        }) {DataUse} : (tensor<128x128xf32>) -> tensor<128xf32>
        %98 = arith.addf %96, %97 {DataUse} : tensor<128xf32>
        %99 = tt.expand_dims %95 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %100 = tt.broadcast %99 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
        %101 = arith.mulf %100, %arg8 {DataUse} : tensor<128x128xf32>
        %102 = arith.truncf %93 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
        %103 = tt.dot %102, %80, %cst_7 {DataUse, triton_cv12.normalized_dot} : tensor<128x128xbf16> * tensor<128x128xbf16> -> tensor<128x128xf32>
        %104 = arith.addf %103, %101 {DataUse, triton_cv12.add_from_dot} : tensor<128x128xf32>
        scf.yield %104, %89, %98 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>
      } {DataUse}
      %52 = tt.expand_dims %51#2 {DataUse, axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %53 = tt.broadcast %52 {DataUse} : tensor<128x1xf32> -> tensor<128x128xf32>
      %54 = arith.divf %51#0, %53 {DataUse} : tensor<128x128xf32>
      %55 = math.log %51#2 {DataUse} : tensor<128xf32>
      %56 = arith.addf %55, %51#1 {DataUse} : tensor<128xf32>
      %57 = arith.truncf %54 {DataUse} : tensor<128x128xf32> to tensor<128x128xbf16>
      tt.store %28, %57, %37 : tensor<128x128x!tt.ptr<bf16>>
      tt.store %34, %56, %36 : tensor<128x!tt.ptr<f32>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/sdpa_infer/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 32)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @_sdpa_infer_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32) attributes {noinline = false} {
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c2621440_i64 = arith.constant 2621440 : i64
// CHECK-NEXT:     %c524288_i64 = arith.constant 524288 : i64
// CHECK-NEXT:     %c4096_i64 = arith.constant 4096 : i64
// CHECK-NEXT:     %c128_i64 = arith.constant 128 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c320_i32 = arith.constant 320 : i32
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
// CHECK-NEXT:     %c262144_i32 = arith.constant 262144 : i32
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {DataUse} dense<0> : tensor<64x64xi8>
// CHECK-NEXT:     %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
// CHECK-NEXT:     %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT:     %cst_4 = arith.constant {DataUse} dense<0xFF800000> : tensor<64xf32>
// CHECK-NEXT:     %cst_5 = arith.constant {DataUse} dense<1.000000e+00> : tensor<64xf32>
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = arith.index_cast %5 : index to i32
// CHECK-NEXT:     %7 = tt.get_program_id x : i32
// CHECK-NEXT:     %8 = tt.get_num_programs x : i32
// CHECK-NEXT:     %9 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:     %10 = tt.expand_dims %9 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
// CHECK-NEXT:     %11 = arith.muli %10, %cst_3 {MetaUse} : tensor<64x1xi32>
// CHECK-NEXT:     %12 = tt.expand_dims %9 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
// CHECK-NEXT:     %13 = tt.broadcast %12 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
// CHECK-NEXT:     %14 = tt.splat %arg6 {DataUse} : f32 -> tensor<64x64xf32>
// CHECK-NEXT:     scf.for %arg7 = %7 to %c320_i32 step %8  : i32 {
// CHECK-NEXT:       %15 = arith.divsi %arg7, %c64_i32 : i32
// CHECK-NEXT:       %16 = arith.remsi %arg7, %c64_i32 : i32
// CHECK-NEXT:       %17 = arith.divsi %15, %c5_i32 : i32
// CHECK-NEXT:       %18 = arith.remsi %15, %c5_i32 : i32
// CHECK-NEXT:       %19 = arith.divsi %18, %c5_i32 : i32
// CHECK-NEXT:       %20 = arith.extsi %17 : i32 to i64
// CHECK-NEXT:       %21 = arith.muli %20, %c2621440_i64 : i64
// CHECK-NEXT:       %22 = arith.extsi %18 : i32 to i64
// CHECK-NEXT:       %23 = arith.muli %22, %c524288_i64 : i64
// CHECK-NEXT:       %24 = arith.addi %21, %23 : i64
// CHECK-NEXT:       %25 = arith.muli %20, %c524288_i64 : i64
// CHECK-NEXT:       %26 = arith.extsi %19 : i32 to i64
// CHECK-NEXT:       %27 = arith.muli %26, %c524288_i64 : i64
// CHECK-NEXT:       %28 = arith.addi %25, %27 : i64
// CHECK-NEXT:       %29 = tt.addptr %arg0, %24 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %30 = arith.muli %16, %c64_i32 : i32
// CHECK-NEXT:       %31 = tt.make_tensor_ptr %29, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%30, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %32 = tt.addptr %arg1, %28 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %33 = tt.make_tensor_ptr %32, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %34 = tt.addptr %arg2, %28 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %35 = tt.make_tensor_ptr %34, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xbf16>>
// CHECK-NEXT:       %36 = tt.addptr %arg5, %24 : !tt.ptr<bf16>, i64
// CHECK-NEXT:       %37 = arith.addi %30, %6 : i32
// CHECK-NEXT:       %38 = tt.make_tensor_ptr %36, [%c4096_i64, %c128_i64], [%c128_i64, %c1_i64], [%37, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<32x128xbf16>>
// CHECK-NEXT:       %39 = tt.load %31 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       %40 = arith.muli %16, %c262144_i32 : i32
// CHECK-NEXT:       %41 = tt.addptr %arg3, %40 : !tt.ptr<i1>, i32
// CHECK-NEXT:       %42:5 = scf.for %arg8 = %c0_i32 to %c4096_i32 step %c64_i32 iter_args(%arg9 = %cst_5, %arg10 = %cst, %arg11 = %cst_4, %arg12 = %35, %arg13 = %33) -> (tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>)  : i32 {
// CHECK-NEXT:         %47 = tt.addptr %41, %arg8 : !tt.ptr<i1>, i32
// CHECK-NEXT:         %48 = tt.splat %47 {MetaUse} : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
// CHECK-NEXT:         %49 = tt.addptr %48, %11 {MetaUse} : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
// CHECK-NEXT:         %50 = tt.broadcast %49 {MetaUse} : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
// CHECK-NEXT:         %51 = tt.addptr %50, %13 {MetaUse} : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
// CHECK-NEXT:         %52 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:         %53 = tt.trans %52 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
// CHECK-NEXT:         %54 = tt.dot %39, %53, %cst_2 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
// CHECK-NEXT:         %55 = arith.mulf %54, %14 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %56 = tt.bitcast %51 {MetaUse} : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:         %57 = tt.load %56 {DataUse, was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
// CHECK-NEXT:         %58 = arith.cmpi ne, %57, %cst_1 {DataUse} : tensor<64x64xi8>
// CHECK-NEXT:         %59 = arith.select %58, %55, %cst_0 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK-NEXT:         %60 = "tt.reduce"(%59) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg14: f32, %arg15: f32):
// CHECK-NEXT:           %82 = arith.maximumf %arg14, %arg15 : f32
// CHECK-NEXT:           tt.reduce.return %82 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %61 = arith.maximumf %arg11, %60 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %62 = tt.expand_dims %61 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %63 = tt.broadcast %62 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
// CHECK-NEXT:         %64 = arith.subf %59, %63 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %65 = math.exp %64 {DataUse} : tensor<64x64xf32>
// CHECK-NEXT:         %66 = arith.truncf %65 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
// CHECK-NEXT:         %67 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:         %68 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
// CHECK-NEXT:         ^bb0(%arg14: f32, %arg15: f32):
// CHECK-NEXT:           %82 = arith.addf %arg14, %arg15 : f32
// CHECK-NEXT:           tt.reduce.return %82 : f32
// CHECK-NEXT:         }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
// CHECK-NEXT:         %69 = arith.subf %arg11, %61 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %70 = math.exp %69 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %71 = arith.mulf %arg9, %70 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %72 = arith.addf %71, %68 {DataUse} : tensor<64xf32>
// CHECK-NEXT:         %73 = tt.expand_dims %70 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:         %74 = tt.broadcast %73 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:         %75 = arith.mulf %arg10, %74 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:         %extracted_slice_6 = tensor.extract_slice %66[%3, 0] [32, 64] [1, 1] {to_be_bubbled_slice} : tensor<64x64xbf16> to tensor<32x64xbf16>
// CHECK-NEXT:         %76 = tensor.empty() : tensor<64x64xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice_6 into %76[%3, 0] [32, 64] [1, 1] {cv_communication_slice} : tensor<32x64xbf16> into tensor<64x64xbf16>
// CHECK-NEXT:         %77 = tt.dot %inserted_slice, %67, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
// CHECK-NEXT:         %78 = arith.addf %77, %75 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
// CHECK-NEXT:         %extracted_slice_7 = tensor.extract_slice %78[%4, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xf32> to tensor<32x128xf32>
// CHECK-NEXT:         annotation.mark %extracted_slice_7 {hivm.tile_mix_cube_num = 2 : i32, tiled_op} : tensor<32x128xf32>
// CHECK-NEXT:         %79 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
// CHECK-NEXT:         %80 = tt.advance %arg13, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
// CHECK-NEXT:         %extracted_slice_8 = tensor.extract_slice %61[%2] [32] [1] {to_be_bubbled_slice} : tensor<64xf32> to tensor<32xf32>
// CHECK-NEXT:         %81 = tensor.empty() : tensor<64xf32>
// CHECK-NEXT:         %inserted_slice_9 = tensor.insert_slice %extracted_slice_8 into %81[%2] [32] [1] {to_be_eliminated_slice} : tensor<32xf32> into tensor<64xf32>
// CHECK-NEXT:         scf.yield %72, %78, %inserted_slice_9, %79, %80 : tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>
// CHECK-NEXT:       } {DataUse, tiled_op, tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
// CHECK-NEXT:       %43 = tt.expand_dims %42#0 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:       %44 = tt.broadcast %43 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
// CHECK-NEXT:       %45 = arith.divf %42#1, %44 {DataUse} : tensor<64x128xf32>
// CHECK-NEXT:       %46 = arith.truncf %45 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
// CHECK-NEXT:       %extracted_slice = tensor.extract_slice %46[%5, 0] [32, 128] [1, 1] {to_be_bubbled_slice} : tensor<64x128xbf16> to tensor<32x128xbf16>
// CHECK-NEXT:       tt.store %38, %extracted_slice {tiled_op} : !tt.ptr<tensor<32x128xbf16>>
// CHECK-NEXT:     }
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @_sdpa_infer_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: f32) attributes {noinline = false} {
    %cst = arith.constant {DataUse} dense<1.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant {DataUse} dense<0xFF800000> : tensor<64xf32>
    %c4096_i32 = arith.constant 4096 : i32
    %cst_1 = arith.constant {MetaUse} dense<4096> : tensor<64x1xi32>
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x64xf32>
    %cst_3 = arith.constant {DataUse} dense<0> : tensor<64x64xi8>
    %cst_4 = arith.constant {DataUse} dense<-1.000000e+06> : tensor<64x64xf32>
    %c262144_i32 = arith.constant 262144 : i32
    %cst_5 = arith.constant {DataUse} dense<0.000000e+00> : tensor<64x128xf32>
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
    %2 = tt.make_range {MetaUse, end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.expand_dims %2 {MetaUse, axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %4 = arith.muli %3, %cst_1 {MetaUse} : tensor<64x1xi32>
    %5 = tt.expand_dims %2 {MetaUse, axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %6 = tt.broadcast %5 {MetaUse} : tensor<1x64xi32> -> tensor<64x64xi32>
    %7 = tt.splat %arg6 {DataUse} : f32 -> tensor<64x64xf32>
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
      %31 = tt.load %24 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
      %32 = arith.muli %9, %c262144_i32 : i32
      %33 = tt.addptr %arg3, %32 : !tt.ptr<i1>, i32
      %34:5 = scf.for %arg8 = %c0_i32 to %c4096_i32 step %c64_i32 iter_args(%arg9 = %cst, %arg10 = %cst_5, %arg11 = %cst_0, %arg12 = %28, %arg13 = %26) -> (tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>)  : i32 {
        %39 = tt.addptr %33, %arg8 : !tt.ptr<i1>, i32
        %40 = tt.splat %39 {MetaUse} : !tt.ptr<i1> -> tensor<64x1x!tt.ptr<i1>>
        %41 = tt.addptr %40, %4 {MetaUse} : tensor<64x1x!tt.ptr<i1>>, tensor<64x1xi32>
        %42 = tt.broadcast %41 {MetaUse} : tensor<64x1x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i1>>
        %43 = tt.addptr %42, %6 {MetaUse} : tensor<64x64x!tt.ptr<i1>>, tensor<64x64xi32>
        %44 = tt.load %arg13 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
        %45 = tt.trans %44 {DataUse, order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
        %46 = tt.dot %31, %45, %cst_2 {DataUse, triton_cv12.normalized_dot} : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
        %47 = arith.mulf %46, %7 {DataUse} : tensor<64x64xf32>
        %48 = tt.bitcast %43 {MetaUse} : tensor<64x64x!tt.ptr<i1>> -> tensor<64x64x!tt.ptr<i8>>
        %49 = tt.load %48 {DataUse, was_bool_to_int8 = true} : tensor<64x64x!tt.ptr<i8>>
        %50 = arith.cmpi ne, %49, %cst_3 {DataUse} : tensor<64x64xi8>
        %51 = arith.select %50, %47, %cst_4 {DataUse} : tensor<64x64xi1>, tensor<64x64xf32>
        %52 = "tt.reduce"(%51) <{axis = 1 : i32}> ({
        ^bb0(%arg14: f32, %arg15: f32):
          %72 = arith.maximumf %arg14, %arg15 : f32
          tt.reduce.return %72 : f32
        }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
        %53 = arith.maximumf %arg11, %52 {DataUse} : tensor<64xf32>
        %54 = tt.expand_dims %53 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %55 = tt.broadcast %54 {DataUse} : tensor<64x1xf32> -> tensor<64x64xf32>
        %56 = arith.subf %51, %55 {DataUse} : tensor<64x64xf32>
        %57 = math.exp %56 {DataUse} : tensor<64x64xf32>
        %58 = arith.truncf %57 {DataUse} : tensor<64x64xf32> to tensor<64x64xbf16>
        %59 = tt.load %arg12 {DataUse} : !tt.ptr<tensor<64x128xbf16>>
        %60 = "tt.reduce"(%57) <{axis = 1 : i32}> ({
        ^bb0(%arg14: f32, %arg15: f32):
          %72 = arith.addf %arg14, %arg15 : f32
          tt.reduce.return %72 : f32
        }) {DataUse} : (tensor<64x64xf32>) -> tensor<64xf32>
        %61 = arith.subf %arg11, %53 {DataUse} : tensor<64xf32>
        %62 = math.exp %61 {DataUse} : tensor<64xf32>
        %63 = arith.mulf %arg9, %62 {DataUse} : tensor<64xf32>
        %64 = arith.addf %63, %60 {DataUse} : tensor<64xf32>
        %65 = tt.expand_dims %62 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
        %66 = tt.broadcast %65 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
        %67 = arith.mulf %arg10, %66 {DataUse} : tensor<64x128xf32>
        %68 = tt.dot %58, %59, %cst_5 {DataUse, triton_cv12.normalized_dot} : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
        %69 = arith.addf %68, %67 {DataUse, triton_cv12.add_from_dot} : tensor<64x128xf32>
        annotation.mark %69 {hivm.tile_mix_cube_num = 2 : i32} : tensor<64x128xf32>
        %70 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
        %71 = tt.advance %arg13, [%c64_i32, %c0_i32] : <tensor<64x128xbf16>>
        scf.yield %64, %69, %53, %70, %71 : tensor<64xf32>, tensor<64x128xf32>, tensor<64xf32>, !tt.ptr<tensor<64x128xbf16>>, !tt.ptr<tensor<64x128xbf16>>
      } {DataUse, tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
      %35 = tt.expand_dims %34#0 {DataUse, axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %36 = tt.broadcast %35 {DataUse} : tensor<64x1xf32> -> tensor<64x128xf32>
      %37 = arith.divf %34#1, %36 {DataUse} : tensor<64x128xf32>
      %38 = arith.truncf %37 {DataUse} : tensor<64x128xf32> to tensor<64x128xbf16>
      tt.store %30, %38 : !tt.ptr<tensor<64x128xbf16>>
    }
    tt.return
  }
}


// -----
// Source: gaoyou/0701/solve_tril/sep_md.mlir
// CHECK: #map = affine_map<()[s0] -> (s0 * 8)>
// CHECK-NEXT: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NEXT:   tt.func public @merge_16x16_to_64x64_inverse_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %c64_i32 = arith.constant 64 : i32
// CHECK-NEXT:     %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:     %c2048_i64 = arith.constant 2048 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c16_i32 = arith.constant 16 : i32
// CHECK-NEXT:     %c48_i32 = arith.constant 48 : i32
// CHECK-NEXT:     %cst = arith.constant {DataUse} dense<0.000000e+00> : tensor<16x16xf32>
// CHECK-NEXT:     %cst_0 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16xf32>
// CHECK-NEXT:     %cst_1 = arith.constant {MetaUse} dense<16> : tensor<16xi32>
// CHECK-NEXT:     %cst_2 = arith.constant {MetaUse} dense<32> : tensor<16xi32>
// CHECK-NEXT:     %cst_3 = arith.constant {MetaUse} dense<48> : tensor<16xi32>
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c2048_i32 = arith.constant 2048 : i32
// CHECK-NEXT:     %c18_i32 = arith.constant 18 : i32
// CHECK-NEXT:     %c34_i32 = arith.constant 34 : i32
// CHECK-NEXT:     %c50_i32 = arith.constant 50 : i32
// CHECK-NEXT:     %0 = hivm.hir.get_sub_block_idx -> i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = affine.apply #map()[%1]
// CHECK-NEXT:     %3 = affine.apply #map()[%1]
// CHECK-NEXT:     %4 = affine.apply #map()[%1]
// CHECK-NEXT:     %5 = affine.apply #map()[%1]
// CHECK-NEXT:     %6 = affine.apply #map()[%1]
// CHECK-NEXT:     %7 = affine.apply #map()[%1]
// CHECK-NEXT:     %8 = affine.apply #map()[%1]
// CHECK-NEXT:     %9 = affine.apply #map()[%1]
// CHECK-NEXT:     %10 = affine.apply #map()[%1]
// CHECK-NEXT:     %11 = affine.apply #map()[%1]
// CHECK-NEXT:     %12 = affine.apply #map()[%1]
// CHECK-NEXT:     %13 = affine.apply #map()[%1]
// CHECK-NEXT:     %14 = affine.apply #map()[%1]
// CHECK-NEXT:     %15 = affine.apply #map()[%1]
// CHECK-NEXT:     %16 = affine.apply #map()[%1]
// CHECK-NEXT:     %17 = affine.apply #map()[%1]
// CHECK-NEXT:     %18 = affine.apply #map()[%1]
// CHECK-NEXT:     %19 = arith.index_cast %18 : index to i32
// CHECK-NEXT:     %20 = affine.apply #map()[%1]
// CHECK-NEXT:     %21 = arith.index_cast %20 : index to i32
// CHECK-NEXT:     %22 = affine.apply #map()[%1]
// CHECK-NEXT:     %23 = arith.index_cast %22 : index to i32
// CHECK-NEXT:     %24 = affine.apply #map()[%1]
// CHECK-NEXT:     %25 = arith.index_cast %24 : index to i32
// CHECK-NEXT:     %26 = affine.apply #map()[%1]
// CHECK-NEXT:     %27 = arith.index_cast %26 : index to i32
// CHECK-NEXT:     %28 = affine.apply #map()[%1]
// CHECK-NEXT:     %29 = arith.index_cast %28 : index to i32
// CHECK-NEXT:     %30 = affine.apply #map()[%1]
// CHECK-NEXT:     %31 = arith.index_cast %30 : index to i32
// CHECK-NEXT:     %32 = affine.apply #map()[%1]
// CHECK-NEXT:     %33 = arith.index_cast %32 : index to i32
// CHECK-NEXT:     %34 = affine.apply #map()[%1]
// CHECK-NEXT:     %35 = arith.index_cast %34 : index to i32
// CHECK-NEXT:     %36 = affine.apply #map()[%1]
// CHECK-NEXT:     %37 = arith.index_cast %36 : index to i32
// CHECK-NEXT:     %38 = tt.get_program_id x : i32
// CHECK-NEXT:     %39 = tt.get_program_id y : i32
// CHECK-NEXT:     %40 = arith.divsi %39, %c32_i32 : i32
// CHECK-NEXT:     %41 = arith.remsi %39, %c32_i32 : i32
// CHECK-NEXT:     %42 = arith.muli %40, %arg2 : i32
// CHECK-NEXT:     %43 = tt.make_range {MetaUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT:     %44 = tt.make_range {DataUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK-NEXT:     %45 = tt.expand_dims %44 {DataUse, axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK-NEXT:     %46 = tt.expand_dims %44 {DataUse, axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
// CHECK-NEXT:     %47 = tt.broadcast %45 {DataUse} : tensor<16x1xi32> -> tensor<16x16xi32>
// CHECK-NEXT:     %48 = tt.broadcast %46 {DataUse} : tensor<1x16xi32> -> tensor<16x16xi32>
// CHECK-NEXT:     %49 = arith.cmpi sgt, %47, %48 {DataUse} : tensor<16x16xi32>
// CHECK-NEXT:     %50 = arith.cmpi eq, %47, %48 {DataUse} : tensor<16x16xi32>
// CHECK-NEXT:     %51 = arith.muli %42, %c32_i32 : i32
// CHECK-NEXT:     %52 = arith.addi %51, %41 : i32
// CHECK-NEXT:     %53 = arith.muli %52, %c64_i32 : i32
// CHECK-NEXT:     %54 = tt.addptr %arg0, %53 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %55 = tt.addptr %arg1, %53 : !tt.ptr<f32>, i32
// CHECK-NEXT:     %56 = arith.muli %38, %c64_i32 : i32
// CHECK-NEXT:     %57 = arith.extsi %arg2 : i32 to i64
// CHECK-NEXT:     %58 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%56, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %59 = arith.addi %56, %c16_i32 : i32
// CHECK-NEXT:     %60 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%59, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %61 = arith.addi %56, %c32_i32 : i32
// CHECK-NEXT:     %62 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%61, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %63 = arith.addi %56, %c48_i32 : i32
// CHECK-NEXT:     %64 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%63, %c48_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %65 = tt.load %58 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %66 = tt.load %60 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %67 = tt.load %62 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %68 = tt.load %64 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %69 = arith.select %49, %65, %cst {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:     %70 = arith.subf %cst, %69 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %71 = arith.select %49, %66, %cst {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:     %72 = arith.subf %cst, %71 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %73 = arith.select %49, %67, %cst {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:     %74 = arith.subf %cst, %73 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %75 = arith.select %49, %68, %cst {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:     %76 = arith.subf %cst, %75 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %77 = arith.subi %arg2, %56 : i32
// CHECK-NEXT:     %78 = arith.minsi %77, %c16_i32 : i32
// CHECK-NEXT:     %79 = scf.for %arg3 = %c2_i32 to %78 step %c1_i32 iter_args(%arg4 = %70) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:       %160 = arith.addi %56, %arg3 : i32
// CHECK-NEXT:       %161 = arith.muli %160, %c2048_i32 : i32
// CHECK-NEXT:       %162 = tt.addptr %54, %161 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %163 = tt.splat %162 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %164 = tt.addptr %163, %43 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:       %165 = tt.load %164 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %166 = arith.subf %cst_0, %165 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %167 = tt.splat %arg3 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:       %168 = arith.cmpi slt, %44, %167 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %169 = arith.select %168, %166, %cst_0 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:       %170 = tt.expand_dims %169 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:       %171 = tt.broadcast %170 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %172 = arith.mulf %171, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %173 = "tt.reduce"(%172) <{axis = 0 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:         %181 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:         tt.reduce.return %181 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:       %174 = arith.addf %169, %173 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %175 = arith.cmpi eq, %44, %167 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %176 = tt.expand_dims %175 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:       %177 = tt.expand_dims %174 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:       %178 = tt.broadcast %177 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %179 = tt.broadcast %176 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:       %180 = arith.select %179, %178, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       scf.yield %180 : tensor<16x16xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %80 = arith.minsi %77, %c32_i32 : i32
// CHECK-NEXT:     %81 = scf.for %arg3 = %c18_i32 to %80 step %c1_i32 iter_args(%arg4 = %72) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:       %160 = arith.addi %56, %arg3 : i32
// CHECK-NEXT:       %161 = arith.muli %160, %c2048_i32 : i32
// CHECK-NEXT:       %162 = tt.addptr %54, %161 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %163 = tt.splat %162 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %164 = tt.addptr %163, %43 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:       %165 = tt.addptr %164, %cst_1 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:       %166 = tt.load %165 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %167 = arith.subf %cst_0, %166 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %168 = arith.subi %arg3, %c16_i32 : i32
// CHECK-NEXT:       %169 = tt.splat %168 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:       %170 = arith.cmpi slt, %44, %169 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %171 = arith.select %170, %167, %cst_0 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:       %172 = tt.expand_dims %171 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:       %173 = tt.broadcast %172 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %174 = arith.mulf %173, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %175 = "tt.reduce"(%174) <{axis = 0 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:         %183 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:         tt.reduce.return %183 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:       %176 = arith.addf %171, %175 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %177 = arith.cmpi eq, %44, %169 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %178 = tt.expand_dims %177 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:       %179 = tt.expand_dims %176 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:       %180 = tt.broadcast %179 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %181 = tt.broadcast %178 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:       %182 = arith.select %181, %180, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       scf.yield %182 : tensor<16x16xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %82 = arith.minsi %77, %c48_i32 : i32
// CHECK-NEXT:     %83 = scf.for %arg3 = %c34_i32 to %82 step %c1_i32 iter_args(%arg4 = %74) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:       %160 = arith.addi %56, %arg3 : i32
// CHECK-NEXT:       %161 = arith.muli %160, %c2048_i32 : i32
// CHECK-NEXT:       %162 = tt.addptr %54, %161 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %163 = tt.splat %162 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %164 = tt.addptr %163, %43 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:       %165 = tt.addptr %164, %cst_2 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:       %166 = tt.load %165 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %167 = arith.subf %cst_0, %166 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %168 = arith.subi %arg3, %c32_i32 : i32
// CHECK-NEXT:       %169 = tt.splat %168 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:       %170 = arith.cmpi slt, %44, %169 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %171 = arith.select %170, %167, %cst_0 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:       %172 = tt.expand_dims %171 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:       %173 = tt.broadcast %172 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %174 = arith.mulf %173, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %175 = "tt.reduce"(%174) <{axis = 0 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:         %183 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:         tt.reduce.return %183 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:       %176 = arith.addf %171, %175 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %177 = arith.cmpi eq, %44, %169 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %178 = tt.expand_dims %177 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:       %179 = tt.expand_dims %176 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:       %180 = tt.broadcast %179 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %181 = tt.broadcast %178 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:       %182 = arith.select %181, %180, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       scf.yield %182 : tensor<16x16xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %84 = arith.minsi %77, %c64_i32 : i32
// CHECK-NEXT:     %85 = scf.for %arg3 = %c50_i32 to %84 step %c1_i32 iter_args(%arg4 = %76) -> (tensor<16x16xf32>)  : i32 {
// CHECK-NEXT:       %160 = arith.addi %56, %arg3 : i32
// CHECK-NEXT:       %161 = arith.muli %160, %c2048_i32 : i32
// CHECK-NEXT:       %162 = tt.addptr %54, %161 : !tt.ptr<f32>, i32
// CHECK-NEXT:       %163 = tt.splat %162 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %164 = tt.addptr %163, %43 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:       %165 = tt.addptr %164, %cst_3 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK-NEXT:       %166 = tt.load %165 {DataUse} : tensor<16x!tt.ptr<f32>>
// CHECK-NEXT:       %167 = arith.subf %cst_0, %166 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %168 = arith.subi %arg3, %c48_i32 : i32
// CHECK-NEXT:       %169 = tt.splat %168 {DataUse} : i32 -> tensor<16xi32>
// CHECK-NEXT:       %170 = arith.cmpi slt, %44, %169 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %171 = arith.select %170, %167, %cst_0 {DataUse} : tensor<16xi1>, tensor<16xf32>
// CHECK-NEXT:       %172 = tt.expand_dims %171 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
// CHECK-NEXT:       %173 = tt.broadcast %172 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %174 = arith.mulf %173, %arg4 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:       %175 = "tt.reduce"(%174) <{axis = 0 : i32}> ({
// CHECK-NEXT:       ^bb0(%arg5: f32, %arg6: f32):
// CHECK-NEXT:         %183 = arith.addf %arg5, %arg6 : f32
// CHECK-NEXT:         tt.reduce.return %183 : f32
// CHECK-NEXT:       }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
// CHECK-NEXT:       %176 = arith.addf %171, %175 {DataUse} : tensor<16xf32>
// CHECK-NEXT:       %177 = arith.cmpi eq, %44, %169 {DataUse} : tensor<16xi32>
// CHECK-NEXT:       %178 = tt.expand_dims %177 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
// CHECK-NEXT:       %179 = tt.expand_dims %176 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
// CHECK-NEXT:       %180 = tt.broadcast %179 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:       %181 = tt.broadcast %178 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
// CHECK-NEXT:       %182 = arith.select %181, %180, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK-NEXT:       scf.yield %182 : tensor<16x16xf32>
// CHECK-NEXT:     } {DataUse}
// CHECK-NEXT:     %86 = arith.uitofp %50 {DataUse} : tensor<16x16xi1> to tensor<16x16xf32>
// CHECK-NEXT:     %87 = arith.addf %79, %86 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %88 = arith.addf %81, %86 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %89 = arith.addf %83, %86 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %90 = arith.addf %85, %86 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %91 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%59, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %92 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%61, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %93 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%61, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %94 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%63, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %95 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%63, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %96 = tt.make_tensor_ptr %54, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%63, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
// CHECK-NEXT:     %97 = tt.load %91 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %98 = tt.load %92 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %99 = tt.load %93 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %100 = tt.load %94 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %101 = tt.load %95 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %102 = tt.load %96 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
// CHECK-NEXT:     %extracted_slice = tensor.extract_slice %88[0, %2] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %103 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice = tensor.insert_slice %extracted_slice into %103[0, %2] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %104 = tt.dot %inserted_slice, %97, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_4 = tensor.extract_slice %87[0, %3] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %105 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_5 = tensor.insert_slice %extracted_slice_4 into %105[0, %3] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %106 = tt.dot %104, %inserted_slice_5, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %107 = arith.subf %cst, %106 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_6 = tensor.extract_slice %89[0, %4] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %108 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_7 = tensor.insert_slice %extracted_slice_6 into %108[0, %4] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %109 = tt.dot %inserted_slice_7, %99, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_8 = tensor.extract_slice %88[0, %5] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %110 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_9 = tensor.insert_slice %extracted_slice_8 into %110[0, %5] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %111 = tt.dot %109, %inserted_slice_9, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %112 = arith.subf %cst, %111 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_10 = tensor.extract_slice %90[0, %6] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %113 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_11 = tensor.insert_slice %extracted_slice_10 into %113[0, %6] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %114 = tt.dot %inserted_slice_11, %102, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_12 = tensor.extract_slice %89[0, %7] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %115 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_13 = tensor.insert_slice %extracted_slice_12 into %115[0, %7] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %116 = tt.dot %114, %inserted_slice_13, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %117 = arith.subf %cst, %116 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_14 = tensor.extract_slice %87[0, %8] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %118 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_15 = tensor.insert_slice %extracted_slice_14 into %118[0, %8] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %119 = tt.dot %98, %inserted_slice_15, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_16 = tensor.extract_slice %107[%9, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     %120 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_17 = tensor.insert_slice %extracted_slice_16 into %120[%9, 0] [8, 16] [1, 1] {cv_communication_slice} : tensor<8x16xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %121 = tt.dot %99, %inserted_slice_17, %119 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_18 = tensor.extract_slice %89[0, %10] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %122 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_19 = tensor.insert_slice %extracted_slice_18 into %122[0, %10] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %123 = tt.dot %inserted_slice_19, %121, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %124 = arith.subf %cst, %123 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_20 = tensor.extract_slice %88[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %125 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_21 = tensor.insert_slice %extracted_slice_20 into %125[0, %11] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %126 = tt.dot %101, %inserted_slice_21, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_22 = tensor.extract_slice %112[%12, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     %127 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_23 = tensor.insert_slice %extracted_slice_22 into %127[%12, 0] [8, 16] [1, 1] {cv_communication_slice} : tensor<8x16xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %128 = tt.dot %102, %inserted_slice_23, %126 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_24 = tensor.extract_slice %90[0, %13] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %129 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %129[0, %13] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %130 = tt.dot %inserted_slice_25, %128, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %131 = arith.subf %cst, %130 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_26 = tensor.extract_slice %87[0, %14] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %132 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_27 = tensor.insert_slice %extracted_slice_26 into %132[0, %14] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %133 = tt.dot %100, %inserted_slice_27, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_28 = tensor.extract_slice %107[%15, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     %134 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_29 = tensor.insert_slice %extracted_slice_28 into %134[%15, 0] [8, 16] [1, 1] {cv_communication_slice} : tensor<8x16xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %135 = tt.dot %101, %inserted_slice_29, %133 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_30 = tensor.extract_slice %124[%16, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     %136 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_31 = tensor.insert_slice %extracted_slice_30 into %136[%16, 0] [8, 16] [1, 1] {cv_communication_slice} : tensor<8x16xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %137 = tt.dot %102, %inserted_slice_31, %135 {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %extracted_slice_32 = tensor.extract_slice %90[0, %17] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     %138 = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:     %inserted_slice_33 = tensor.insert_slice %extracted_slice_32 into %138[0, %17] [16, 8] [1, 1] {cv_communication_slice} : tensor<16x8xf32> into tensor<16x16xf32>
// CHECK-NEXT:     %139 = tt.dot %inserted_slice_33, %137, %cst {DataUse, tiled_op, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK-NEXT:     %140 = arith.subf %cst, %139 {DataUse} : tensor<16x16xf32>
// CHECK-NEXT:     %141 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%56, %19] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:     %142 = arith.addi %21, %c16_i32 : i32
// CHECK-NEXT:     %143 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%59, %142] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:     %144 = arith.addi %23, %c32_i32 : i32
// CHECK-NEXT:     %145 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%61, %144] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:     %146 = arith.addi %25, %c48_i32 : i32
// CHECK-NEXT:     %147 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%63, %146] {order = array<i32: 1, 0>, tiled_op} : <tensor<16x8xf32>>
// CHECK-NEXT:     %148 = arith.addi %59, %27 : i32
// CHECK-NEXT:     %149 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%148, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:     %150 = arith.addi %61, %29 : i32
// CHECK-NEXT:     %151 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%150, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:     %152 = arith.addi %61, %31 : i32
// CHECK-NEXT:     %153 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%152, %c16_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:     %154 = arith.addi %63, %33 : i32
// CHECK-NEXT:     %155 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%154, %c0_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:     %156 = arith.addi %63, %35 : i32
// CHECK-NEXT:     %157 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%156, %c16_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:     %158 = arith.addi %63, %37 : i32
// CHECK-NEXT:     %159 = tt.make_tensor_ptr %55, [%57, %c64_i64], [%c2048_i64, %c1_i64], [%158, %c32_i32] {order = array<i32: 1, 0>, tiled_op} : <tensor<8x16xf32>>
// CHECK-NEXT:     %extracted_slice_34 = tensor.extract_slice %87[0, %18] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     tt.store %141, %extracted_slice_34 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:     %extracted_slice_35 = tensor.extract_slice %88[0, %20] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     tt.store %143, %extracted_slice_35 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:     %extracted_slice_36 = tensor.extract_slice %89[0, %22] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     tt.store %145, %extracted_slice_36 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:     %extracted_slice_37 = tensor.extract_slice %90[0, %24] [16, 8] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<16x8xf32>
// CHECK-NEXT:     tt.store %147, %extracted_slice_37 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<16x8xf32>>
// CHECK-NEXT:     %extracted_slice_38 = tensor.extract_slice %107[%26, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     tt.store %149, %extracted_slice_38 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:     %extracted_slice_39 = tensor.extract_slice %124[%28, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     tt.store %151, %extracted_slice_39 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:     %extracted_slice_40 = tensor.extract_slice %112[%30, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     tt.store %153, %extracted_slice_40 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:     %extracted_slice_41 = tensor.extract_slice %140[%32, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     tt.store %155, %extracted_slice_41 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:     %extracted_slice_42 = tensor.extract_slice %131[%34, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     tt.store %157, %extracted_slice_42 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:     %extracted_slice_43 = tensor.extract_slice %117[%36, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
// CHECK-NEXT:     tt.store %159, %extracted_slice_43 {boundaryCheck = array<i32: 0, 1>, tiled_op} : !tt.ptr<tensor<8x16xf32>>
// CHECK-NEXT:     tt.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @merge_16x16_to_64x64_inverse_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
    %c50_i32 = arith.constant 50 : i32
    %c34_i32 = arith.constant 34 : i32
    %c18_i32 = arith.constant 18 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant {MetaUse} dense<48> : tensor<16xi32>
    %cst_0 = arith.constant {MetaUse} dense<32> : tensor<16xi32>
    %cst_1 = arith.constant {MetaUse} dense<16> : tensor<16xi32>
    %cst_2 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16xf32>
    %cst_3 = arith.constant {DataUse} dense<0.000000e+00> : tensor<16x16xf32>
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
    %5 = tt.make_range {MetaUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = tt.make_range {DataUse, end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %7 = tt.expand_dims %6 {DataUse, axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %8 = tt.expand_dims %6 {DataUse, axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %9 = tt.broadcast %7 {DataUse} : tensor<16x1xi32> -> tensor<16x16xi32>
    %10 = tt.broadcast %8 {DataUse} : tensor<1x16xi32> -> tensor<16x16xi32>
    %11 = arith.cmpi sgt, %9, %10 {DataUse} : tensor<16x16xi32>
    %12 = arith.cmpi eq, %9, %10 {DataUse} : tensor<16x16xi32>
    %13 = arith.muli %4, %c32_i32 : i32
    %14 = arith.addi %13, %3 : i32
    %15 = arith.muli %14, %c64_i32 : i32
    %16 = tt.addptr %arg0, %15 : !tt.ptr<f32>, i32
    %17 = tt.addptr %arg1, %15 : !tt.ptr<f32>, i32
    %18 = arith.muli %0, %c64_i32 : i32
    %19 = arith.extsi %arg2 : i32 to i64
    %20 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%18, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %21 = arith.addi %18, %c16_i32 : i32
    %22 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %23 = arith.addi %18, %c32_i32 : i32
    %24 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %25 = arith.addi %18, %c48_i32 : i32
    %26 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c48_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %27 = tt.load %20 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %28 = tt.load %22 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %29 = tt.load %24 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %30 = tt.load %26 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %31 = arith.select %11, %27, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
    %32 = arith.subf %cst_3, %31 {DataUse} : tensor<16x16xf32>
    %33 = arith.select %11, %28, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
    %34 = arith.subf %cst_3, %33 {DataUse} : tensor<16x16xf32>
    %35 = arith.select %11, %29, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
    %36 = arith.subf %cst_3, %35 {DataUse} : tensor<16x16xf32>
    %37 = arith.select %11, %30, %cst_3 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
    %38 = arith.subf %cst_3, %37 {DataUse} : tensor<16x16xf32>
    %39 = arith.subi %arg2, %18 : i32
    %40 = arith.minsi %39, %c16_i32 : i32
    %41 = scf.for %arg3 = %c2_i32 to %40 step %c1_i32 iter_args(%arg4 = %32) -> (tensor<16x16xf32>)  : i32 {
      %97 = arith.addi %18, %arg3 : i32
      %98 = arith.muli %97, %c2048_i32 : i32
      %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
      %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %102 = tt.load %101 {DataUse} : tensor<16x!tt.ptr<f32>>
      %103 = arith.subf %cst_2, %102 {DataUse} : tensor<16xf32>
      %104 = tt.splat %arg3 {DataUse} : i32 -> tensor<16xi32>
      %105 = arith.cmpi slt, %6, %104 {DataUse} : tensor<16xi32>
      %106 = arith.select %105, %103, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
      %107 = tt.expand_dims %106 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
      %108 = tt.broadcast %107 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
      %109 = arith.mulf %108, %arg4 {DataUse} : tensor<16x16xf32>
      %110 = "tt.reduce"(%109) <{axis = 0 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %118 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %118 : f32
      }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
      %111 = arith.addf %106, %110 {DataUse} : tensor<16xf32>
      %112 = arith.cmpi eq, %6, %104 {DataUse} : tensor<16xi32>
      %113 = tt.expand_dims %112 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
      %114 = tt.expand_dims %111 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
      %115 = tt.broadcast %114 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
      %116 = tt.broadcast %113 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
      %117 = arith.select %116, %115, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
      scf.yield %117 : tensor<16x16xf32>
    } {DataUse}
    %42 = arith.minsi %39, %c32_i32 : i32
    %43 = scf.for %arg3 = %c18_i32 to %42 step %c1_i32 iter_args(%arg4 = %34) -> (tensor<16x16xf32>)  : i32 {
      %97 = arith.addi %18, %arg3 : i32
      %98 = arith.muli %97, %c2048_i32 : i32
      %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
      %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %102 = tt.addptr %101, %cst_1 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %103 = tt.load %102 {DataUse} : tensor<16x!tt.ptr<f32>>
      %104 = arith.subf %cst_2, %103 {DataUse} : tensor<16xf32>
      %105 = arith.subi %arg3, %c16_i32 : i32
      %106 = tt.splat %105 {DataUse} : i32 -> tensor<16xi32>
      %107 = arith.cmpi slt, %6, %106 {DataUse} : tensor<16xi32>
      %108 = arith.select %107, %104, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
      %109 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
      %110 = tt.broadcast %109 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
      %111 = arith.mulf %110, %arg4 {DataUse} : tensor<16x16xf32>
      %112 = "tt.reduce"(%111) <{axis = 0 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %120 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %120 : f32
      }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
      %113 = arith.addf %108, %112 {DataUse} : tensor<16xf32>
      %114 = arith.cmpi eq, %6, %106 {DataUse} : tensor<16xi32>
      %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
      %116 = tt.expand_dims %113 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
      %117 = tt.broadcast %116 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
      %118 = tt.broadcast %115 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
      %119 = arith.select %118, %117, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
      scf.yield %119 : tensor<16x16xf32>
    } {DataUse}
    %44 = arith.minsi %39, %c48_i32 : i32
    %45 = scf.for %arg3 = %c34_i32 to %44 step %c1_i32 iter_args(%arg4 = %36) -> (tensor<16x16xf32>)  : i32 {
      %97 = arith.addi %18, %arg3 : i32
      %98 = arith.muli %97, %c2048_i32 : i32
      %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
      %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %102 = tt.addptr %101, %cst_0 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %103 = tt.load %102 {DataUse} : tensor<16x!tt.ptr<f32>>
      %104 = arith.subf %cst_2, %103 {DataUse} : tensor<16xf32>
      %105 = arith.subi %arg3, %c32_i32 : i32
      %106 = tt.splat %105 {DataUse} : i32 -> tensor<16xi32>
      %107 = arith.cmpi slt, %6, %106 {DataUse} : tensor<16xi32>
      %108 = arith.select %107, %104, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
      %109 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
      %110 = tt.broadcast %109 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
      %111 = arith.mulf %110, %arg4 {DataUse} : tensor<16x16xf32>
      %112 = "tt.reduce"(%111) <{axis = 0 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %120 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %120 : f32
      }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
      %113 = arith.addf %108, %112 {DataUse} : tensor<16xf32>
      %114 = arith.cmpi eq, %6, %106 {DataUse} : tensor<16xi32>
      %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
      %116 = tt.expand_dims %113 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
      %117 = tt.broadcast %116 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
      %118 = tt.broadcast %115 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
      %119 = arith.select %118, %117, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
      scf.yield %119 : tensor<16x16xf32>
    } {DataUse}
    %46 = arith.minsi %39, %c64_i32 : i32
    %47 = scf.for %arg3 = %c50_i32 to %46 step %c1_i32 iter_args(%arg4 = %38) -> (tensor<16x16xf32>)  : i32 {
      %97 = arith.addi %18, %arg3 : i32
      %98 = arith.muli %97, %c2048_i32 : i32
      %99 = tt.addptr %16, %98 : !tt.ptr<f32>, i32
      %100 = tt.splat %99 {MetaUse} : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %101 = tt.addptr %100, %5 {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %102 = tt.addptr %101, %cst {MetaUse} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %103 = tt.load %102 {DataUse} : tensor<16x!tt.ptr<f32>>
      %104 = arith.subf %cst_2, %103 {DataUse} : tensor<16xf32>
      %105 = arith.subi %arg3, %c48_i32 : i32
      %106 = tt.splat %105 {DataUse} : i32 -> tensor<16xi32>
      %107 = arith.cmpi slt, %6, %106 {DataUse} : tensor<16xi32>
      %108 = arith.select %107, %104, %cst_2 {DataUse} : tensor<16xi1>, tensor<16xf32>
      %109 = tt.expand_dims %108 {DataUse, axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
      %110 = tt.broadcast %109 {DataUse} : tensor<16x1xf32> -> tensor<16x16xf32>
      %111 = arith.mulf %110, %arg4 {DataUse} : tensor<16x16xf32>
      %112 = "tt.reduce"(%111) <{axis = 0 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %120 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %120 : f32
      }) {DataUse} : (tensor<16x16xf32>) -> tensor<16xf32>
      %113 = arith.addf %108, %112 {DataUse} : tensor<16xf32>
      %114 = arith.cmpi eq, %6, %106 {DataUse} : tensor<16xi32>
      %115 = tt.expand_dims %114 {DataUse, axis = 1 : i32} : tensor<16xi1> -> tensor<16x1xi1>
      %116 = tt.expand_dims %113 {DataUse, axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
      %117 = tt.broadcast %116 {DataUse} : tensor<1x16xf32> -> tensor<16x16xf32>
      %118 = tt.broadcast %115 {DataUse} : tensor<16x1xi1> -> tensor<16x16xi1>
      %119 = arith.select %118, %117, %arg4 {DataUse} : tensor<16x16xi1>, tensor<16x16xf32>
      scf.yield %119 : tensor<16x16xf32>
    } {DataUse}
    %48 = arith.uitofp %12 {DataUse} : tensor<16x16xi1> to tensor<16x16xf32>
    %49 = arith.addf %41, %48 {DataUse} : tensor<16x16xf32>
    %50 = arith.addf %43, %48 {DataUse} : tensor<16x16xf32>
    %51 = arith.addf %45, %48 {DataUse} : tensor<16x16xf32>
    %52 = arith.addf %47, %48 {DataUse} : tensor<16x16xf32>
    %53 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %54 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %55 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %56 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %57 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %58 = tt.make_tensor_ptr %16, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %59 = tt.load %53 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %60 = tt.load %54 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %61 = tt.load %55 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %62 = tt.load %56 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %63 = tt.load %57 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %64 = tt.load %58 {DataUse, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
    %65 = tt.dot %50, %59, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %66 = tt.dot %65, %49, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %67 = arith.subf %cst_3, %66 {DataUse} : tensor<16x16xf32>
    %68 = tt.dot %51, %61, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %69 = tt.dot %68, %50, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %70 = arith.subf %cst_3, %69 {DataUse} : tensor<16x16xf32>
    %71 = tt.dot %52, %64, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %72 = tt.dot %71, %51, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %73 = arith.subf %cst_3, %72 {DataUse} : tensor<16x16xf32>
    %74 = tt.dot %60, %49, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %75 = tt.dot %61, %67, %74 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %76 = tt.dot %51, %75, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %77 = arith.subf %cst_3, %76 {DataUse} : tensor<16x16xf32>
    %78 = tt.dot %63, %50, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %79 = tt.dot %64, %70, %78 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %80 = tt.dot %52, %79, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %81 = arith.subf %cst_3, %80 {DataUse} : tensor<16x16xf32>
    %82 = tt.dot %62, %49, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %83 = tt.dot %63, %67, %82 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %84 = tt.dot %64, %77, %83 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %85 = tt.dot %52, %84, %cst_3 {DataUse, triton_cv12.normalized_dot} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %86 = arith.subf %cst_3, %85 {DataUse} : tensor<16x16xf32>
    %87 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%18, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %88 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %89 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %90 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c48_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %91 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %92 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %93 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%23, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %94 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %95 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    %96 = tt.make_tensor_ptr %17, [%19, %c64_i64], [%c2048_i64, %c1_i64], [%25, %c32_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
    tt.store %87, %49 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %88, %50 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %89, %51 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %90, %52 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %91, %67 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %92, %77 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %93, %70 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %94, %86 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %95, %81 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.store %96, %73 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.return
  }
}

