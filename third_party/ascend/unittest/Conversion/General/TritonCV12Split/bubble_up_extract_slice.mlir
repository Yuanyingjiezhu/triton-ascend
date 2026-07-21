// RUN: triton-opt --bubble-up-extract-slice %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @if_result(
  // CHECK-SAME: %[[COND:.*]]: i1, %[[LHS:.*]]: tensor<64x128xf32>,
  // CHECK-SAME: %[[RHS:.*]]: tensor<64x128xf32>)
  // CHECK: %[[OTHER:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]]
  // CHECK: %[[RESULT:.*]] = scf.if %[[COND]] -> (tensor<32x128xf32>) {
  // CHECK:   %[[THEN_SLICE:.*]] = tensor.extract_slice %[[RHS]][0, 0] [32, 128] [1, 1]
  // CHECK:   scf.yield %[[THEN_SLICE]] : tensor<32x128xf32>
  // CHECK: } else {
  // CHECK:   %[[ELSE_SLICE:.*]] = tensor.extract_slice %[[LHS]][0, 0] [32, 128] [1, 1]
  // CHECK:   scf.yield %[[ELSE_SLICE]] : tensor<32x128xf32>
  // CHECK: }
  // CHECK: return %[[OTHER]], %[[RESULT]] : tensor<64x128xf32>, tensor<32x128xf32>
  func.func @if_result(
      %cond: i1, %lhs: tensor<64x128xf32>, %rhs: tensor<64x128xf32>)
      -> (tensor<64x128xf32>, tensor<32x128xf32>) {
    %result:2 = scf.if %cond
        -> (tensor<64x128xf32>, tensor<64x128xf32>) {
      scf.yield %lhs, %rhs
          : tensor<64x128xf32>, tensor<64x128xf32>
    } else {
      scf.yield %rhs, %lhs
          : tensor<64x128xf32>, tensor<64x128xf32>
    }
    %slice = tensor.extract_slice %result#1[0, 0] [32, 128] [1, 1]
        {to_be_bubbled_slice}
        : tensor<64x128xf32> to tensor<32x128xf32>
    return %result#0, %slice
        : tensor<64x128xf32>, tensor<32x128xf32>
  }

  // CHECK-LABEL: func.func @preserve_source_attributes
  // CHECK: %[[BROADCAST:.*]] = tt.broadcast
  // CHECK-SAME: {preserve_me, test.value = 7 : i32}
  // CHECK: return %[[BROADCAST]]
  func.func @preserve_source_attributes(%arg: tensor<1x64xf32>)
      -> tensor<32x64xf32> {
    %broadcast = tt.broadcast %arg {preserve_me, test.value = 7 : i32}
        : tensor<1x64xf32> -> tensor<64x64xf32>
    %slice = tensor.extract_slice %broadcast[0, 0] [32, 64] [1, 1]
        {to_be_bubbled_slice}
        : tensor<64x64xf32> to tensor<32x64xf32>
    return %slice : tensor<32x64xf32>
  }

  // Preserve tags without restoring the old semantic start/end attributes.
  // CHECK-LABEL: func.func @preserve_make_range_attributes
  // CHECK: tt.make_range
  // CHECK-SAME: end = 32 : i32
  // CHECK-SAME: preserve_me
  // CHECK-SAME: start = 0 : i32
  func.func @preserve_make_range_attributes() -> tensor<32xi32> {
    %range = tt.make_range
        {end = 64 : i32, preserve_me, start = 0 : i32} : tensor<64xi32>
    %slice = tensor.extract_slice %range[16] [32] [1]
        {to_be_bubbled_slice} : tensor<64xi32> to tensor<32xi32>
    return %slice : tensor<32xi32>
  }

  // CHECK-LABEL: func.func @eliminate_nested_if_yield_slices
  // CHECK: %[[INNER:.*]] = scf.if %{{.*}} -> (tensor<8x8xf32>) {
  // CHECK:   scf.yield %{{.*}} : tensor<8x8xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %{{.*}} : tensor<8x8xf32>
  // CHECK: }
  // CHECK: %[[OUTER:.*]] = scf.if %{{.*}} -> (tensor<8x8xf32>) {
  // CHECK:   scf.yield %[[INNER]] : tensor<8x8xf32>
  // CHECK: } else {
  // CHECK:   scf.yield %{{.*}} : tensor<8x8xf32>
  // CHECK: }
  // CHECK-NOT: tensor.insert_slice
  // CHECK: scf.yield %[[OUTER]] : tensor<8x8xf32>
  // CHECK: %[[RESULT:.*]] = tensor.extract_slice %{{.*}}[0, 0] [4, 8] [1, 1]
  // CHECK-SAME: {cv_communication_slice}
  // CHECK: return %[[RESULT]] : tensor<4x8xf32>
  func.func @eliminate_nested_if_yield_slices(
      %cond0: i1, %cond1: i1, %lb: index, %ub: index, %step: index,
      %arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>,
      %arg2: tensor<8x8xf32>) -> tensor<4x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf32>
    %loop = scf.for %iv = %lb to %ub step %step
        iter_args(%iter = %arg0) -> tensor<8x8xf32> {
      %inner = scf.if %cond1 -> (tensor<4x8xf32>) {
        %slice0 = tensor.extract_slice %iter[0, 0] [4, 8] [1, 1]
            {should_kept_slice}
            : tensor<8x8xf32> to tensor<4x8xf32>
        scf.yield %slice0 : tensor<4x8xf32>
      } else {
        %slice1 = tensor.extract_slice %arg1[0, 0] [4, 8] [1, 1]
            {cv_communication_slice}
            : tensor<8x8xf32> to tensor<4x8xf32>
        scf.yield %slice1 : tensor<4x8xf32>
      }
      %outer = scf.if %cond0 -> (tensor<4x8xf32>) {
        scf.yield %inner : tensor<4x8xf32>
      } else {
        %slice2 = tensor.extract_slice %arg2[0, 0] [4, 8] [1, 1]
            {cv_communication_slice}
            : tensor<8x8xf32> to tensor<4x8xf32>
        scf.yield %slice2 : tensor<4x8xf32>
      }
      %insert = tensor.insert_slice %outer into %empty[0, 0] [4, 8] [1, 1]
          {to_be_eliminated_slice}
          : tensor<4x8xf32> into tensor<8x8xf32>
      scf.yield %insert : tensor<8x8xf32>
    }
    %result = tensor.extract_slice %loop[0, 0] [4, 8] [1, 1]
        {to_be_eliminated_slice}
        : tensor<8x8xf32> to tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }
}
