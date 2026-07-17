// RUN: triton-opt --normalize-dot %s | FileCheck %s

module {
  // CHECK-LABEL: tt.func public @elementwise_accumulator
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[ZERO]]
  // CHECK-SAME: {triton_cv12.normalized_dot}
  // CHECK: %[[ADD:.*]] = arith.addf %[[DOT]], %{{.*}} {triton_cv12.add_from_dot}
  // CHECK: tt.return %[[ADD]]
  tt.func public @elementwise_accumulator(
      %a: tensor<16x32xf32>,
      %b: tensor<32x16xf32>,
      %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %dot = tt.dot %a, %b, %c
        : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
    tt.return %dot : tensor<16x16xf32>
  }

  // CHECK-LABEL: tt.func public @zero_accumulator
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[ZERO]]
  // CHECK-SAME: {triton_cv12.normalized_dot}
  // CHECK-NOT: triton_cv12.add_from_dot
  // CHECK: tt.return %[[DOT]]
  tt.func public @zero_accumulator(
      %a: tensor<16x32xf32>,
      %b: tensor<32x16xf32>) -> tensor<16x16xf32> {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32>
    %dot = tt.dot %a, %b, %zero
        : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
    tt.return %dot : tensor<16x16xf32>
  }

  // CHECK-LABEL: tt.func public @dot_chains
  // CHECK: %[[FIRST:.*]] = tt.dot
  // CHECK: %[[SECOND:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[FIRST]]
  // CHECK: %[[THIRD:.*]] = tt.dot %[[SECOND]], %{{.*}}, %{{.*}}
  // CHECK-NOT: triton_cv12.add_from_dot
  // CHECK: tt.return %[[THIRD]]
  tt.func public @dot_chains(
      %a: tensor<16x32xf32>,
      %b: tensor<32x16xf32>,
      %next: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %zero = arith.constant dense<0.0> : tensor<16x16xf32>
    %first = tt.dot %a, %b, %zero
        : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
    %second = tt.dot %next, %next, %first
        : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %third = tt.dot %second, %next, %zero
        : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return %third : tensor<16x16xf32>
  }

  // CHECK-LABEL: tt.func public @per_channel_accumulator
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[BROADCAST:.*]] = tt.broadcast
  // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[ZERO]]
  // CHECK-SAME: {triton_cv12.normalized_dot}
  // CHECK: %[[ADD:.*]] = arith.addf %[[DOT]], %[[BROADCAST]] {triton_cv12.add_from_dot}
  // CHECK: tt.return %[[ADD]]
  tt.func public @per_channel_accumulator(
      %a: tensor<16x32xf32>,
      %b: tensor<32x16xf32>,
      %bias: tensor<16xf32>) -> tensor<16x16xf32> {
    %expanded = tt.expand_dims %bias {axis = 0 : i32}
        : tensor<16xf32> -> tensor<1x16xf32>
    %broadcast = tt.broadcast %expanded
        : tensor<1x16xf32> -> tensor<16x16xf32>
    %dot = tt.dot %a, %b, %broadcast
        : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
    tt.return %dot : tensor<16x16xf32>
  }

  // CHECK-LABEL: tt.func public @dynamic_elementwise_accumulator
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ZERO]])
  // CHECK: } {hivm.matmul_limited_in_cube}
  // CHECK: %[[EXECUTED:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
  // CHECK: %[[IF:.*]] = scf.if %[[EXECUTED]]
  // CHECK: %[[ADD:.*]] = arith.addf %[[IF]], %{{.*}} {triton_cv12.add_from_dot}
  // CHECK: tt.return %[[ADD]]
  tt.func public @dynamic_elementwise_accumulator(
      %a: tensor<16x32xf32>,
      %b: tensor<32x16xf32>,
      %c: tensor<16x16xf32>,
      %upper: index) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %upper step %c1
        iter_args(%acc = %c) -> tensor<16x16xf32> {
      %dot = tt.dot %a, %b, %acc
          : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
      scf.yield %dot : tensor<16x16xf32>
    }
    tt.return %result : tensor<16x16xf32>
  }

  // CHECK-LABEL: tt.func public @reuse_guaranteed_loop
  // CHECK: %[[FIRST:.*]] = scf.for
  // CHECK: %[[SECOND:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[FIRST]])
  // CHECK-NOT: triton_cv12.add_from_dot
  // CHECK-NOT: scf.if
  // CHECK: tt.return %[[SECOND]]
  tt.func public @reuse_guaranteed_loop(
      %a: tensor<16x32xf32>,
      %b: tensor<32x16xf32>,
      %upper: index) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant dense<0.0> : tensor<16x16xf32>
    %first = scf.for %i = %c0 to %c2 step %c1
        iter_args(%acc = %zero) -> tensor<16x16xf32> {
      %dot = tt.dot %a, %b, %acc
          : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
      scf.yield %dot : tensor<16x16xf32>
    }
    %second = scf.for %i = %c0 to %upper step %c1
        iter_args(%acc = %first) -> tensor<16x16xf32> {
      %dot = tt.dot %a, %b, %acc
          : tensor<16x32xf32> * tensor<32x16xf32> -> tensor<16x16xf32>
      scf.yield %dot : tensor<16x16xf32>
    }
    tt.return %second : tensor<16x16xf32>
  }
}
