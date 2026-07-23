// RUN: triton-opt --normalize-reduce %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: tt.func @f16_sum
  // CHECK: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64xf16> to tensor<64xf32>
  // CHECK: %[[REDUCE:.*]] = "tt.reduce"(%[[EXT]]) <{axis = 0 : i32}> ({
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
  // CHECK: arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK: tt.reduce.return %{{.*}} : f32
  // CHECK: %[[TRUNC:.*]] = arith.truncf %[[REDUCE]] : f32 to f16
  // CHECK: tt.return %[[TRUNC]] : f16
  tt.func @f16_sum(%input: tensor<64xf16>) -> f16 {
    %result = "tt.reduce"(%input) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f16, %rhs: f16):
      %sum = arith.addf %lhs, %rhs : f16
      tt.reduce.return %sum : f16
    }) : (tensor<64xf16>) -> f16
    tt.return %result : f16
  }

  // CHECK-LABEL: tt.func @f32_sum
  // CHECK-NOT: arith.extf
  // CHECK: "tt.reduce"(%{{.*}})
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
  // CHECK: arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NOT: arith.truncf
  tt.func @f32_sum(%input: tensor<64xf32>) -> f32 {
    %result = "tt.reduce"(%input) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<64xf32>) -> f32
    tt.return %result : f32
  }
}
