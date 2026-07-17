// RUN: triton-opt --start-slice %s | FileCheck %s

module {
  // CHECK-LABEL: tt.func public @preserve_store_attributes
  // CHECK: tt.store
  // CHECK-SAME: preserve_me
  // CHECK-SAME: test.value = 7 : i32
  // CHECK-SAME: tiled_op
  tt.func public @preserve_store_attributes(
      %src: tensor<64x!tt.ptr<f32>>,
      %dst: tensor<64x!tt.ptr<f32>>) {
    %value = tt.load %src : tensor<64x!tt.ptr<f32>>
    %sum = arith.addf %value, %value : tensor<64xf32>
    tt.store %dst, %sum {preserve_me, test.value = 7 : i32}
        : tensor<64x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: tt.func public @convert_final_maxnumf
  // CHECK: "tt.reduce"
  // CHECK: arith.maxnumf
  // CHECK: "tt.reduce"
  // CHECK: arith.maximumf
  tt.func public @convert_final_maxnumf(
      %src: tensor<64x!tt.ptr<f32>>,
      %dst: !tt.ptr<f32>,
      %vector_dst: tensor<64x!tt.ptr<f32>>) {
    %value = tt.load %src : tensor<64x!tt.ptr<f32>>
    %sum = arith.addf %value, %value : tensor<64xf32>
    tt.store %vector_dst, %sum : tensor<64x!tt.ptr<f32>>
    %max = "tt.reduce"(%sum) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %result = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %result : f32
    }) : (tensor<64xf32>) -> f32
    tt.store %dst, %max : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: tt.func public @convert_final_minnumf
  // CHECK: "tt.reduce"
  // CHECK: arith.minnumf
  // CHECK: "tt.reduce"
  // CHECK: arith.minimumf
  tt.func public @convert_final_minnumf(
      %src: tensor<64x!tt.ptr<f32>>,
      %dst: !tt.ptr<f32>,
      %vector_dst: tensor<64x!tt.ptr<f32>>) {
    %value = tt.load %src : tensor<64x!tt.ptr<f32>>
    %sum = arith.addf %value, %value : tensor<64xf32>
    tt.store %vector_dst, %sum : tensor<64x!tt.ptr<f32>>
    %min = "tt.reduce"(%sum) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %result = arith.minnumf %lhs, %rhs : f32
      tt.reduce.return %result : f32
    }) : (tensor<64xf32>) -> f32
    tt.store %dst, %min : !tt.ptr<f32>
    tt.return
  }
}
