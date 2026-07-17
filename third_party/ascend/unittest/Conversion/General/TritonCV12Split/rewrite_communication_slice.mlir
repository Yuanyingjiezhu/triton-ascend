// RUN: triton-opt --rewrite-communication-slice %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @extract_slice_in_if(
  // CHECK: %[[WORKSPACE:.*]] = memref_ext.alloc_workspace
  // CHECK: bufferization.materialize_in_destination %[[SOURCE:.*]] in writable %[[WORKSPACE]]
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[WORKSPACE]]
  // CHECK: %[[LOCAL:.*]] = memref.alloc()
  // CHECK: memref.copy %[[SUBVIEW]], %[[LOCAL]]
  // CHECK: %[[SLICE:.*]] = bufferization.to_tensor %[[LOCAL]] restrict writable
  // CHECK-NEXT: %[[RESULT:.*]] = scf.if %[[COND:.*]]
  // CHECK: scf.yield %[[SLICE]]
  func.func @extract_slice_in_if(
      %cond: i1, %source: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %result = scf.if %cond -> (tensor<32x128xf32>) {
      %slice = tensor.extract_slice %source[0, 0] [32, 128] [1, 1]
          {cv_communication_slice}
          : tensor<64x128xf32> to tensor<32x128xf32>
      scf.yield %slice : tensor<32x128xf32>
    } else {
      %slice = tensor.extract_slice %source[32, 0] [32, 128] [1, 1]
          : tensor<64x128xf32> to tensor<32x128xf32>
      scf.yield %slice : tensor<32x128xf32>
    }
    return %result : tensor<32x128xf32>
  }
}
