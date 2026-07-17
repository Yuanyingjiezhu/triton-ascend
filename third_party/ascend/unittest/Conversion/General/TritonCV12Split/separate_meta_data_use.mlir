// RUN: triton-opt --triton-separate-meta-data-use %s | FileCheck %s
// RUN: triton-opt --pass-pipeline='builtin.module(triton-separate-meta-data-use,triton-remove-use-tags)' %s \
// RUN:   | FileCheck %s --check-prefix=CLEAN \
// RUN:     --implicit-check-not=MetaUse --implicit-check-not=DataUse \
// RUN:     --implicit-check-not=MixUse --implicit-check-not=Undefined

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @separate_chain(
      %src: !tt.ptr<i32>,
      %dst: !tt.ptr<i32>) {
    %seed = arith.constant dense<1> : tensor<8xi32>
    %first = arith.addi %seed, %seed : tensor<8xi32>
    %second = arith.addi %first, %seed : tensor<8xi32>

    %srcs = tt.splat %src : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %read_ptrs = tt.addptr %srcs, %second
        : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %loaded = tt.load %read_ptrs : tensor<8x!tt.ptr<i32>>

    %dsts = tt.splat %dst : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %write_ptrs = tt.addptr %dsts, %seed
        : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %write_ptrs, %second : tensor<8x!tt.ptr<i32>>
    tt.store %write_ptrs, %loaded : tensor<8x!tt.ptr<i32>>
    tt.return
  }

  // %mixed must not be split while one of its direct users remains MixUse.
  tt.func public @blocked_by_mixed_user(%ptr: !tt.ptr<i32>) {
    %seed = arith.constant dense<1> : tensor<8xi32>
    %mixed = arith.addi %seed, %seed : tensor<8xi32>
    %cond = arith.constant true
    %selected = arith.select %cond, %mixed, %mixed : tensor<8xi32>
    %ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %mixed_ptrs = tt.addptr %ptrs, %mixed
        : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %selected_ptrs = tt.addptr %ptrs, %selected
        : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %mixed_ptrs, %selected : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @separate_chain
// CHECK: %[[FIRST_META:.*]] = arith.addi {{.*}} {MetaUse}
// CHECK: %[[FIRST_DATA:.*]] = arith.addi {{.*}} {DataUse}
// CHECK: %[[SECOND_META:.*]] = arith.addi %[[FIRST_META]], {{.*}} {MetaUse}
// CHECK: %[[SECOND_DATA:.*]] = arith.addi %[[FIRST_DATA]], {{.*}} {DataUse}
// CHECK: tt.addptr {{.*}}, %[[SECOND_META]]
// CHECK: tt.load {{.*}} {DataUse}
// CHECK: tt.store {{.*}}, %[[SECOND_DATA]]

// Every tensor-producing operation is annotated after separation, including
// operations that were not cloned.

// CHECK-LABEL: tt.func public @blocked_by_mixed_user
// CHECK: arith.addi {{.*}} {MixUse}
// CHECK: arith.select {{.*}} {MixUse}

// CLEAN-LABEL: tt.func public @separate_chain
