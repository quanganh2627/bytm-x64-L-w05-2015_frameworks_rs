
; Shuffle Built-ins

define <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %x, <4 x float> %y, <4 x i32> %mask) nounwind readnone {
entry:
  %and = and <4 x i32> %mask, <i32 7, i32 7, i32 7, i32 7>
  %and.i.i = and <4 x i32> %mask, <i32 3, i32 3, i32 3, i32 3>
  %0 = bitcast <4 x i32> %and.i.i to <16 x i8>
  %1 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %0, <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 4, i8 4, i8 4, i8 4, i8 8, i8 8, i8 8, i8 8, i8 12, i8 12, i8 12, i8 12>) nounwind
  %astype1.i.i = bitcast <16 x i8> %1 to <8 x i16>
  %shl.i.i = shl <8 x i16> %astype1.i.i, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %astype3.i.i = bitcast <8 x i16> %shl.i.i to <16 x i8>
  %2 = tail call <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8> %astype3.i.i, <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>) nounwind
  %3 = bitcast <4 x float> %x to <16 x i8>
  %4 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %3, <16 x i8> %2) nounwind
  %astype1.i = bitcast <16 x i8> %4 to <4 x float>
  %5 = bitcast <4 x float> %y to <16 x i8>
  %6 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %5, <16 x i8> %2) nounwind
  %astype1.i9 = bitcast <16 x i8> %6 to <4 x float>
  %cmp = icmp ult <4 x i32> %and, <i32 4, i32 4, i32 4, i32 4>
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %7 = icmp slt <4 x i32> %sext, zeroinitializer
  %8 = select <4 x i1> %7, <4 x float> %astype1.i, <4 x float> %astype1.i9
  ret <4 x float> %8
}

define <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %x, <4 x i32> %y, <4 x i32> %mask) nounwind readnone {
entry:
  %and = and <4 x i32> %mask, <i32 7, i32 7, i32 7, i32 7>
  %and.i.i = and <4 x i32> %mask, <i32 3, i32 3, i32 3, i32 3>
  %0 = bitcast <4 x i32> %and.i.i to <16 x i8>
  %1 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %0, <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 4, i8 4, i8 4, i8 4, i8 8, i8 8, i8 8, i8 8, i8 12, i8 12, i8 12, i8 12>) nounwind
  %astype1.i.i = bitcast <16 x i8> %1 to <8 x i16>
  %shl.i.i = shl <8 x i16> %astype1.i.i, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %astype3.i.i = bitcast <8 x i16> %shl.i.i to <16 x i8>
  %2 = tail call <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8> %astype3.i.i, <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>) nounwind
  %3 = bitcast <4 x i32> %x to <16 x i8>
  %4 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %3, <16 x i8> %2) nounwind
  %astype7.i.i = bitcast <16 x i8> %4 to <4 x i32>
  %5 = bitcast <4 x i32> %y to <16 x i8>
  %6 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %5, <16 x i8> %2) nounwind
  %astype7.i.i9 = bitcast <16 x i8> %6 to <4 x i32>
  %cmp = icmp ult <4 x i32> %and, <i32 4, i32 4, i32 4, i32 4>
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %7 = icmp slt <4 x i32> %sext, zeroinitializer
  %8 = select <4 x i1> %7, <4 x i32> %astype7.i.i, <4 x i32> %astype7.i.i9
  ret <4 x i32> %8
}

define <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %x, <16 x i8> %y, <16 x i8> %mask) nounwind readnone {
entry:
  %and = and <16 x i8> %mask, <i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31, i8 31>
  %and.i.i = and <16 x i8> %mask, <i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15, i8 15>
  %0 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %x, <16 x i8> %and.i.i) nounwind
  %1 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %y, <16 x i8> %and.i.i) nounwind
  %cmp = icmp ult <16 x i8> %and, <i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16, i8 16>
  %sext = sext <16 x i1> %cmp to <16 x i8>
  %2 = icmp slt <16 x i8> %sext, zeroinitializer
  %3 = select <16 x i1> %2, <16 x i8> %0, <16 x i8> %1
  ret <16 x i8> %3
}

define <8 x i16> @_Z8shuffle2Dv8_sS_Dv8_t(<8 x i16> %x, <8 x i16> %y, <8 x i16> %mask) nounwind readnone {
entry:
  %and = and <8 x i16> %mask, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  %and.i.i = and <8 x i16> %mask, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  %0 = bitcast <8 x i16> %and.i.i to <16 x i8>
  %1 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %0, <16 x i8> <i8 0, i8 0, i8 2, i8 2, i8 4, i8 4, i8 6, i8 6, i8 8, i8 8, i8 10, i8 10, i8 12, i8 12, i8 14, i8 14>) nounwind
  %astype.i.i = bitcast <16 x i8> %1 to <8 x i16>
  %shl.i.i = shl <8 x i16> %astype.i.i, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %astype1.i.i = bitcast <8 x i16> %shl.i.i to <16 x i8>
  %2 = tail call <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8> %astype1.i.i, <16 x i8> <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>) nounwind
  %3 = bitcast <8 x i16> %x to <16 x i8>
  %4 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %3, <16 x i8> %2) nounwind
  %astype5.i.i = bitcast <16 x i8> %4 to <8 x i16>
  %5 = bitcast <8 x i16> %y to <16 x i8>
  %6 = tail call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %5, <16 x i8> %2) nounwind
  %astype5.i.i9 = bitcast <16 x i8> %6 to <8 x i16>
  %cmp = icmp ult <8 x i16> %and, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %sext = sext <8 x i1> %cmp to <8 x i16>
  %7 = icmp slt <8 x i16> %sext, zeroinitializer
  %8 = select <8 x i1> %7, <8 x i16> %astype5.i.i, <8 x i16> %astype5.i.i9
  ret <8 x i16> %8
}

; Optimized vload\vstore built-ins

define void @_Z8vstore16Dv16_cjPc(<16 x i8> %data, i32 %offset, i8* nocapture %p) nounwind {
entry:
  %mul = shl i32 %offset, 4
  %add.ptr = getelementptr i8* %p, i32 %mul
  %data.addr.0.add.ptr.cast = bitcast i8* %add.ptr to <16 x i8>*
  store <16 x i8> %data, <16 x i8>* %data.addr.0.add.ptr.cast, align 1
  ret void
}

define <16 x i8> @_Z7vload16jPKc(i32 %offset, i8* nocapture %p) nounwind readonly {
entry:
  %mul = shl i32 %offset, 4
  %add.ptr = getelementptr i8* %p, i32 %mul
  %res.0.add.ptr.cast = bitcast i8* %add.ptr to <16 x i8>*
  %res.0.copyload = load <16 x i8>* %res.0.add.ptr.cast, align 1
  ret <16 x i8> %res.0.copyload
}

; Begin of transpose built-ins

define void @__ocl_load_transpose_char4x4(<4 x i8>* %pLoadAdd, <4 x i8>* nocapture %xOut, <4 x i8>* nocapture %yOut, <4 x i8>* nocapture %zOut, <4 x i8>* nocapture %wOut) nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds <4 x i8>* %pLoadAdd, i32 0, i32 0
  %call = tail call <16 x i8> @_Z7vload16jPKc(i32 0, i8* %0) nounwind
  %1 = shufflevector <16 x i8> %call, <16 x i8> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  store <4 x i8> %1, <4 x i8>* %xOut, align 4, !tbaa !1
  %2 = bitcast <16 x i8> %call to <2 x i64>
  %3 = tail call <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64> %2, i32 8) nounwind
  %astype.i = bitcast <2 x i64> %3 to <16 x i8>
  %4 = shufflevector <16 x i8> %astype.i, <16 x i8> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  store <4 x i8> %4, <4 x i8>* %yOut, align 4, !tbaa !1
  %5 = tail call <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64> %3, i32 8) nounwind
  %astype2.i = bitcast <2 x i64> %5 to <16 x i8>
  %6 = shufflevector <16 x i8> %astype2.i, <16 x i8> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  store <4 x i8> %6, <4 x i8>* %zOut, align 4, !tbaa !1
  %7 = tail call <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64> %5, i32 8) nounwind
  %astype4.i = bitcast <2 x i64> %7 to <16 x i8>
  %8 = shufflevector <16 x i8> %astype4.i, <16 x i8> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  store <4 x i8> %8, <4 x i8>* %wOut, align 4, !tbaa !1
  ret void
}

define void @__ocl_transpose_store_char4x4(<4 x i8>* %pStoreAdd, <4 x i8> %xIn, <4 x i8> %yIn, <4 x i8> %zIn, <4 x i8> %wIn) nounwind alwaysinline {
entry:
  %0 = shufflevector <4 x i8> %xIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %1 = shufflevector <4 x i8> %yIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = shufflevector <4 x i8> %zIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %3 = shufflevector <4 x i8> %wIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %call.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %0, <16 x i8> %1, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype.i = bitcast <16 x i8> %call.i to <8 x i16>
  %call1.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %2, <16 x i8> %3, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype2.i = bitcast <16 x i8> %call1.i to <8 x i16>
  %call3.i = tail call <8 x i16> @_Z8shuffle2Dv8_sS_Dv8_t(<8 x i16> %astype.i, <8 x i16> %astype2.i, <8 x i16> <i16 0, i16 8, i16 1, i16 9, i16 2, i16 10, i16 3, i16 11>) nounwind readnone
  %astype4.i = bitcast <8 x i16> %call3.i to <16 x i8>
  %4 = getelementptr inbounds <4 x i8>* %pStoreAdd, i32 0, i32 0
  tail call void @_Z8vstore16Dv16_cjPc(<16 x i8> %astype4.i, i32 0, i8* %4) nounwind
  ret void
}

define void @__ocl_transpose_scatter_char4x4(<4 x i8>* nocapture %pStoreAdd0, <4 x i8>* nocapture %pStoreAdd1, <4 x i8>* nocapture %pStoreAdd2, <4 x i8>* nocapture %pStoreAdd3, <4 x i8> %xIn, <4 x i8> %yIn, <4 x i8> %zIn, <4 x i8> %wIn) nounwind alwaysinline {
entry:
  %0 = shufflevector <4 x i8> %xIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %1 = shufflevector <4 x i8> %yIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = shufflevector <4 x i8> %zIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %3 = shufflevector <4 x i8> %wIn, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %call.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %0, <16 x i8> %1, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype.i = bitcast <16 x i8> %call.i to <8 x i16>
  %call1.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %2, <16 x i8> %3, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype2.i = bitcast <16 x i8> %call1.i to <8 x i16>
  %call3.i = tail call <8 x i16> @_Z8shuffle2Dv8_sS_Dv8_t(<8 x i16> %astype.i, <8 x i16> %astype2.i, <8 x i16> <i16 0, i16 8, i16 1, i16 9, i16 2, i16 10, i16 3, i16 11>) nounwind readnone
  %4 = bitcast <8 x i16> %call3.i to <4 x i32>
  %5 = extractelement <4 x i32> %4, i32 0
  %6 = bitcast <4 x i8>* %pStoreAdd0 to i32*
  store i32 %5, i32* %6, align 4, !tbaa !3
  %7 = extractelement <4 x i32> %4, i32 1
  %8 = bitcast <4 x i8>* %pStoreAdd1 to i32*
  store i32 %7, i32* %8, align 4, !tbaa !3
  %9 = extractelement <4 x i32> %4, i32 2
  %10 = bitcast <4 x i8>* %pStoreAdd2 to i32*
  store i32 %9, i32* %10, align 4, !tbaa !3
  %11 = extractelement <4 x i32> %4, i32 3
  %12 = bitcast <4 x i8>* %pStoreAdd3 to i32*
  store i32 %11, i32* %12, align 4, !tbaa !3
  ret void
}

define void @__ocl_load_transpose_char4x8(<4 x i8>* %pLoadAdd, <8 x i8>* nocapture %xOut, <8 x i8>* nocapture %yOut, <8 x i8>* nocapture %zOut, <8 x i8>* nocapture %wOut) nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds <4 x i8>* %pLoadAdd, i32 0, i32 0
  %call.i = tail call <16 x i8> @_Z7vload16jPKc(i32 0, i8* %0) nounwind
  %call1.i = tail call <16 x i8> @_Z7vload16jPKc(i32 1, i8* %0) nounwind
  %1 = shufflevector <16 x i8> %call.i, <16 x i8> undef, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %2 = shufflevector <16 x i8> %call1.i, <16 x i8> undef, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %astype.i.i = bitcast <16 x i8> %1 to <4 x i32>
  %astype1.i.i = bitcast <16 x i8> %2 to <4 x i32>
  %call.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %astype.i.i, <4 x i32> %astype1.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %astype2.i.i = bitcast <4 x i32> %call.i.i to <16 x i8>
  %call5.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %astype.i.i, <4 x i32> %astype1.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %astype6.i.i = bitcast <4 x i32> %call5.i.i to <16 x i8>
  %3 = shufflevector <16 x i8> %astype2.i.i, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i8> %3, <8 x i8>* %xOut, align 8, !tbaa !1
  %4 = shufflevector <16 x i8> %astype2.i.i, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <8 x i8> %4, <8 x i8>* %yOut, align 8, !tbaa !1
  %5 = shufflevector <16 x i8> %astype6.i.i, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <8 x i8> %5, <8 x i8>* %zOut, align 8, !tbaa !1
  %6 = shufflevector <16 x i8> %astype6.i.i, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  store <8 x i8> %6, <8 x i8>* %wOut, align 8, !tbaa !1
  ret void
}

define void @__ocl_transpose_store_char4x8(<4 x i8>* %pStoreAdd, <8 x i8> %xIn, <8 x i8> %yIn, <8 x i8> %zIn, <8 x i8> %wIn) nounwind alwaysinline {
entry:
  %0 = shufflevector <8 x i8> %xIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %1 = shufflevector <8 x i8> %yIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = shufflevector <8 x i8> %zIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %3 = shufflevector <8 x i8> %wIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %call.i.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %0, <16 x i8> %1, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype.i.i = bitcast <16 x i8> %call.i.i to <8 x i16>
  %call1.i.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %2, <16 x i8> %3, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype2.i.i = bitcast <16 x i8> %call1.i.i to <8 x i16>
  %call3.i.i = tail call <8 x i16> @_Z8shuffle2Dv8_sS_Dv8_t(<8 x i16> %astype.i.i, <8 x i16> %astype2.i.i, <8 x i16> <i16 0, i16 8, i16 1, i16 9, i16 2, i16 10, i16 3, i16 11>) nounwind readnone
  %astype4.i.i = bitcast <8 x i16> %call3.i.i to <16 x i8>
  %call5.i.i = tail call <8 x i16> @_Z8shuffle2Dv8_sS_Dv8_t(<8 x i16> %astype.i.i, <8 x i16> %astype2.i.i, <8 x i16> <i16 4, i16 12, i16 5, i16 13, i16 6, i16 14, i16 7, i16 15>) nounwind readnone
  %astype6.i.i = bitcast <8 x i16> %call5.i.i to <16 x i8>
  %4 = getelementptr inbounds <4 x i8>* %pStoreAdd, i32 0, i32 0
  tail call void @_Z8vstore16Dv16_cjPc(<16 x i8> %astype4.i.i, i32 0, i8* %4) nounwind
  tail call void @_Z8vstore16Dv16_cjPc(<16 x i8> %astype6.i.i, i32 1, i8* %4) nounwind
  ret void
}

define void @__ocl_transpose_scatter_char4x8(<4 x i8>* nocapture %pStoreAdd0, <4 x i8>* nocapture %pStoreAdd1, <4 x i8>* nocapture %pStoreAdd2, <4 x i8>* nocapture %pStoreAdd3, <4 x i8>* nocapture %pStoreAdd4, <4 x i8>* nocapture %pStoreAdd5, <4 x i8>* nocapture %pStoreAdd6, <4 x i8>* nocapture %pStoreAdd7, <8 x i8> %xIn, <8 x i8> %yIn, <8 x i8> %zIn, <8 x i8> %wIn) nounwind alwaysinline {
entry:
  %0 = shufflevector <8 x i8> %xIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %1 = shufflevector <8 x i8> %yIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %2 = shufflevector <8 x i8> %zIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %3 = shufflevector <8 x i8> %wIn, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %call.i.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %0, <16 x i8> %1, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype.i.i = bitcast <16 x i8> %call.i.i to <8 x i16>
  %call1.i.i = tail call <16 x i8> @_Z8shuffle2Dv16_cS_Dv16_h(<16 x i8> %2, <16 x i8> %3, <16 x i8> <i8 0, i8 16, i8 1, i8 17, i8 2, i8 18, i8 3, i8 19, i8 4, i8 20, i8 5, i8 21, i8 6, i8 22, i8 7, i8 23>) nounwind readnone
  %astype2.i.i = bitcast <16 x i8> %call1.i.i to <8 x i16>
  %call3.i.i = tail call <8 x i16> @_Z8shuffle2Dv8_sS_Dv8_t(<8 x i16> %astype.i.i, <8 x i16> %astype2.i.i, <8 x i16> <i16 0, i16 8, i16 1, i16 9, i16 2, i16 10, i16 3, i16 11>) nounwind readnone
  %4 = bitcast <8 x i16> %call3.i.i to <4 x i32>
  %call5.i.i = tail call <8 x i16> @_Z8shuffle2Dv8_sS_Dv8_t(<8 x i16> %astype.i.i, <8 x i16> %astype2.i.i, <8 x i16> <i16 4, i16 12, i16 5, i16 13, i16 6, i16 14, i16 7, i16 15>) nounwind readnone
  %5 = bitcast <8 x i16> %call5.i.i to <4 x i32>
  %6 = extractelement <4 x i32> %4, i32 0
  %7 = bitcast <4 x i8>* %pStoreAdd0 to i32*
  store i32 %6, i32* %7, align 4, !tbaa !3
  %8 = extractelement <4 x i32> %4, i32 1
  %9 = bitcast <4 x i8>* %pStoreAdd1 to i32*
  store i32 %8, i32* %9, align 4, !tbaa !3
  %10 = extractelement <4 x i32> %4, i32 2
  %11 = bitcast <4 x i8>* %pStoreAdd2 to i32*
  store i32 %10, i32* %11, align 4, !tbaa !3
  %12 = extractelement <4 x i32> %4, i32 3
  %13 = bitcast <4 x i8>* %pStoreAdd3 to i32*
  store i32 %12, i32* %13, align 4, !tbaa !3
  %14 = extractelement <4 x i32> %5, i32 0
  %15 = bitcast <4 x i8>* %pStoreAdd4 to i32*
  store i32 %14, i32* %15, align 4, !tbaa !3
  %16 = extractelement <4 x i32> %5, i32 1
  %17 = bitcast <4 x i8>* %pStoreAdd5 to i32*
  store i32 %16, i32* %17, align 4, !tbaa !3
  %18 = extractelement <4 x i32> %5, i32 2
  %19 = bitcast <4 x i8>* %pStoreAdd6 to i32*
  store i32 %18, i32* %19, align 4, !tbaa !3
  %20 = extractelement <4 x i32> %5, i32 3
  %21 = bitcast <4 x i8>* %pStoreAdd7 to i32*
  store i32 %20, i32* %21, align 4, !tbaa !3
  ret void
}

define void @__ocl_load_transpose_int4x4(<4 x i32>* nocapture %pLoadAdd, <4 x i32>* nocapture %xOut, <4 x i32>* nocapture %yOut, <4 x i32>* nocapture %zOut, <4 x i32>* nocapture %wOut) nounwind alwaysinline {
entry:
  %0 = load <4 x i32>* %pLoadAdd, align 16, !tbaa !1
  %arrayidx1.i = getelementptr <4 x i32>* %pLoadAdd, i32 1
  %1 = load <4 x i32>* %arrayidx1.i, align 16, !tbaa !1
  %arrayidx2.i = getelementptr <4 x i32>* %pLoadAdd, i32 2
  %2 = load <4 x i32>* %arrayidx2.i, align 16, !tbaa !1
  %arrayidx3.i = getelementptr <4 x i32>* %pLoadAdd, i32 3
  %3 = load <4 x i32>* %arrayidx3.i, align 16, !tbaa !1
  %call.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %0, <4 x i32> %2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %0, <4 x i32> %2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %1, <4 x i32> %3, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %1, <4 x i32> %3, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call4.i.i, <4 x i32>* %xOut, align 16, !tbaa !1
  %call5.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call5.i.i, <4 x i32>* %yOut, align 16, !tbaa !1
  %call6.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call6.i.i, <4 x i32>* %zOut, align 16, !tbaa !1
  %call7.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call7.i.i, <4 x i32>* %wOut, align 16, !tbaa !1
  ret void
}

define void @__ocl_transpose_store_int4x4(<4 x i32>* nocapture %pStoreAdd, <4 x i32> %xIn, <4 x i32> %yIn, <4 x i32> %zIn, <4 x i32> %wIn) nounwind alwaysinline {
entry:
  %arrayidx1.i = getelementptr <4 x i32>* %pStoreAdd, i32 1
  %arrayidx2.i = getelementptr <4 x i32>* %pStoreAdd, i32 2
  %arrayidx3.i = getelementptr <4 x i32>* %pStoreAdd, i32 3
  %call.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %xIn, <4 x i32> %zIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %xIn, <4 x i32> %zIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %yIn, <4 x i32> %wIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %yIn, <4 x i32> %wIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call4.i.i, <4 x i32>* %pStoreAdd, align 16, !tbaa !1
  %call5.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call5.i.i, <4 x i32>* %arrayidx1.i, align 16, !tbaa !1
  %call6.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call6.i.i, <4 x i32>* %arrayidx2.i, align 16, !tbaa !1
  %call7.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call7.i.i, <4 x i32>* %arrayidx3.i, align 16, !tbaa !1
  ret void
}

define void @__ocl_gather_transpose_int4x4(<4 x i32>* nocapture %pLoadAdd0, <4 x i32>* nocapture %pLoadAdd1, <4 x i32>* nocapture %pLoadAdd2, <4 x i32>* nocapture %pLoadAdd3, <4 x i32>* nocapture %xOut, <4 x i32>* nocapture %yOut, <4 x i32>* nocapture %zOut, <4 x i32>* nocapture %wOut) nounwind alwaysinline {
entry:
  %0 = load <4 x i32>* %pLoadAdd0, align 16, !tbaa !1
  %1 = load <4 x i32>* %pLoadAdd1, align 16, !tbaa !1
  %2 = load <4 x i32>* %pLoadAdd2, align 16, !tbaa !1
  %3 = load <4 x i32>* %pLoadAdd3, align 16, !tbaa !1
  %call.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %0, <4 x i32> %2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %0, <4 x i32> %2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %1, <4 x i32> %3, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %1, <4 x i32> %3, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call4.i.i, <4 x i32>* %xOut, align 16, !tbaa !1
  %call5.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call5.i.i, <4 x i32>* %yOut, align 16, !tbaa !1
  %call6.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call6.i.i, <4 x i32>* %zOut, align 16, !tbaa !1
  %call7.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call7.i.i, <4 x i32>* %wOut, align 16, !tbaa !1
  ret void
}

define void @__ocl_transpose_scatter_int4x4(<4 x i32>* nocapture %pStoreAdd0, <4 x i32>* nocapture %pStoreAdd1, <4 x i32>* nocapture %pStoreAdd2, <4 x i32>* nocapture %pStoreAdd3, <4 x i32> %xIn, <4 x i32> %yIn, <4 x i32> %zIn, <4 x i32> %wIn) nounwind alwaysinline {
entry:
  %call.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %xIn, <4 x i32> %zIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %xIn, <4 x i32> %zIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %yIn, <4 x i32> %wIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %yIn, <4 x i32> %wIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call4.i.i, <4 x i32>* %pStoreAdd0, align 16, !tbaa !1
  %call5.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call.i.i, <4 x i32> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call5.i.i, <4 x i32>* %pStoreAdd1, align 16, !tbaa !1
  %call6.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x i32> %call6.i.i, <4 x i32>* %pStoreAdd2, align 16, !tbaa !1
  %call7.i.i = tail call <4 x i32> @_Z8shuffle2Dv4_iS_Dv4_j(<4 x i32> %call1.i.i, <4 x i32> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x i32> %call7.i.i, <4 x i32>* %pStoreAdd3, align 16, !tbaa !1
  ret void
}

define void @__ocl_load_transpose_float4x4(<4 x float>* nocapture %pLoadAdd, <4 x float>* nocapture %xOut, <4 x float>* nocapture %yOut, <4 x float>* nocapture %zOut, <4 x float>* nocapture %wOut) nounwind alwaysinline {
entry:
  %0 = load <4 x float>* %pLoadAdd, align 16, !tbaa !1
  %arrayidx1.i = getelementptr <4 x float>* %pLoadAdd, i32 1
  %1 = load <4 x float>* %arrayidx1.i, align 16, !tbaa !1
  %arrayidx2.i = getelementptr <4 x float>* %pLoadAdd, i32 2
  %2 = load <4 x float>* %arrayidx2.i, align 16, !tbaa !1
  %arrayidx3.i = getelementptr <4 x float>* %pLoadAdd, i32 3
  %3 = load <4 x float>* %arrayidx3.i, align 16, !tbaa !1
  %call.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %0, <4 x float> %2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %0, <4 x float> %2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %1, <4 x float> %3, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %1, <4 x float> %3, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call4.i.i, <4 x float>* %xOut, align 16, !tbaa !1
  %call5.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call5.i.i, <4 x float>* %yOut, align 16, !tbaa !1
  %call6.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call6.i.i, <4 x float>* %zOut, align 16, !tbaa !1
  %call7.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call7.i.i, <4 x float>* %wOut, align 16, !tbaa !1
  ret void
}

define void @__ocl_transpose_store_float4x4(<4 x float>* nocapture %pStoreAdd, <4 x float> %xIn, <4 x float> %yIn, <4 x float> %zIn, <4 x float> %wIn) nounwind alwaysinline {
entry:
  %arrayidx1.i = getelementptr <4 x float>* %pStoreAdd, i32 1
  %arrayidx2.i = getelementptr <4 x float>* %pStoreAdd, i32 2
  %arrayidx3.i = getelementptr <4 x float>* %pStoreAdd, i32 3
  %call.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %xIn, <4 x float> %zIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %xIn, <4 x float> %zIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %yIn, <4 x float> %wIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %yIn, <4 x float> %wIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call4.i.i, <4 x float>* %pStoreAdd, align 16, !tbaa !1
  %call5.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call5.i.i, <4 x float>* %arrayidx1.i, align 16, !tbaa !1
  %call6.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call6.i.i, <4 x float>* %arrayidx2.i, align 16, !tbaa !1
  %call7.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call7.i.i, <4 x float>* %arrayidx3.i, align 16, !tbaa !1
  ret void
}

define void @__ocl_gather_transpose_float4x4(<4 x float>* nocapture %pLoadAdd0, <4 x float>* nocapture %pLoadAdd1, <4 x float>* nocapture %pLoadAdd2, <4 x float>* nocapture %pLoadAdd3, <4 x float>* nocapture %xOut, <4 x float>* nocapture %yOut, <4 x float>* nocapture %zOut, <4 x float>* nocapture %wOut) nounwind alwaysinline {
entry:
  %0 = load <4 x float>* %pLoadAdd0, align 16, !tbaa !1
  %1 = load <4 x float>* %pLoadAdd1, align 16, !tbaa !1
  %2 = load <4 x float>* %pLoadAdd2, align 16, !tbaa !1
  %3 = load <4 x float>* %pLoadAdd3, align 16, !tbaa !1
  %call.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %0, <4 x float> %2, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %0, <4 x float> %2, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %1, <4 x float> %3, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %1, <4 x float> %3, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call4.i.i, <4 x float>* %xOut, align 16, !tbaa !1
  %call5.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call5.i.i, <4 x float>* %yOut, align 16, !tbaa !1
  %call6.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call6.i.i, <4 x float>* %zOut, align 16, !tbaa !1
  %call7.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call7.i.i, <4 x float>* %wOut, align 16, !tbaa !1
  ret void
}

define void @__ocl_transpose_scatter_float4x4(<4 x float>* nocapture %pStoreAdd0, <4 x float>* nocapture %pStoreAdd1, <4 x float>* nocapture %pStoreAdd2, <4 x float>* nocapture %pStoreAdd3, <4 x float> %xIn, <4 x float> %yIn, <4 x float> %zIn, <4 x float> %wIn) nounwind alwaysinline {
entry:
  %call.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %xIn, <4 x float> %zIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call1.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %xIn, <4 x float> %zIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call2.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %yIn, <4 x float> %wIn, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  %call3.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %yIn, <4 x float> %wIn, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  %call4.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call4.i.i, <4 x float>* %pStoreAdd0, align 16, !tbaa !1
  %call5.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call.i.i, <4 x float> %call2.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call5.i.i, <4 x float>* %pStoreAdd1, align 16, !tbaa !1
  %call6.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 0, i32 4, i32 1, i32 5>) nounwind readnone
  store <4 x float> %call6.i.i, <4 x float>* %pStoreAdd2, align 16, !tbaa !1
  %call7.i.i = tail call <4 x float> @_Z8shuffle2Dv4_fS_Dv4_j(<4 x float> %call1.i.i, <4 x float> %call3.i.i, <4 x i32> <i32 2, i32 6, i32 3, i32 7>) nounwind readnone
  store <4 x float> %call7.i.i, <4 x float>* %pStoreAdd3, align 16, !tbaa !1
  ret void
}

; End of transposes built-ins

define i1 @__ocl_allOne(i1 %pred) {
entry:
  ret i1 %pred
}

define i1 @__ocl_allOne_v2(<2 x i1> %pred) {
entry:
  %elem0 = extractelement <2 x i1> %pred, i32 0
  %elem1 = extractelement <2 x i1> %pred, i32 1
  %res = and i1 %elem0, %elem1
  ret i1 %res
}

define i1 @__ocl_allOne_v4(<4 x i1> %pred) {
entry:
  %elem0 = extractelement <4 x i1> %pred, i32 0
  %elem1 = extractelement <4 x i1> %pred, i32 1
  %elem2 = extractelement <4 x i1> %pred, i32 2
  %elem3 = extractelement <4 x i1> %pred, i32 3

  %res1 = and i1 %elem0, %elem1
  %res2 = and i1 %elem2, %elem3

  %res = and i1 %res1, %res2
  ret i1 %res
}

define i1 @__ocl_allOne_v8(<8 x i1> %pred) {
entry:
  %elem0 = extractelement <8 x i1> %pred, i32 0
  %elem1 = extractelement <8 x i1> %pred, i32 1
  %elem2 = extractelement <8 x i1> %pred, i32 2
  %elem3 = extractelement <8 x i1> %pred, i32 3
  %elem4 = extractelement <8 x i1> %pred, i32 4
  %elem5 = extractelement <8 x i1> %pred, i32 5
  %elem6 = extractelement <8 x i1> %pred, i32 6
  %elem7 = extractelement <8 x i1> %pred, i32 7

  %res1 = and i1 %elem0, %elem1
  %res2 = and i1 %elem2, %elem3
  %res3 = and i1 %elem4, %elem5
  %res4 = and i1 %elem6, %elem7

  %res5 = and i1 %res1, %res2
  %res6 = and i1 %res3, %res4

  %res = and i1 %res5, %res6
  ret i1 %res
}

define i1 @__ocl_allOne_v16(<16 x i1> %pred) {
entry:
  %elem0 = extractelement <16 x i1> %pred, i32 0
  %elem1 = extractelement <16 x i1> %pred, i32 1
  %elem2 = extractelement <16 x i1> %pred, i32 2
  %elem3 = extractelement <16 x i1> %pred, i32 3
  %elem4 = extractelement <16 x i1> %pred, i32 4
  %elem5 = extractelement <16 x i1> %pred, i32 5
  %elem6 = extractelement <16 x i1> %pred, i32 6
  %elem7 = extractelement <16 x i1> %pred, i32 7
  %elem8 = extractelement <16 x i1> %pred, i32 8
  %elem9 = extractelement <16 x i1> %pred, i32 9
  %elem10 = extractelement <16 x i1> %pred, i32 10
  %elem11 = extractelement <16 x i1> %pred, i32 11
  %elem12 = extractelement <16 x i1> %pred, i32 12
  %elem13 = extractelement <16 x i1> %pred, i32 13
  %elem14 = extractelement <16 x i1> %pred, i32 14
  %elem15 = extractelement <16 x i1> %pred, i32 15

  %res1 = and i1 %elem0, %elem1
  %res2 = and i1 %elem2, %elem3
  %res3 = and i1 %elem4, %elem5
  %res4 = and i1 %elem6, %elem7
  %res5 = and i1 %elem8, %elem9
  %res6 = and i1 %elem10, %elem11
  %res7 = and i1 %elem12, %elem13
  %res8 = and i1 %elem14, %elem15

  %res9 = and i1 %res1, %res2
  %res10 = and i1 %res3, %res4
  %res11 = and i1 %res5, %res6
  %res12 = and i1 %res7, %res8

  %res13 = and i1 %res9, %res10
  %res14 = and i1 %res11, %res12

  %res = and i1 %res13, %res14
  ret i1 %res
}



define i1 @__ocl_allZero(i1 %t) {
entry:
  %pred = xor i1 %t, true
  ret i1 %pred
}

define i1 @__ocl_allZero_v2(<2 x i1> %t) {
entry:
  %pred = xor <2 x i1> %t, <i1 true, i1 true>
  %elem0 = extractelement <2 x i1> %pred, i32 0
  %elem1 = extractelement <2 x i1> %pred, i32 1
  %res = and i1 %elem0, %elem1
  ret i1 %res
}

define i1 @__ocl_allZero_v4(<4 x i1> %t) {
entry:
  %pred = xor <4 x i1> %t, <i1 true, i1 true, i1 true, i1 true>
  %elem0 = extractelement <4 x i1> %pred, i32 0
  %elem1 = extractelement <4 x i1> %pred, i32 1
  %elem2 = extractelement <4 x i1> %pred, i32 2
  %elem3 = extractelement <4 x i1> %pred, i32 3

  %res1 = and i1 %elem0, %elem1
  %res2 = and i1 %elem2, %elem3

  %res = and i1 %res1, %res2
  ret i1 %res
}

define i1 @__ocl_allZero_v8(<8 x i1> %t) {
entry:
  %pred = xor <8 x i1> %t, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>
  %elem0 = extractelement <8 x i1> %pred, i32 0
  %elem1 = extractelement <8 x i1> %pred, i32 1
  %elem2 = extractelement <8 x i1> %pred, i32 2
  %elem3 = extractelement <8 x i1> %pred, i32 3
  %elem4 = extractelement <8 x i1> %pred, i32 4
  %elem5 = extractelement <8 x i1> %pred, i32 5
  %elem6 = extractelement <8 x i1> %pred, i32 6
  %elem7 = extractelement <8 x i1> %pred, i32 7

  %res1 = and i1 %elem0, %elem1
  %res2 = and i1 %elem2, %elem3
  %res3 = and i1 %elem4, %elem5
  %res4 = and i1 %elem6, %elem7

  %res5 = and i1 %res1, %res2
  %res6 = and i1 %res3, %res4

  %res = and i1 %res5, %res6
  ret i1 %res
}

define i1 @__ocl_allZero_v16(<16 x i1> %t) {
entry:
  %pred = xor <16 x i1> %t, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>
  %elem0 = extractelement <16 x i1> %pred, i32 0
  %elem1 = extractelement <16 x i1> %pred, i32 1
  %elem2 = extractelement <16 x i1> %pred, i32 2
  %elem3 = extractelement <16 x i1> %pred, i32 3
  %elem4 = extractelement <16 x i1> %pred, i32 4
  %elem5 = extractelement <16 x i1> %pred, i32 5
  %elem6 = extractelement <16 x i1> %pred, i32 6
  %elem7 = extractelement <16 x i1> %pred, i32 7
  %elem8 = extractelement <16 x i1> %pred, i32 8
  %elem9 = extractelement <16 x i1> %pred, i32 9
  %elem10 = extractelement <16 x i1> %pred, i32 10
  %elem11 = extractelement <16 x i1> %pred, i32 11
  %elem12 = extractelement <16 x i1> %pred, i32 12
  %elem13 = extractelement <16 x i1> %pred, i32 13
  %elem14 = extractelement <16 x i1> %pred, i32 14
  %elem15 = extractelement <16 x i1> %pred, i32 15

  %res1 = and i1 %elem0, %elem1
  %res2 = and i1 %elem2, %elem3
  %res3 = and i1 %elem4, %elem5
  %res4 = and i1 %elem6, %elem7
  %res5 = and i1 %elem8, %elem9
  %res6 = and i1 %elem10, %elem11
  %res7 = and i1 %elem12, %elem13
  %res8 = and i1 %elem14, %elem15

  %res9 = and i1 %res1, %res2
  %res10 = and i1 %res3, %res4
  %res11 = and i1 %res5, %res6
  %res12 = and i1 %res7, %res8

  %res13 = and i1 %res9, %res10
  %res14 = and i1 %res11, %res12

  %res = and i1 %res13, %res14
  ret i1 %res
}

declare <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64>, i32) nounwind readnone

declare void @llvm.x86.sse.sfence() nounwind

declare void @llvm.x86.sse.storeu.ps(i8*, <4 x float>) nounwind

declare <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32>, i32) nounwind readnone

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

declare <2 x double> @llvm.x86.sse2.cmp.pd(<2 x double>, <2 x double>, i8) nounwind readnone

declare <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64>, i32) nounwind readnone

declare <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32>, i32) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.psubus.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.psubus.b(<16 x i8>, <16 x i8>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.psubs.b(<16 x i8>, <16 x i8>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.sqrt.pd(<2 x double>) nounwind readnone

declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>) nounwind readnone

declare void @llvm.x86.sse2.lfence() nounwind

declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone

declare <2 x i64> @llvm.x86.sse2.pmulu.dq(<4 x i32>, <4 x i32>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.pmulhu.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.pmulh.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.min.pd(<2 x double>, <2 x double>) nounwind readnone

declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone

declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>) nounwind readnone

declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone

declare <8 x i16> @llvm.x86.sse41.pminuw(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.pmins.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.pminu.b(<16 x i8>, <16 x i8>) nounwind readnone

declare <16 x i8> @llvm.x86.sse41.pminsb(<16 x i8>, <16 x i8>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.max.pd(<2 x double>, <2 x double>) nounwind readnone

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone

declare <4 x i32> @llvm.x86.sse41.pmaxud(<4 x i32>, <4 x i32>) nounwind readnone

declare <4 x i32> @llvm.x86.sse41.pmaxsd(<4 x i32>, <4 x i32>) nounwind readnone

declare <8 x i16> @llvm.x86.sse41.pmaxuw(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.pmaxs.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.pmaxu.b(<16 x i8>, <16 x i8>) nounwind readnone

declare <16 x i8> @llvm.x86.sse41.pmaxsb(<16 x i8>, <16 x i8>) nounwind readnone

declare <2 x i64> @llvm.x86.sse41.pmuldq(<4 x i32>, <4 x i32>) nounwind readnone

declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>) nounwind readnone

declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone

declare i32 @llvm.x86.sse2.movmsk.pd(<2 x double>) nounwind readnone

declare i32 @llvm.x86.sse.movmsk.ps(<4 x float>) nounwind readnone

declare <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double>, <2 x double>) nounwind readnone

declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.packuswb.128(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse41.packusdw(<4 x i32>, <4 x i32>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.packsswb.128(<8 x i16>, <8 x i16>) nounwind readnone

declare <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double>) nounwind readnone

declare <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32>, <4 x i32>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.cvtps2pd(<4 x float>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.cvtdq2pd(<4 x i32>) nounwind readnone

declare <4 x float> @llvm.x86.sse2.cvtpd2ps(<2 x double>) nounwind readnone

declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone

declare void @llvm.x86.sse2.mfence() nounwind

declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.paddus.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16>, <8 x i16>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.paddus.b(<16 x i8>, <16 x i8>) nounwind readnone

declare <16 x i8> @llvm.x86.sse2.padds.b(<16 x i8>, <16 x i8>) nounwind readnone

declare <4 x i32> @llvm.x86.ssse3.pabs.d.128(<4 x i32>) nounwind readnone

declare <8 x i16> @llvm.x86.ssse3.pabs.w.128(<8 x i16>) nounwind readnone

declare <16 x i8> @llvm.x86.ssse3.pabs.b.128(<16 x i8>) nounwind readnone

declare { i8, i1 } @llvm.uadd.with.overflow.i8(i8, i8) nounwind readnone

declare { i16, i1 } @llvm.uadd.with.overflow.i16(i16, i16) nounwind readnone

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone


!opencl.build.options = !{!0}

!0 = metadata !{metadata !"-cl-std=CL1.2"}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"int", metadata !1}
