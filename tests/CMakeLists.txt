# build tests
add_executable(TestDerivatives TestDerivatives.cc)
add_executable(TestPseudoInverseMatrix TestPseudoInverseMatrix.cc)
add_executable(TestTiming TestTiming.cc)
add_test(TestDerivatives TestDerivatives)
add_test(TestPseudoInverseMatrix TestPseudoInverseMatrix)
add_test(TestTiming TestTiming)
