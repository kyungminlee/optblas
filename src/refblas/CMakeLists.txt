add_library(blasf SHARED dgemm.f dtrsm.f lsame.f xerbla.f)
set_target_properties(blasf PROPERTIES LINKER_LANGUAGE "C")
