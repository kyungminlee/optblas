#include <stdio.h>
__attribute__((weak))
void xerbla_(char const * srname, int const *info, int len) {
  printf("Error in %s: %d\n", srname, *info);
}
