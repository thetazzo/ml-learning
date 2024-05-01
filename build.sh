#!/bin/sh

set -xe

## Session 1
# clang -Wall -Wextra -o twice ./session_1/twice.c
# clang -Wall -Wextra -o and_or ./session_1/and_or.c -lm

## Session 2
# clang -Wall -Wextra -o xor ./session_2/xor.c -lm

## Session 3
# clang -Wall -Wextra -o main ./session_3/main.c -lm
# clang -Wall -Wextra -o xor ./session_3/xor.c -lm

## Session 4
# clang -Wall -Wextra -o twice ./session_4/twice.c -lm
# clang -Wall -Wextra -o and_or ./session_4/and_or.c -lm
# clang -Wall -Wextra -o xor ./session_4/xor.c -lm

## Session 5
# clang -Wall -Wextra -o adder ./session_5/adder.c -lm

# Session 6
# clang -Wall -Wextra -o xor ./session_6/xor.c -lm

# Session 7
# clang -O3 -Wall -Wextra -o dump_nn ./session_7/dump_nn.c -lm
# clang -O3 -Wall -Wextra -o xor ./session_7/xor.c -lm
# clang -Wall -Wextra -o adder ./session_8/adder.c -lm

# Session 8
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/
RFLAGS=`pkg-config --cflags raylib`
RLIBS="`pkg-config --libs raylib` -ldl -lpthread"

CFLAGS="-O3 -Wall -Wextra"
LIBS="-lm"

raylib_clang() {
    clang $CFLAGS $RFLAGS -o $2 $1.c $RLIBS $LIBS;
}
# raylib_clang ./session_8/adder adder
# raylib_clang ./session_8/nnf nnf
# clang $CFLAGS -o adder.gen ./session_8/adder.gen.c $LIBS
# clang $CFLAGS -o xor.gen ./session_8/xor.gen.c $LIBS

# Session 9
# raylib_clang ./session_9/nnf nnf
# raylib_clang ./session_9/img2nn img2nn

# Session 10
# raylib_clang ./session_10/img2nn img2nn

# Session 11
# raylib_clang ./session_11/img2nn img2nn

# Session 12
# raylib_clang ./session_12/img2nn img2nn

# Session 13
# raylib_clang ./session_13/img2nn img2nn

# Session 14
# raylib_clang ./session_14/img2nn img2nn
raylib_clang ./session_14/layout layout
