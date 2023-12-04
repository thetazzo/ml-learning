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
clang -Wall -Wextra -o twice ./session_4/twice.c -lm
