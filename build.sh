#!/bin/sh

set -xe

clang -Wall -Wextra -o twice ./session_1_twice/twice.c
clang -Wall -Wextra -o and_or ./session_1_twice/and_or.c -lm
