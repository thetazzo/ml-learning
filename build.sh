#!/bin/sh

set -xe

clang -Wall -Wextra -o twice ./session_1/twice.c
clang -Wall -Wextra -o and_or ./session_1/and_or.c -lm
