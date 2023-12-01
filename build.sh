#!/bin/sh

set -xe

clang -Wall -Wextra -o twice ./session_1_twice/twice.c
clang -Wall -Wextra -o or    ./session_1_twice/or.c
