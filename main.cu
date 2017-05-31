#include <cstdio>
#include <cassert>
#include "board.hpp"
#include "solver.hpp"

struct Base81 {
  ull table_p[256];
  ull table_o[256];
  Base81() {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            int index = i + 3*j + 9*k + 32*l + 33;
            table_p[index] = (i==1) + 2*(j==1) + 4*(k==1) + 8*(l==1);
            table_o[index] = (i/2) + 2*(j/2) + 4*(k/2) + 8*(l/2);
          }
        }
      }
    }
  }
  __host__ void output(const Board &bd) {
    for (int i = 0; i < 64; ++i) {
      if ((player(bd) >> i) & 1) {
        fputc('x', stderr);
      } else if ((opponent(bd) >> i) & 1) {
        fputc('o', stderr);
      } else {
        fputc('.', stderr);
      }
      if ((i % 8) == 7) fputc('\n', stderr);
    }
    fputc('\n', stderr);
  }
  __host__ Board input_bd(FILE *fp, int l) {
    char buf[18];
    if (fgets(buf, 18, fp) == NULL) {
      fprintf(stderr, "unexpected EOF\n");
      exit(-1);
    }
    ull p = 0, o = 0;
    if (strlen(buf) < 16) {
      fprintf(stderr, "too short input: line %d\n", l+1);
      exit(-1);
    }
    for (int i = 0; i < 16; ++i) {
      p |= table_p[buf[i]] << (i*4);
      o |= table_o[buf[i]] << (i*4);
    }
    assert(!(p & o));
    return make_board(p, o);
  }
};

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "%s [INPUT] [OUTPUT]\n", argv[0]);
    return EXIT_FAILURE;
  }
  FILE *fp = fopen(argv[1], "r");
  if (fp == nullptr) {
    fprintf(stderr, "no such file: %s\n", argv[1]);
    return EXIT_FAILURE;
  }
  FILE *ofp = fopen(argv[2], "w");
  if (fp == nullptr) {
    fprintf(stderr, "cannot open file: %s\n", argv[2]);
    return EXIT_FAILURE;
  }
  int n;
  fscanf(fp, "%d ", &n);
  fprintf(stderr, "start read, data size is %d\n", n);
  Base81 b81;
  Board * const board = (Board*)malloc(sizeof(Board) * n);
  for (int i = 0; i < n; ++i) {
    board[i] = b81.input_bd(fp, i);
  }
  int * const result = (int*)malloc(sizeof(int) * n);
  int * const nodes_count = (int*)malloc(sizeof(int) * n);
  fputs("start solve\n", stderr);
  solve(board, result, nodes_count, n);
  fputs("end solve\n", stderr);
  ull nodes_total = 0;
  for (int i = 0; i < n; ++i) {
    fprintf(ofp, "%d\n", result[i]);
    nodes_total += nodes_count[i];
  }
  fprintf(stderr, "%lu\n", nodes_total);
  free(board);
  free(result);
  free(nodes_count);
  return 0;
}
