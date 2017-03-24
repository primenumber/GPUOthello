#include <cstdio>
#include <cassert>

constexpr int threadsPerBlock = 128;
constexpr int simdWidth = 4;
constexpr int nodesPerBlock = threadsPerBlock/simdWidth;
constexpr int MAX_DEPTH = 10;

using ull = unsigned long long;

struct Board {
  ull x, y;
  __host__ __device__ Board() = default;
  __host__ __device__ Board(const ull &x, const ull &y)
    : x(x), y(y) {}
};

Board make_board(const ull &player, const ull &opponent) {
  return Board(~opponent, ~player);
}

__host__ __device__ ull player(const Board &bd) {
  return bd.x & ~bd.y;
}

__host__ __device__ ull opponent(const Board &bd) {
  return bd.y & ~bd.x;
}

__host__ __device__ ull puttable(const Board &bd) {
  return bd.x & bd.y;
}

__host__ __device__ ull stones(const Board &bd) {
  return bd.x ^ bd.y;
}

// puttable(bd) must be 0
__host__ __device__ Board move(const Board &bd, const ull &flipped, const ull &pos_bit) {
  ull old_stones = bd.x | bd.y;
  return Board((bd.y ^ flipped) | ~(old_stones | pos_bit), (bd.x ^ flipped) | ~old_stones);
}

// puttable(bd) must be 0
__host__ __device__ Board move_pass(const Board &bd) {
  return Board(~bd.x, ~bd.y);
}

struct Node {
  Board bd;
  char alpha;
  char beta;
  char pass;
  char passed;
  __device__ void update(char value) {
    alpha = max(alpha, -value);
  }
};

__shared__ Node nodes_stack[nodesPerBlock][MAX_DEPTH+1];

__device__ int score(const Board &bd) {
  int pnum = __popcll(player(bd));
  int onum = __popcll(opponent(bd));
  if (pnum == onum) return 0;
  if (pnum > onum) return 64 - 2*onum;
  else return 2*pnum - 64;
}
__constant__ ull mask1[4] = {
  0x0080808080808080ULL,
  0x7f00000000000000ULL,
  0x0102040810204000ULL,
  0x0040201008040201ULL
};
__constant__ ull mask2[4] = {
  0x0101010101010100ULL,
  0x00000000000000feULL,
  0x0002040810204080ULL,
  0x8040201008040200ULL
};

__device__ ull flip(const Board &bd, int pos, int index) {
  ull om = opponent(bd);
  if (index) om &= 0x7E7E7E7E7E7E7E7EULL;
  ull mask = mask1[index] >> (63 - pos);
  ull outflank = (0x8000000000000000ULL >> __clzll(~om & mask)) & player(bd);
  ull flipped = (-outflank << 1) & mask;
  mask = mask2[index] << pos;
  outflank = mask & ((om | ~mask) + 1) & player(bd);
  flipped |= (outflank - (outflank != 0)) & mask;
  return flipped;
}

__device__ ull flip_all(const Board &bd, int pos) {
  return flip(bd, pos, 0) | flip(bd, pos, 1) | flip(bd, pos, 2) | flip(bd, pos, 3);
}

struct Arrays {
  const Board *bd_ary;
  int *res_ary;
  int *nodes_count;
  size_t index;
  const size_t size;
};

__device__ bool get_next_node(Arrays &arys, const int node_index, const int simd_index, const int res, const int count) {
  if (simd_index == 0) {
    arys.res_ary[arys.index] = res;
    arys.nodes_count[arys.index] = count;
  }
  arys.index += (blockDim.x * gridDim.x) / simdWidth;
  if (arys.index >= arys.size) return true;
  if (simd_index == 0) {
    Node &root = nodes_stack[node_index][0];
    root.bd = arys.bd_ary[arys.index];
    root.alpha = -64;
    root.beta = 64;
    root.pass = true;
    root.passed = false;
  }
  return false;
}

__device__ void alpha_beta(Arrays &arys) {
  int node_index = threadIdx.x / simdWidth;
  int simd_index = threadIdx.x % simdWidth;
  int stack_index = 0;
  int count = 1;
  while (true) {
    Node &node = nodes_stack[node_index][stack_index];
    ull puttable_bits = puttable(node.bd);
    if (puttable_bits == 0) {
      if (node.pass) {
        if (node.passed) {
          if (stack_index) {
            Node &parent = nodes_stack[node_index][stack_index-1];
            if (simd_index == 0) {
              parent.update(-score(node.bd));
            }
            --stack_index;
          } else {
            if (get_next_node(arys, node_index, simd_index, -score(node.bd), count))
              return;
            count = 1; 
          }
        } else {
          if (simd_index == 0) {
            node.bd = move_pass(node.bd);
            int tmp = node.alpha;
            node.alpha = -node.beta;
            node.beta = -tmp;
            node.passed = true;
          }
        }
      } else {
        if (stack_index) {
          Node &parent = nodes_stack[node_index][stack_index-1];
          if (simd_index == 0) {
            parent.update((node.passed ? -1 : 1) * node.alpha);
          }
          --stack_index;
        } else {
          if (get_next_node(arys, node_index, simd_index, (node.passed ? -1 : 1) * node.alpha, count))
            return;
          count = 1; 
        }
      }
    } else if (node.alpha >= node.beta) {
      if (stack_index) {
        Node &parent = nodes_stack[node_index][stack_index-1];
        if (simd_index == 0) {
          parent.update((node.passed ? -1 : 1) * node.alpha);
        }
        --stack_index;
      } else {
        if (get_next_node(arys, node_index, simd_index, (node.passed ? -1 : 1) * node.alpha, count))
          return;
        count = 1; 
      }
    } else {
      ull bit = puttable_bits & -puttable_bits;
      if (simd_index == 0) {
        node.bd.x ^= bit;
        node.bd.y ^= bit;
      }
      int pos = __popcll(bit-1);
      ull flipped = flip(node.bd, pos, simd_index);
      flipped |= __shfl_xor(flipped, 1);
      flipped |= __shfl_xor(flipped, 2);
      if (flipped) {
        ++stack_index;
        if (simd_index == 0) {
          Node &next = nodes_stack[node_index][stack_index];
          node.pass = false;
          next.bd = move(node.bd, flipped, bit);
          next.alpha = -node.beta;
          next.beta = -node.alpha;
          next.pass = true;
          next.passed = false;
        }
        ++count;
      }
    }
  }
}

struct Node2 {
  Board bd;
  ull puttable;
  unsigned char parent;
  char alpha;
  char beta;
  bool passed;
};

__global__ void search_noordering(const Board *bd_ary, int *res_ary, int *nodes_count, const size_t size) {
  size_t index = (threadIdx.x + blockIdx.x * blockDim.x) / simdWidth;
  if (index < size) {
    int simd_index = threadIdx.x % simdWidth;
    int node_index = threadIdx.x / simdWidth;
    if (simd_index == 0) {
      Node &root = nodes_stack[node_index][0];
      root.bd = bd_ary[index];
      root.alpha = -64;
      root.beta = 64;
      root.pass = true;
      root.passed = false;
    }
    Arrays arys = {
      bd_ary,
      res_ary,
      nodes_count,
      index,
      size
    };
    alpha_beta(arys);
  }
}
/*
__device__ ull puttable_bits(const Board &bd) {
  ull result = 0;
  for (ull empties = ~(bd.player | bd.opponent); empties;) {
    ull bit = empties & -empties;
    empties ^= bit;
    if (flip_all(bd, __popcll(bit-1))) result |= bit;
  }
  return result;
}

__device__ Board move(const Board &bd, const int pos) {
  ull flip_bits = flip_all(bd, pos);
  return Board(bd.opponent ^ flip_bits, (bd.player ^ flip_bits) | (1 << pos));
}

struct NextBoard {
  Board bd;
  int pt;
};

__device__ bool operator<(const NextBoard &lhs, const NextBoard &rhs) {
  return lhs.pt < rhs.pt;
}

__device__ void sort(NextBoard *nb, size_t size) {
  if (size <= 1) return;
  if (nb[1] < nb[0]) {
    NextBoard tmp = nb[0];
    nb[0] = nb[1];
    nb[1] = tmp;
  }
  for (int i = 2; i < size; ++i) {
    NextBoard tmp = nb[i];
    if (tmp < nb[i-1]) {
      int j = i;
      do {
        nb[j] = nb[j-1];
        --j;
      } while(j > 0 && tmp < nb[j-1]);
      nb[j] = tmp;
    }
  }
}

__device__ int alpha_beta_ordered(const Board bd, Node2 *nodes_buf, Arrays &arys) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  ull result = 0;
  Board top_board;
  int top_puttable = -1;
  int puttable_count = 0;
  for (ull empties = ~(bd.player | bd.opponent); empties;) {
    ull bit = empties & -empties;
    empties ^= bit;
    int pos = __ffsll(bit)-1;
    ull flip_bits = flip_all(bd, pos);
    if (flip_bits) {
      ++puttable_count;
      Board next(bd.opponent ^ flip_bits, (bd.player ^ flip_bits) | bit);
      int puttable_count = __popcll(puttable_bits(next));
      if (puttable_count > top_puttable) {
        top_board = next;
        top_puttable = puttable_count;
      }
    }
  }
  if (puttable_count > 0) {
    nodes_buf[index] = {top_board, 0, 0, 0, 0};
  } else {
    nodes_buf[index] = move_pass(bd);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    size_t offset = blockIdx.x * blockDim.x;
    search_noordering<<<1, threadsPerBlock>>>(nodes_buf + offset, arys.res_ary + offset, arys.nodes_count + offset, blockDim.x);
    cudaDeviceSynchronize();
  }
  __syncthreads();
  char alpha = -arys.res_ary[index];
}

__global__ void search_static_ordering(
    const Board *bd, const Node2 *nodes_buf, const size_t pitch, int *alpha, int *res, int *nodes_count, const size_t size) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  alpha_beta_ordered(bd, bd_buf + index * pitch, arys);
}
*/
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
  printf("%zu\n", sizeof(Node));
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
  Board *bd_ary_host = (Board*)malloc(sizeof(Board) * n);
  for (int i = 0; i < n; ++i) {
    bd_ary_host[i] = b81.input_bd(fp, i);
  }
  Board *bd_ary;
  int *res_ary;
  int *nodes_count;
  cudaMalloc(&bd_ary, sizeof(Board) * n);
  cudaMallocManaged(&res_ary, sizeof(int) * n);
  cudaMallocManaged(&nodes_count, sizeof(int) * n);
  cudaMemcpy(bd_ary, bd_ary_host, sizeof(Board) * n, cudaMemcpyHostToDevice);
  cudaMemset(res_ary, 0, sizeof(int) * n);
  cudaMemset(nodes_count, 0, sizeof(int) * n);
  fputs("start solve\n", stderr);
  search_noordering<<<2048, threadsPerBlock>>>(bd_ary, res_ary, nodes_count, n);
  cudaDeviceSynchronize();
  fputs("end solve\n", stderr);
  ull nodes_total = 0;
  for (int i = 0; i < n; ++i) {
    fprintf(ofp, "%d\n", res_ary[i]);
    nodes_total += nodes_count[i];
  }
  fprintf(stderr, "%lu\n", nodes_total);
  free(bd_ary_host);
  cudaFree(bd_ary);
  cudaFree(res_ary);
  cudaFree(nodes_count);
  return 0;
}
