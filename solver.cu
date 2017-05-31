#include "solver.hpp"
constexpr int threadsPerBlock = 128;
constexpr int simdWidth = 4;
constexpr int nodesPerBlock = threadsPerBlock/simdWidth;
constexpr int MAX_DEPTH = 10;

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

struct Arrays {
  const Board *bd_ary;
  int *res_ary;
  int *nodes_count;
  size_t index;
  const size_t size;
};

__device__ bool get_next_node(Arrays &arys, const int res, const int count) {
  if (threadIdx.x == 0) {
    arys.res_ary[arys.index] = res;
    arys.nodes_count[arys.index] = count;
  }
  arys.index += blockDim.y * gridDim.y;
  if (arys.index >= arys.size) return true;
  if (threadIdx.x == 0) {
    Node &root = nodes_stack[threadIdx.y][0];
    root.bd = arys.bd_ary[arys.index];
    root.alpha = -64;
    root.beta = 64;
    root.pass = true;
    root.passed = false;
  }
  return false;
}

__device__ void commit(int &stack_index) {
  if (threadIdx.x == 0) {
    Node &node = nodes_stack[threadIdx.y][stack_index];
    Node &parent = nodes_stack[threadIdx.y][stack_index-1];
    parent.update((node.passed ? -1 : 1) * node.alpha);
  }
  --stack_index;
}

__device__ void alpha_beta(Arrays &arys) {
  int node_index = threadIdx.y;
  int simd_index = threadIdx.x;
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
            if (get_next_node(arys, -score(node.bd), count))
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
          commit(stack_index);
        } else {
          if (get_next_node(arys, (node.passed ? -1 : 1) * node.alpha, count))
            return;
          count = 1; 
        }
      }
    } else if (node.alpha >= node.beta) {
      if (stack_index) {
        commit(stack_index);
      } else {
        if (get_next_node(arys, (node.passed ? -1 : 1) * node.alpha, count))
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

__global__ void search_noordering(const Board *bd_ary, int *res_ary, int *nodes_count, const size_t size) {
  size_t index = threadIdx.y + blockIdx.y * blockDim.y;
  if (index < size) {
    int simd_index = threadIdx.x;
    int node_index = threadIdx.y;
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

void solve(const Board *board, int * const result, int * const nodes_count, const size_t length) {
  Board *board_dev;
  int *nodes_count_dev;
  int *result_dev;
  cudaMalloc((void**)&board_dev,        sizeof(Board) * length);
  cudaMalloc((void**)&result_dev,       sizeof(int)   * length);
  cudaMalloc((void**)&nodes_count_dev,  sizeof(int)   * length);
  cudaMemcpy(board_dev, board, sizeof(Board) * length, cudaMemcpyHostToDevice);
  dim3 block(simdWidth, nodesPerBlock);
  dim3 grid(1, 2048);
  search_noordering<<<grid, block>>>(board_dev, result_dev, nodes_count_dev, length);
  cudaMemcpy(result, result_dev, sizeof(int) * length, cudaMemcpyDeviceToHost);
  cudaMemcpy(nodes_count, nodes_count_dev, sizeof(int) * length, cudaMemcpyDeviceToHost);
  cudaFree(board_dev);
  cudaFree(result_dev);
  cudaFree(nodes_count_dev);
}
