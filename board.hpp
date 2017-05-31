#pragma once
using ull = unsigned long long;

struct Board {
  ull x, y;
  __host__ __device__ Board() = default;
  __host__ __device__ Board(const Board &) = default;
  __host__ __device__ Board& operator=(const Board &) = default;
 private:
  __host__ __device__ Board(const ull &x, const ull &y)
    : x(x), y(y) {}
  friend __host__ __device__ Board make_board(const ull &, const ull &);
  friend __host__ __device__ Board move(const Board &, const ull &, const ull &);
  friend __host__ __device__ Board move_pass(const Board &);
};

inline Board make_board(const ull &player, const ull &opponent) {
  return Board(~opponent, ~player);
}

inline __host__ __device__ ull player(const Board &bd) {
  return bd.x & ~bd.y;
}

inline __host__ __device__ ull opponent(const Board &bd) {
  return bd.y & ~bd.x;
}

inline __host__ __device__ ull puttable(const Board &bd) {
  return bd.x & bd.y;
}

inline __host__ __device__ ull stones(const Board &bd) {
  return bd.x ^ bd.y;
}

// puttable(bd) must be 0
inline __host__ __device__ Board move(const Board &bd, const ull &flipped, const ull &pos_bit) {
  ull old_stones = bd.x | bd.y;
  return Board((bd.y ^ flipped) | ~(old_stones | pos_bit), (bd.x ^ flipped) | ~old_stones);
}

// puttable(bd) must be 0
inline __host__ __device__ Board move_pass(const Board &bd) {
  return Board(~bd.x, ~bd.y);
}

__host__ __device__ int score(const Board &bd);
__host__ __device__ ull flip(const Board &bd, int pos, int index);
__host__ __device__ ull flip_all(const Board &bd, int pos);
