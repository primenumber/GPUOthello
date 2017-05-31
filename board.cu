#include "board.hpp"
#include "x86intrin.h"

__host__ __device__ int score(const Board &bd) {
#ifdef __CUDA_ARCH__
  int pnum = __popcll(player(bd));
  int onum = __popcll(opponent(bd));
#else
  int pnum = _popcnt64(player(bd));
  int onum = _popcnt64(opponent(bd));
#endif
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
constexpr ull mask1_host[4] = {
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
constexpr ull mask2_host[4] = {
  0x0101010101010100ULL,
  0x00000000000000feULL,
  0x0002040810204080ULL,
  0x8040201008040200ULL
};

__host__ __device__ ull flip(const Board &bd, int pos, int index) {
  ull om = opponent(bd);
  if (index) om &= 0x7E7E7E7E7E7E7E7EULL;
#ifdef __CUDA_ARCH__
  ull mask = mask1[index] >> (63 - pos);
  ull outflank = (0x8000000000000000ULL >> __clzll(~om & mask)) & player(bd);
#else
  ull mask = mask1_host[index] >> (63 - pos);
  ull outflank = (0x8000000000000000ULL >> _lzcnt_u64(~om & mask)) & player(bd);
#endif
  ull flipped = (-outflank << 1) & mask;
#ifdef __CUDA_ARCH__
  mask = mask2[index] << pos;
#else
  mask = mask2_host[index] << pos;
#endif
  outflank = mask & ((om | ~mask) + 1) & player(bd);
  flipped |= (outflank - (outflank != 0)) & mask;
  return flipped;
}

__host__ __device__ ull flip_all(const Board &bd, int pos) {
  return flip(bd, pos, 0) | flip(bd, pos, 1) | flip(bd, pos, 2) | flip(bd, pos, 3);
}
