#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

#include "glass/common.hpp"
#include "glass/memory.hpp"

namespace glass {

template <typename Pool>
concept NeighborPoolConcept =
    requires(Pool pool, int32_t u, typename Pool::dist_type dist, int32_t *ids,
             int32_t length, int32_t n, int32_t ef, int32_t cap) {
      { pool.reset(n, ef, cap) };
      { pool.insert(u, dist) } -> std::same_as<bool>;
      { pool.pop() } -> std::same_as<int32_t>;
      { pool.has_next() } -> std::same_as<bool>;
      { pool.set_visited(u) };
      { pool.check_visited(u) } -> std::same_as<bool>;
      { pool.to_sorted(ids, length) };
    };

template <typename Block = uint64_t> struct Bitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int32_t nb = 0;
  int nbytes = 0;
  Block *data = nullptr;

  Bitset() = default;

  explicit Bitset(int n)
      : nb(n), nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((Block *)align_alloc(nbytes)) {}

  friend void swap(Bitset &lhs, Bitset &rhs) {
    using std::swap;
    swap(lhs.nb, rhs.nb);
    swap(lhs.nbytes, rhs.nbytes);
    swap(lhs.data, rhs.data);
  }

  Bitset(const Bitset &) = delete;

  Bitset(Bitset &&rhs) { swap(*this, rhs); }

  Bitset &operator=(const Bitset &) = delete;

  Bitset &operator=(Bitset &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  ~Bitset() {
    if (data) {
      free(data);
    }
  }

  void reset(int32_t n) {
    if (n != nb) {
      // printf("Reset bitset size to %d\n", n);
      *this = Bitset(n);
    } else {
      memset(data, 0, nbytes);
    }
  }

  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }

  bool get(int i) const {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }
};

template <typename Block = uint32_t> struct TwoLevelBitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int32_t nb = 0;
  int nbytes = 0;
  Block *data = nullptr;
  std::vector<bool> visited;

  TwoLevelBitset() = default;

  explicit TwoLevelBitset(int n)
      : nb(n), nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((Block *)align_alloc(nbytes)),
        visited((n + block_size - 1) / block_size) {}

  friend void swap(TwoLevelBitset &lhs, TwoLevelBitset &rhs) {
    using std::swap;
    swap(lhs.nb, rhs.nb);
    swap(lhs.nbytes, rhs.nbytes);
    swap(lhs.data, rhs.data);
    swap(lhs.visited, rhs.visited);
  }

  TwoLevelBitset(const TwoLevelBitset &) = delete;

  TwoLevelBitset(TwoLevelBitset &&rhs) { swap(*this, rhs); }

  TwoLevelBitset &operator=(const TwoLevelBitset &) = delete;

  TwoLevelBitset &operator=(TwoLevelBitset &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  ~TwoLevelBitset() {
    if (data) {
      free(data);
    }
  }

  void reset(int32_t n) {
    if (n != nb) {
      *this = TwoLevelBitset(n);
    } else {
      std::fill(visited.begin(), visited.end(), false);
    }
  }

  void set(int i) {
    if (!visited[i / block_size]) {
      visited[i / block_size] = true;
      data[i / block_size] = 0;
    }
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }

  bool get(int i) const {
    if (!visited[i / block_size]) {
      return false;
    }
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }
};

struct BitsetStl {
  int32_t nb = 0;
  std::vector<bool> data;

  BitsetStl() = default;

  explicit BitsetStl(int n) : data(n) {}

  BitsetStl(const BitsetStl &) = delete;

  BitsetStl(BitsetStl &&rhs) = default;

  BitsetStl &operator=(const BitsetStl &) = delete;

  BitsetStl &operator=(BitsetStl &&rhs) = default;

  void reset(int32_t n) {
    if (n != nb) {
      *this = BitsetStl(n);
    } else {
      std::fill(data.begin(), data.end(), false);
    }
  }

  void set(int i) { data[i] = true; }

  bool get(int i) const { return data[i]; }
};

struct BitsetSet {
  std::unordered_set<int32_t> data;

  BitsetSet() = default;

  explicit BitsetSet(int32_t /*n*/) {}

  BitsetSet(const BitsetSet &) = delete;

  BitsetSet(BitsetSet &&rhs) = default;

  BitsetSet &operator=(const BitsetSet &) = delete;

  BitsetSet &operator=(BitsetSet &&rhs) = default;

  void reset(int32_t /*n*/) { data.clear(); }

  void set(int i) { data.insert(i); }

  bool get(int i) const { return data.count(i); }
};

template <typename dist_t = float> struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }

  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};

template <typename T> struct MaxHeap {
  int sz = 0, capacity;
  std::vector<T> pool;

  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}

  void push(T x) {
    if (sz < capacity) {
      pool[sz] = x;
      std::push_heap(pool.begin(), pool.begin() + ++sz);
    } else if (x < pool[0]) {
      sift_down(0, x);
    }
  }

  auto pop() {
    std::pop_heap(pool.begin(), pool.begin() + sz--);
    return pool[sz];
  }

  void sift_down(int i, T x) {
    pool[i] = x;
    for (; 2 * i + 1 < sz;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l] > x) {
        j = l;
      }
      if (r < sz && pool[r] > std::max(pool[l], x)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = x;
  }

  int32_t size() const { return sz; }

  bool empty() const { return size() == 0; }

  auto top() const { return pool[0]; }
};

template <typename dist_t> struct MinMaxHeap {

  explicit MinMaxHeap(int capacity) : capacity(capacity), pool(capacity) {}

  bool push(int u, dist_t dist) {
    if (cur == capacity) {
      if (dist >= pool[0].distance) {
        return false;
      }
      if (pool[0].id != -1) {
        sz--;
      }
      std::pop_heap(pool.begin(), pool.begin() + cur--);
    }
    pool[cur] = {u, dist};
    std::push_heap(pool.begin(), pool.begin() + ++cur);
    sz++;
    return true;
  }

  int32_t size() const { return sz; }

  dist_t max() const { return pool[0].distance; }

  void clear() { sz = cur = 0; }

  int pop_min() {
    int i = cur - 1;
    for (; i >= 0 && pool[i].id == -1; --i)
      ;
    if (i == -1) {
      return -1;
    }
    int imin = i;
    dist_t vmin = pool[i].distance;
    for (; --i >= 0;) {
      if (pool[i].id != -1 && pool[i].distance < vmin) {
        vmin = pool[i].distance;
        imin = i;
      }
    }
    int ret = pool[imin].id;
    pool[imin].id = -1;
    --sz;
    return ret;
  }

  int32_t count_below(float thresh) const {
    int32_t n_below = 0;
    for (int32_t i = 0; i < cur; ++i) {
      if (pool[i].distance < thresh) {
        n_below++;
      }
    }
    return n_below;
  }

  int sz = 0, cur = 0, capacity;
  std::vector<Neighbor<dist_t>> pool;
};

template <typename dist_t, typename BitsetType = Bitset<>> struct LinearPool {
  using dist_type = dist_t;

  LinearPool() = default;

  LinearPool(int n, int ef, int capacity)
      : nb(n), ef_(ef), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  friend void swap(LinearPool &lhs, LinearPool &rhs) {
    using std::swap;
    swap(lhs.nb, rhs.nb);
    swap(lhs.size_, rhs.size_);
    swap(lhs.cur_, rhs.cur_);
    swap(lhs.ef_, rhs.ef_);
    swap(lhs.capacity_, rhs.capacity_);
    swap(lhs.data_, rhs.data_);
    swap(lhs.vis, rhs.vis);
  }

  LinearPool(const LinearPool &) = delete;

  LinearPool(LinearPool &&rhs) { swap(*this, rhs); }

  LinearPool &operator=(const LinearPool &) = delete;

  LinearPool &operator=(LinearPool &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  void reset(int32_t n, int ef, int32_t cap) {
    nb = n;
    size_ = cur_ = 0;
    ef_ = ef;
    capacity_ = cap;
    if (data_.size() < cap + 1) {
      data_.resize(cap + 1);
    }
    vis.reset(n);
  }

  GLASS_INLINE int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
    // int len = size_;
    // int loc = 0;
    // while (len > 1) {
    //   int half = len / 2;
    //   loc += (dist > data_[loc + half - 1].distance) * half;
    //   len -= half;
    // }
    // return loc;
  }

  GLASS_INLINE bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_ && cur_ < ef_; }
  int id(int i) const { return get_id(data_[i].id); }
  dist_type dist(int i) const { return data_[i].distance; }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  void set_visited(int32_t u) { vis.set(u); }
  bool check_visited(int32_t u) const { return vis.get(u); }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) const { return id >> 31 & 1; }

  void to_sorted(int32_t *ids, int32_t length) const {
    for (int32_t i = 0; i < length; ++i) {
      ids[i] = id(i);
    }
  }

  int nb, size_ = 0, cur_ = 0, ef_, capacity_;
  std::vector<Neighbor<dist_t>> data_;
  BitsetType vis;
};

struct FlagNeighbor {
  int id;
  float distance;
  bool flag;

  FlagNeighbor() = default;
  FlagNeighbor(int id, float distance, bool f)
      : id(id), distance(distance), flag(f) {}

  inline friend bool operator<(const FlagNeighbor &lhs,
                               const FlagNeighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
};

using Node = Neighbor<float>;

} // namespace glass
