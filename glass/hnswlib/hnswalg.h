#pragma once

#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"
#include <atomic>
#include <cstring>
#include <mutex>
#include <queue>
#include <random>

namespace glass {

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <SymComputerConcept ComputerType> class HierarchicalNSW {
public:
  using dist_t = typename ComputerType::dist_type;
  using pair_t = std::pair<dist_t, tableint>;
  ComputerType computer;

  size_t max_elements_{0};
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0}, revSize_{0.0};
  int maxlevel_{0};

  std::mutex global;
  std::vector<std::mutex> link_list_locks_;

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};

  char *data_level0_memory_{nullptr};
  char **linkLists_{nullptr};
  std::vector<int> element_levels_; // keeps level of each element

  void *dist_func_param_{nullptr};

  std::default_random_engine level_generator_;

  int32_t po = 1;
  int32_t pl = 1;

  HierarchicalNSW(const ComputerType &computer, size_t max_elements,
                  size_t M = 16, size_t ef_construction = 200,
                  size_t random_seed = 100)
      : computer(computer), link_list_locks_(max_elements),
        element_levels_(max_elements), po(5), pl(1) {
    max_elements_ = max_elements;
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_;

    data_level0_memory_ =
        (char *)align_alloc(max_elements_ * size_data_per_element_);
    memset(data_level0_memory_, 0, max_elements * size_data_per_element_);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char **)align_alloc(sizeof(void *) * max_elements_);
    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
  }

  ~HierarchicalNSW() {
    free(data_level0_memory_);
    for (tableint i = 0; i < max_elements_; i++) {
      if (element_levels_[i] > 0)
        free(linkLists_[i]);
    }
    free(linkLists_);
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  std::priority_queue<pair_t> searchBaseLayer(tableint ep_id, int32_t u,
                                              int layer) {
    LinearPool<typename ComputerType::dist_type> pool(
        max_elements_, ef_construction_, ef_construction_);

    pool.insert(ep_id, computer(u, ep_id));
    pool.set_visited(ep_id);

    while (pool.has_next()) {
      auto x = pool.pop();
      std::unique_lock<std::mutex> lock(link_list_locks_[x]);
      int *data;
      if (layer == 0) {
        data = (int *)get_linklist0(x);
      } else {
        data = (int *)get_linklist(x, layer);
      }
      int32_t size = getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);

      for (int32_t j = 0; j < std::min(po, size); ++j) {
        computer.prefetch(datal[j], pl);
      }

      for (int32_t j = 0; j < size; j++) {
        tableint y = *(datal + j);

        if (j + po < size) {
          computer.prefetch(datal[j + po], pl);
        }

        if (pool.check_visited(y)) {
          continue;
        }
        pool.set_visited(y);

        pool.insert(y, computer(u, y));
      }
    }
    std::priority_queue<pair_t> top_candidates;
    for (int i = 0; i < pool.size(); ++i) {
      top_candidates.emplace((dist_t)pool.dist(i), pool.id(i));
    }
    return top_candidates;
  }

  void getNeighborsByHeuristic2(std::priority_queue<pair_t> &top_candidates,
                                const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<pair_t> queue_closest;
    std::vector<pair_t> return_list;
    while (top_candidates.size() > 0) {
      auto [dist, u] = top_candidates.top();
      top_candidates.pop();
      queue_closest.emplace(-dist, u);
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M) {
        break;
      }
      auto [dist, u] = queue_closest.top();
      queue_closest.pop();
      dist_t dist_to_query = -dist;
      bool good = true;

      for (auto [_, v] : return_list) {
        dist_t curdist = computer(u, v);
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back({dist, u});
      }
    }

    for (auto [dist, u] : return_list) {
      top_candidates.emplace(-dist, u);
    }
  }

  linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_);
  }

  linklistsizeint *get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] +
                               (level - 1) * size_links_per_element_);
  }

  tableint mutuallyConnectNewElement(
      tableint u, std::priority_queue<pair_t> &top_candidates, int level) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();
    linklistsizeint *ll_cur;
    if (level == 0)
      ll_cur = get_linklist0(u);
    else
      ll_cur = get_linklist(u, level);

    setListCount(ll_cur, selectedNeighbors.size());
    tableint *data = (tableint *)(ll_cur + 1);
    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      data[idx] = selectedNeighbors[idx];
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      std::unique_lock<std::mutex> lock(
          link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint *ll_other;
      if (level == 0)
        ll_other = get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = get_linklist(selectedNeighbors[idx], level);

      size_t sz_link_list_other = getListCount(ll_other);

      tableint *data = (tableint *)(ll_other + 1);

      if (sz_link_list_other < Mcurmax) {
        data[sz_link_list_other] = u;
        setListCount(ll_other, sz_link_list_other + 1);
      } else {
        // finding the "weakest" element to replace it with the new one
        dist_t d_max = computer(u, selectedNeighbors[idx]);
        // Heuristic:
        std::priority_queue<pair_t> candidates;
        candidates.emplace(d_max, u);

        for (size_t j = 0; j < sz_link_list_other; j++) {
          candidates.emplace(computer(data[j], selectedNeighbors[idx]),
                             data[j]);
        }

        getNeighborsByHeuristic2(candidates, Mcurmax);

        int indx = 0;
        while (candidates.size() > 0) {
          data[indx] = candidates.top().second;
          candidates.pop();
          indx++;
        }

        setListCount(ll_other, indx);
        // Nearest K:
        /*int indx = -1;
        for (int j = 0; j < sz_link_list_other; j++) {
            dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
        getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
                indx = j;
                d_max = d;
            }
        }
        if (indx >= 0) {
            data[indx] = cur_c;
        } */
      }
    }
    return next_closest_entry_point;
  }

  unsigned short int getListCount(linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
  }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
  }

  void addPoint(int32_t u, int level = -1) {
    std::unique_lock<std::mutex> lock_el(link_list_locks_[u]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0)
      curlevel = level;

    element_levels_[u] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy)
      templock.unlock();
    tableint currObj = enterpoint_node_;

    if (curlevel) {
      linkLists_[u] =
          (char *)align_alloc(size_links_per_element_ * curlevel + 1);
      memset(linkLists_[u], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        dist_t curdist = computer(u, currObj);
        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;
            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
            data = get_linklist(currObj, level);
            int size = getListCount(data);

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              if (i + 1 < size) {
                computer.prefetch(datal[i + 1], 1);
              }
              tableint cand = datal[i];
              dist_t d = computer(u, cand);
              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }

      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
        std::priority_queue<pair_t> top_candidates =
            searchBaseLayer(currObj, u, level);
        currObj = mutuallyConnectNewElement(u, top_candidates, level);
      }
    } else {
      // Do nothing for the first element
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }

    if (curlevel > maxlevelcopy) {
      enterpoint_node_ = u;
      maxlevel_ = curlevel;
    }
  }
};

} // namespace glass
