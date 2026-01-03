#include "shvetsova_k_rad_sort_batch_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

ShvetsovaKRadSortBatchMergeSEQ::ShvetsovaKRadSortBatchMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ShvetsovaKRadSortBatchMergeSEQ::ValidationImpl() {
  return true;
}

bool ShvetsovaKRadSortBatchMergeSEQ::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool ShvetsovaKRadSortBatchMergeSEQ::RunImpl() {
  if (data_.empty()) {
    GetOutput() = OutType{};
    return true;
  }

  RadixSort(data_);
  BatcherOddEvenMergeSort(data_, 0, static_cast<int>(data_.size()));

  GetOutput().assign(data_.begin(), data_.end());

  return true;
}

bool ShvetsovaKRadSortBatchMergeSEQ::PostProcessingImpl() {
  return true;
}

// доп функции //

void ShvetsovaKRadSortBatchMergeSEQ::CompareAndSwap(std::vector<int> &vec, int i, int j) {
  if (vec.at(i) > vec.at(j)) {
    std::swap(vec.at(i), vec.at(j));
  }
}

void ShvetsovaKRadSortBatchMergeSEQ::RadixSort(std::vector<int> &vec) {
  if (vec.empty()) {
    return;
  }

  const int base = 10;
  int max_val = 0;
  for (double x : vec) {
    max_val = std::max(max_val, static_cast<int>(std::abs(x)));
  }

  for (int exp = 1; max_val / exp > 0; exp *= base) {
    std::vector<int> output(vec.size());
    std::array<int, base> count{};

    for (double x : vec) {
      int digit = (static_cast<int>(std::abs(x)) / exp) % base;
      count.at(digit)++;
    }

    for (int i = 1; i < base; i++) {
      count.at(i) += count.at(i - 1);
    }

    for (int i = static_cast<int>(vec.size()) - 1; i >= 0; i--) {
      int digit = (static_cast<int>(std::abs(vec.at(i))) / exp) % base;
      output.at(--count.at(digit)) = vec.at(i);
    }

    vec = std::move(output);
  }
}

void ShvetsovaKRadSortBatchMergeSEQ::ExecuteBatcherStep(std::vector<int> &vec, int left, int n, int p, int k) {
  for (int j = k % p; j <= n - 1 - k; j += 2 * k) {
    int limit = std::min(k, n - j - k);
    for (int i = 0; i < limit; ++i) {
      int idx1 = left + j + i;
      int idx2 = left + j + i + k;
      if (idx1 / (p * 2) == idx2 / (p * 2)) {
        CompareAndSwap(vec, idx1, idx2);
      }
    }
  }
}

void ShvetsovaKRadSortBatchMergeSEQ::BatcherOddEvenMergeSort(std::vector<int> &vec, int left, int right) {
  int n = right - left;
  if (n <= 1) {
    return;
  }

  for (int pp = 1; pp < n; pp <<= 1) {
    for (int k = pp; k >= 1; k >>= 1) {
      ExecuteBatcherStep(vec, left, n, pp, k);
    }
  }
}

}  // namespace shvetsova_k_rad_sort_batch_merge
