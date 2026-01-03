#include "shvetsova_k_rad_sort_batch_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

ShvetsovaKRadSortBatchMergeMPI::ShvetsovaKRadSortBatchMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ShvetsovaKRadSortBatchMergeMPI::ValidationImpl() {
  return true;
}

bool ShvetsovaKRadSortBatchMergeMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool ShvetsovaKRadSortBatchMergeMPI::RunImpl() {
  int proc_count = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = (rank == 0) ? static_cast<int>(data_.size()) : 0;
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (size == 0) {
    if (rank == 0) {
      GetOutput() = OutType{};
    }
    return true;
  }

  // Выравнивание размера для равномерного распределения
  int padded_size = size;
  if (size % proc_count != 0) {
    padded_size = size + (proc_count - size % proc_count);
  }

  // Добавление максимальных значений для выравнивания
  if (rank == 0) {
    data_.resize(padded_size, INT_MAX);
  }

  // Создание компараторов Бэтчера
  std::vector<std::pair<int, int>> comparators;
  if (rank == 0) {
    comparators = GenerateBatcherComparators(proc_count);
  }

  // Рассылка компараторов всем процессам
  size_t comparators_count = 0;
  if (rank == 0) {
    comparators_count = comparators.size();
  }

  MPI_Bcast(&comparators_count, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    comparators.resize(comparators_count);
  }

  // Отправляем компараторы как плоский массив пар
  std::vector<int> flat_comparators(comparators_count * 2);
  if (rank == 0) {
    for (size_t i = 0; i < comparators_count; i++) {
      flat_comparators[2 * i] = comparators[i].first;
      flat_comparators[2 * i + 1] = comparators[i].second;
    }
  }

  MPI_Bcast(flat_comparators.data(), static_cast<int>(comparators_count * 2), MPI_INT, 0, MPI_COMM_WORLD);

  // Восстанавливаем компараторы из плоского массива
  for (size_t i = 0; i < comparators_count; i++) {
    comparators[i] = {flat_comparators[2 * i], flat_comparators[2 * i + 1]};
  }

  // Распределение данных
  std::vector<int> counts(proc_count);
  std::vector<int> displs(proc_count);
  CreateDistribution(proc_count, padded_size, counts, displs);

  std::vector<int> local(counts.at(rank));
  ScatterData(data_, local, counts, displs, rank);

  // Локальная сортировка
  RadixSortWithNegatives(local);

  // Слияние с использованием сети Бэтчера
  ProcessBatcherComparators(comparators, rank, counts, local);

  // Сбор и рассылка результатов
  data_.resize(static_cast<std::size_t>(padded_size));
  GatherAndBroadcast(data_, local, counts, displs, rank);

  // Удаление добавленных значений и сохранение результата
  if (rank == 0) {
    data_.resize(static_cast<std::size_t>(size));
  }

  GetOutput() = data_;
  return true;
}

// ---------------- Distribution ----------------

void ShvetsovaKRadSortBatchMergeMPI::CreateDistribution(int proc_count, int size, std::vector<int> &counts,
                                                        std::vector<int> &displs) {
  int base = size / proc_count;
  int rem = size % proc_count;
  int offset = 0;

  for (int i = 0; i < proc_count; ++i) {
    counts.at(i) = base + (i < rem ? 1 : 0);
    displs.at(i) = offset;
    offset += counts.at(i);
  }
}

// ---------------- MPI helpers ----------------

void ShvetsovaKRadSortBatchMergeMPI::ScatterData(const std::vector<int> &data, std::vector<int> &local,
                                                 const std::vector<int> &counts, const std::vector<int> &displs,
                                                 int rank) {
  MPI_Scatterv(rank == 0 ? data.data() : nullptr, counts.data(), displs.data(), MPI_INT, local.data(),
               static_cast<int>(local.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

void ShvetsovaKRadSortBatchMergeMPI::GatherAndBroadcast(std::vector<int> &data, const std::vector<int> &local,
                                                        const std::vector<int> &counts, const std::vector<int> &displs,
                                                        int rank) {
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_INT, rank == 0 ? data.data() : nullptr, counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(data.data(), static_cast<int>(data.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

// ---------------- Batcher network ----------------

void ShvetsovaKRadSortBatchMergeMPI::AddComparator(std::vector<std::pair<int, int>> &comparators, int a, int b) {
  if (a < b) {
    comparators.emplace_back(a, b);
  } else if (b < a) {
    comparators.emplace_back(b, a);
  }
}

void ShvetsovaKRadSortBatchMergeMPI::ProcessJStep(int power, int k, int j, int num_processes,
                                                  std::vector<std::pair<int, int>> &comparators) {
  int limit = std::min(k, num_processes - j - k);
  for (int i = 0; i < limit; i++) {
    if ((j + i) / (power * 2) == (j + i + k) / (power * 2)) {
      const auto first_index = j + i;
      const auto second_index = first_index + k;
      AddComparator(comparators, first_index, second_index);
    }
  }
}

void ShvetsovaKRadSortBatchMergeMPI::ProcessKStep(int power, int k, int num_processes,
                                                  std::vector<std::pair<int, int>> &comparators) {
  for (int j = k % power; j < num_processes - k; j += 2 * k) {
    ProcessJStep(power, k, j, num_processes, comparators);
  }
}

void ShvetsovaKRadSortBatchMergeMPI::ProcessPowerStep(int power, int num_processes,
                                                      std::vector<std::pair<int, int>> &comparators) {
  for (int k = power; k >= 1; k /= 2) {
    ProcessKStep(power, k, num_processes, comparators);
  }
}

std::vector<std::pair<int, int>> ShvetsovaKRadSortBatchMergeMPI::GenerateBatcherComparators(int num_processes) {
  std::vector<std::pair<int, int>> comparators;
  for (int power = 1; power < num_processes; power *= 2) {
    ProcessPowerStep(power, num_processes, comparators);
  }
  return comparators;
}

std::vector<int> ShvetsovaKRadSortBatchMergeMPI::MergeTwoSorted(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());

  size_t i = 0;
  size_t j = 0;

  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      result.push_back(a[i]);
      i++;
    } else {
      result.push_back(b[j]);
      j++;
    }
  }

  // Добавляем оставшиеся элементы
  AddRemainingElements(result, a, i);
  AddRemainingElements(result, b, j);

  return result;
}

// Вспомогательная функция для добавления оставшихся элементов
void ShvetsovaKRadSortBatchMergeMPI::AddRemainingElements(std::vector<int> &result, const std::vector<int> &source,
                                                          size_t &index) {
  while (index < source.size()) {
    result.push_back(source[index]);
    index++;
  }
}

// Обработка компараторов Бэтчера
void ShvetsovaKRadSortBatchMergeMPI::ProcessBatcherComparators(const std::vector<std::pair<int, int>> &comparators,
                                                               int rank, const std::vector<int> &counts,
                                                               std::vector<int> &local) {
  for (const auto &comparator : comparators) {
    int a = comparator.first;
    int b = comparator.second;

    if (rank == a || rank == b) {
      ProcessComparator(comparator, rank, counts, local);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

// Обработка одного компаратора
void ShvetsovaKRadSortBatchMergeMPI::ProcessComparator(const std::pair<int, int> &comparator, int rank,
                                                       const std::vector<int> &counts, std::vector<int> &local) {
  int a = comparator.first;
  int b = comparator.second;
  int partner = (rank == a) ? b : a;
  int partner_size = counts.at(partner);

  // Обмен данными с партнером
  std::vector<int> received_data(partner_size);
  MPI_Sendrecv(local.data(), static_cast<int>(local.size()), MPI_INT, partner, 0, received_data.data(), partner_size,
               MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Слияние и распределение элементов
  MergeAndDistribute(local, received_data, rank, a, b);
}

// Слияние и распределение элементов
void ShvetsovaKRadSortBatchMergeMPI::MergeAndDistribute(std::vector<int> &local, const std::vector<int> &received,
                                                        int rank, int a, int b) {
  std::vector<int> merged = MergeTwoSorted(local, received);

  if (rank == std::min(a, b)) {
    // Оставляем меньшие элементы
    std::copy(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(local.size()), local.begin());
  } else {
    // Оставляем большие элементы
    std::copy(merged.begin() + static_cast<std::ptrdiff_t>(local.size()), merged.end(), local.begin());
  }
}

// ---------------- Radix sort helpers ----------------

int ShvetsovaKRadSortBatchMergeMPI::GetDigit(int num, int digit_place) {
  return (num / digit_place) % 10;
}

void ShvetsovaKRadSortBatchMergeMPI::CountDigits(const std::vector<int> &arr, int digit_place,
                                                 std::vector<int> &count) {
  for (int value : arr) {
    int digit = (value / digit_place) % 10;
    count[digit]++;
  }
}

void ShvetsovaKRadSortBatchMergeMPI::AccumulateCounts(std::vector<int> &count) {
  for (int i = 1; i < static_cast<int>(count.size()); i++) {
    count[i] += count[i - 1];
  }
}

void ShvetsovaKRadSortBatchMergeMPI::BuildOutputArray(const std::vector<int> &arr, int digit_place,
                                                      std::vector<int> &count, std::vector<int> &output) {
  for (int i = static_cast<int>(arr.size()) - 1; i >= 0; i--) {
    int digit = (arr[i] / digit_place) % 10;
    output[count[digit] - 1] = arr[i];
    count[digit]--;
  }
}

void ShvetsovaKRadSortBatchMergeMPI::CopyBackToArray(std::vector<int> &arr, const std::vector<int> &output) {
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] = output[i];
  }
}

void ShvetsovaKRadSortBatchMergeMPI::CountingSort(std::vector<int> &arr, int digit_place) {
  int n = static_cast<int>(arr.size());
  const int range = 10;
  std::vector<int> output(n);
  std::vector<int> count(range, 0);

  // Подсчет цифр
  CountDigits(arr, digit_place, count);

  // Накопление счетчиков
  AccumulateCounts(count);

  // Построение выходного массива
  BuildOutputArray(arr, digit_place, count, output);

  // Копирование обратно в исходный массив
  CopyBackToArray(arr, output);
}

void ShvetsovaKRadSortBatchMergeMPI::RadixSort(std::vector<int> &arr) {
  if (arr.empty()) {
    return;
  }

  int max_num = *std::max_element(arr.begin(), arr.end());

  for (int digit_place = 1; max_num / digit_place > 0; digit_place *= 10) {
    CountingSort(arr, digit_place);
  }
}

std::vector<int> ShvetsovaKRadSortBatchMergeMPI::ExtractNegatives(const std::vector<int> &arr) {
  std::vector<int> negatives;
  for (int value : arr) {
    if (value < 0) {
      negatives.push_back(-value);
    }
  }
  return negatives;
}

std::vector<int> ShvetsovaKRadSortBatchMergeMPI::ExtractNonNegatives(const std::vector<int> &arr) {
  std::vector<int> non_negatives;
  for (int value : arr) {
    if (value >= 0) {
      non_negatives.push_back(value);
    }
  }
  return non_negatives;
}

void ShvetsovaKRadSortBatchMergeMPI::SortSeparatedArrays(std::vector<int> &negatives, std::vector<int> &non_negatives) {
  RadixSort(negatives);
  RadixSort(non_negatives);
}

void ShvetsovaKRadSortBatchMergeMPI::ReverseAndRestoreSign(std::vector<int> &negatives) {
  for (size_t i = 0; i < negatives.size() / 2; i++) {
    std::swap(negatives[i], negatives[negatives.size() - 1 - i]);
  }

  for (int &negative : negatives) {
    negative = -negative;
  }
}

void ShvetsovaKRadSortBatchMergeMPI::CombineSortedArrays(std::vector<int> &arr, std::vector<int> &negatives,
                                                         std::vector<int> &non_negatives) {
  // Реверс и восстановление знака для отрицательных чисел
  ReverseAndRestoreSign(negatives);

  // Очистка и объединение
  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), non_negatives.begin(), non_negatives.end());
}

void ShvetsovaKRadSortBatchMergeMPI::RadixSortWithNegatives(std::vector<int> &arr) {
  if (arr.empty()) {
    return;
  }

  // Разделение на отрицательные и неотрицательные числа
  std::vector<int> negatives = ExtractNegatives(arr);
  std::vector<int> non_negatives = ExtractNonNegatives(arr);

  // Сортировка разделенных массивов
  SortSeparatedArrays(negatives, non_negatives);

  // Объединение результатов
  CombineSortedArrays(arr, negatives, non_negatives);
}

bool ShvetsovaKRadSortBatchMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_rad_sort_batch_merge
