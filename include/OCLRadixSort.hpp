#ifndef __OCLRADIXSORT_HPP
#define __OCLRADIXSORT_HPP

//
#define __CL_ENABLE_EXCEPTIONS

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include <algorithm>
#include <numeric>

#include <boost/compute.hpp>
#include <sstream>
#include <string_view>

namespace bc = boost::compute;

template <size_t v> struct is_power_of_2 {
  static constexpr bool value = v == 0 || ((v - 1) & v) == 0;
};

// cl type for integral type
template <typename T,
          typename = std::enable_if_t<
              std::is_unsigned<T>::value && std::is_integral<T>::value &&
              sizeof(T) <= 8 && is_power_of_2<sizeof(T)>::value>>
struct cl_type_name {
  inline static constexpr const char *name() noexcept {
    switch (sizeof(T)) {
    case 1:
      return "uchar";
    case 2:
      return "ushort";
    case 4:
      return "uint";
    case 8:
      return "ulong";

    default:
      return "XXX bad type XXX";
    }
  }
};

template <typename T> struct Opt {
  const char *name;
  const T value;

  Opt(const char *name, T value) : name(name), value(value) {}
};

namespace std {

template <typename T> ostream &operator<<(ostream &stream, const Opt<T> &opt) {
  return stream << " -D" << opt.name << '=' << opt.value;
}

} // namespace std

using namespace std::literals::string_view_literals;

const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void histogram(
        // in
        const global KEY_TYPE *keys, const ulong length, const int pass,
        // out
        global KEY_TYPE *global_histograms,
        // local
        local KEY_TYPE *histograms) {
  uint index = get_local_id(0) + get_group_id(0);
  global_histograms[index] += 8;
});

/* const char source[] = R"SOURCE( */
/* #ifdef HOST_PTR_IS_32bit */
/* #define SIZE uint */
/* #else */
/* #define SIZE ulong */
/* #endif */

/* inline SIZE index(SIZE i, SIZE n) { */
/* #ifdef TRANSPOSE */
/*   const SIZE k = i / (n / (_GROUPS * _ITEMS)); */
/*   const SIZE l = i % (n / (_GROUPS * _ITEMS)); */
/*   return l * (_GROUPS * _ITEMS) + k; */
/* #else */
/*   return i; */
/* #endif */
/* } */

/* __kernel void histogram( */
/*     // in */
/*     const global KEY_TYPE *keys, const SIZE length, const int pass, */
/*     // out */
/*     global KEY_TYPE *global_histograms, */
/*     // local */
/*     local KEY_TYPE *histograms) { */
/*   const uint group = get_group_id(0); */
/*   const uint item = get_local_id(0); */
/*   const uint i_g = get_global_id(0); */

/* #if (__OPENCL_VERSION__ >= 200) */
/*   __attribute__((opencl_unroll_hint(_RADIX))) */
/* #endif */
/*   for (int i = 0; i < _RADIX; ++i) { */
/*     histograms[i * _ITEMS + item] = 0; */
/*   } */
/*   barrier(CLK_LOCAL_MEM_FENCE); */

/*   const SIZE size = length / (_GROUPS * _ITEMS); */
/*   const SIZE start = i_g * size; */

/*   for (SIZE i = start; i < start + size; ++i) { */
/*     const KEY_TYPE key = keys[index(i, length)]; */
/*     const KEY_TYPE shortKey = ((key >> (pass * _BITS)) & (_RADIX - 1)); */
/*     ++histograms[shortKey * _ITEMS + item]; */
/*   } */
/*   barrier(CLK_LOCAL_MEM_FENCE); */

/* #if (__OPENCL_VERSION__ >= 200) */
/*   __attribute__((opencl_unroll_hint(_RADIX))) */
/* #endif */
/*   for (int i = 0; i < _RADIX; ++i) { */
/*     global_histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item] = */
/*         histograms[i * _ITEMS + item]; */
/*   } */
/* })SOURCE"; */
/* const char source[] = R"SOURCE( */
/* #ifdef HOST_PTR_IS_32bit */
/* #define SIZE uint */
/* #else */
/* #define SIZE ulong */
/* #endif */

/*     inline SIZE index(SIZE i, SIZE n) { */
/* #ifdef TRANSPOSE */
/*       const SIZE k = i / (n / (_GROUPS * _ITEMS)); */
/*       const SIZE l = i % (n / (_GROUPS * _ITEMS)); */
/*       return l * (_GROUPS * _ITEMS) + k; */
/* #else */
/*       return i; */
/* #endif */
/*     } */

/*     // this kernel creates histograms from key vector */
/*     // defines required: _RADIX (radix), _BITS (size of radix in bits) */
/*     // it is possible to unroll 2 loops inside this kernel, take this into */
/*     // account when providing options to CL C compiler */
/*     //__attribute__((vec_type_hint(KEY_TYPE))) */ 
/*     __kernel void histogram( */
/*         // in */
/*         const global KEY_TYPE *keys, const SIZE length, const int pass, */
/*         // out */
/*         global KEY_TYPE *global_histograms, */
/*         // local */
/*         local KEY_TYPE *histograms) { */
/*       const uint group = get_group_id(0); */
/*       const uint item = get_local_id(0); */
/*       const uint i_g = get_global_id(0); */

/* #if (__OPENCL_VERSION__ >= 200) */
/*       __attribute__((opencl_unroll_hint(_RADIX))) */
/* #endif */
/*       for (int i = 0; i < _RADIX; ++i) { */
/*         histograms[i * _ITEMS + item] = 0; */
/*       } */
/*       barrier(CLK_LOCAL_MEM_FENCE); */

/*       const SIZE size = length / (_GROUPS * _ITEMS); */
/*       const SIZE start = i_g * size; */

/*       for (SIZE i = start; i < start + size; ++i) { */
/*         const KEY_TYPE key = keys[index(i, length)]; */
/*         const KEY_TYPE shortKey = ((key >> (pass * _BITS)) & (_RADIX - 1)); */
/*         ++histograms[shortKey * _ITEMS + item]; */
/*       } */
/*       barrier(CLK_LOCAL_MEM_FENCE); */

/* #if (__OPENCL_VERSION__ >= 200) */
/*       __attribute__((opencl_unroll_hint(_RADIX))) */
/* #endif */
/*       for (int i = 0; i < _RADIX; ++i) { */
/*         global_histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item] = */
/*             histograms[i * _ITEMS + item]; */
/*       } */
/*     } */

/*     // this kernel updates histograms with global sum after scan */
/*     __attribute__((vec_type_hint(KEY_TYPE))) */ 
/*     __kernel void merge( */
/*         // in */
/*         const global KEY_TYPE *sum, */
/*         // in-out */
/*         global KEY_TYPE *histogram) { */
/*       const KEY_TYPE s = sum[get_group_id(0)]; */
/*       const uint gid2 = get_global_id(0) << 1; */

/*       histogram[gid2] += s; */
/*       histogram[gid2 + 1] += s; */
/*     } */

/*     __attribute__((vec_type_hint(KEY_TYPE))) */ 
/*   __kernel void transpose( */
/*         // in */
/*         const global KEY_TYPE *keysIn, const global VALUE_TYPE *valuesIn, */
/*         const SIZE colCount, const SIZE rowCount, */
/*         // out */
/*         global KEY_TYPE *keysOut, global VALUE_TYPE *valuesOut, */
/*         // local */
/*         local KEY_TYPE *blockmat, local VALUE_TYPE *blockval) { */
/*       const int i0 = get_global_id(0) * _TILESIZE; // first row index */
/*       const int j = get_global_id(1);              // column index */
/*       const int j_local = get_local_id(1);         // local column index */

/* #if (__OPENCL_VERSION__ >= 200) */
/*       __attribute__((opencl_unroll_hint(_TILESIZE))) */
/* #endif */
/*       for (int i = 0; i < _TILESIZE; ++i) { */
/*         const int k = (i0 + i) * colCount + j; */
/*         blockmat[i * _TILESIZE + j_local] = keysIn[k]; */
/*         blockval[i * _TILESIZE + j_local] = valuesIn[k]; */
/*       } */

/*       barrier(CLK_LOCAL_MEM_FENCE); */

/*       const int j0 = get_group_id(1) * _TILESIZE; */
/* #if (__OPENCL_VERSION__ >= 200) */
/*       __attribute__((opencl_unroll_hint(_TILESIZE))) */
/* #endif */
/*       for (int i = 0; i < _TILESIZE; ++i) { */
/*         const int k = (j0 + i) * rowCount + i0 + j_local; */
/*         keysOut[k] = blockmat[j_local * _TILESIZE + i]; */
/*         valuesOut[k] = blockval[j_local * _TILESIZE + i]; */
/*       } */
/*     } */

/*     // see Blelloch 1990 */
/*     __attribute__((vec_type_hint(KEY_TYPE))) */ 
/*   __kernel void scan( */
/*         // in-out */
/*         global KEY_TYPE *input, */
/*         // out */
/*         global KEY_TYPE *sum, */
/*         // local */
/*         local KEY_TYPE *temp) { */
/*       const int gid2 = get_global_id(0) << 1; */
/*       const int group = get_group_id(0); */
/*       const int item = get_local_id(0); */
/*       const int n = get_local_size(0) << 1; */

/*       temp[2 * item] = input[gid2]; */
/*       temp[2 * item + 1] = input[gid2 + 1]; */

/*       // parallel prefix sum (algorithm of Blelloch 1990) */
/*       int decale = 1; */
/*       // up sweep phase */
/*       for (int d = n >> 1; d > 0; d >>= 1) { */
/*         barrier(CLK_LOCAL_MEM_FENCE); */
/*         if (item < d) { */
/*           const int ai = decale * ((item << 1) + 1) - 1; */
/*           const int bi = decale * ((item << 1) + 2) - 1; */
/*           temp[bi] += temp[ai]; */
/*         } */
/*         decale <<= 1; */
/*       } */

/*       // store the last element in the global sum vector */
/*       // (maybe used in the next step for constructing the global scan) */
/*       // clear the last element */
/*       if (item == 0) { */
/*         sum[group] = temp[n - 1]; */
/*         temp[n - 1] = 0; */
/*       } */

/*       // down sweep phase */
/*       for (int d = 1; d < n; d <<= 1) { */
/*         decale >>= 1; */
/*         barrier(CLK_LOCAL_MEM_FENCE); */
/*         if (item < d) { */
/*           const int ai = decale * ((item << 1) + 1) - 1; */
/*           const int bi = decale * ((item << 1) + 2) - 1; */
/*           const int t = temp[ai]; */
/*           temp[ai] = temp[bi]; */
/*           temp[bi] += t; */
/*         } */
/*       } */
/*       barrier(CLK_LOCAL_MEM_FENCE); */

/*       input[gid2] = temp[item << 1]; */
/*       input[gid2 + 1] = temp[(item << 1) + 1]; */
/*     } */

/*     __attribute__((vec_type_hint(KEY_TYPE))) */ 
/*   __kernel void reorder( */
/*         // in */
/*         const global KEY_TYPE *keysIn, const global INDEX_TYPE *valuesIn, */
/*         const SIZE length, const global KEY_TYPE *histograms, const int pass, */
/*         // out */
/*         global KEY_TYPE *keysOut, global INDEX_TYPE *valuesOut, */
/*         // local */
/*         local KEY_TYPE *local_histograms) { */
/*       const int item = get_local_id(0); */
/*       const int group = get_group_id(0); */

/*       const SIZE size = length / (_GROUPS * _ITEMS); */
/*       const SIZE start = get_global_id(0) * size; */

/* #if (__OPENCL_VERSION__ >= 200) */
/*       __attribute__((opencl_unroll_hint(_RADIX))) */
/* #endif */
/*       for (int i = 0; i < _RADIX; ++i) { */
/*         local_histograms[i * _ITEMS + item] = */
/*             histograms[i * _GROUPS * _ITEMS + _ITEMS * group + item]; */
/*       } */
/*       barrier(CLK_LOCAL_MEM_FENCE); */

/*       for (SIZE i = start; i < start + size; ++i) { */
/*         const KEY_TYPE key = keysIn[index(i, length)]; */
/*         const KEY_TYPE digit = ((key >> (pass * _BITS)) & (_RADIX - 1)); */
/*         const KEY_TYPE newPosition = local_histograms[digit * _ITEMS + item]; */

/*         local_histograms[digit * _ITEMS + item] = newPosition + 1; */

/*         // WRITE TO GLOBAL (slow) */
/*         keysOut[index(newPosition, length)] = key; */
/*         valuesOut[index(newPosition, length)] = */
/*             valuesIn[index(i, length)]; */
/*         // */
/*       } */
/*     })SOURCE"; */

template <
    int bits, int totalBits,

    typename _KeyType, typename _IndexType, typename _ValueType,

    size_t groups, // TODO move this to runtime?
    size_t items,  // todo same^

    size_t histosplit,

    bool transpose,  // todo same^
    size_t tileSize, // todo same^

    int passes = totalBits / bits,

    _KeyType radix = 1 << bits,
    _KeyType maxInt =
        (static_cast<_KeyType>(1) << (static_cast<_KeyType>(totalBits) - 1)) -
        static_cast<_KeyType>(1),

    typename = std::enable_if_t<
        std::is_integral<_KeyType>::value &&
        std::is_integral<_IndexType>::value &&
        (totalBits / 8 <= sizeof(_KeyType)) && is_power_of_2<groups>::value &&
        is_power_of_2<items>::value && (totalBits % bits == 0) &&
        ((groups * items * radix) % histosplit == 0)>>
class RadixSortBase {

  using HistogramType = _KeyType;

  const bc::context ctx;

  bc::command_queue queue;
  bc::program program;

  bc::kernel kernelTranspose;
  bc::kernel kernelHistogram;
  bc::kernel kernelScan;
  bc::kernel kernelMerge;
  bc::kernel kernelReorder;

  bc::vector<HistogramType> deviceHistograms;
  bc::vector<HistogramType> deviceSum;
  bc::vector<HistogramType> deviceTempSum;

  bc::vector<_KeyType> keysIn;
  bc::vector<_KeyType> keysOut;
  bc::vector<_ValueType> valuesIn;
  bc::vector<_ValueType> valuesOut;

  // helper vars
  bool bound = false;

  // transpose params
  bc::extents<2> localWorkItemsTranspose;
  size_t colCount;
  size_t rowCount;
  _IndexType tileSizeCalibrated;

  void _init(bc::vector<_KeyType> keys, bc::vector<_ValueType> values) {
    const size_t base_size = keys.size();
    const size_t rest = base_size % (groups * items);
    const size_t size =
        rest == 0 ? base_size : (base_size - rest + (groups * items));

    keysIn = std::move(keys);
    valuesIn = std::move(values);
    if (rest != 0) {
      bc::vector<_KeyType> pad(groups * items - rest, maxInt, queue);
      keysIn.insert(keysIn.end(), pad.begin(), pad.end(), queue);
      valuesIn.resize(size, queue);
    }
    keysOut = bc::vector<_KeyType>(size, ctx);
    valuesOut = bc::vector<_ValueType>(size, ctx);

    deviceHistograms = bc::vector<HistogramType>(radix * groups * items, ctx);
    deviceSum = bc::vector<HistogramType>(histosplit, ctx);
    deviceTempSum = bc::vector<HistogramType>(histosplit, ctx);

    if (transpose) {
      rowCount = groups * items;
      colCount = size / rowCount;
      tileSizeCalibrated =
          (rowCount % tileSize != 0 || colCount % tileSize != 0) ? 1 : tileSize;
      kernelTranspose.set_arg(4, tileSizeCalibrated);
      kernelTranspose.set_arg(
          7, sizeof(HistogramType) * tileSizeCalibrated * tileSizeCalibrated,
          nullptr);
      kernelTranspose.set_arg(
          8, sizeof(HistogramType) * tileSizeCalibrated * tileSizeCalibrated,
          nullptr);
      localWorkItemsTranspose = {1, tileSizeCalibrated};
    }

    kernelScan.set_arg(
        2,
        sizeof(HistogramType) *
            std::max(histosplit, radix * groups * items / histosplit),
        nullptr);

    kernelHistogram.set_arg(1, size);
    kernelHistogram.set_arg(3, deviceHistograms);
    kernelHistogram.set_arg(4, sizeof(HistogramType) * radix * items);

    kernelMerge.set_arg(0, deviceSum);
    kernelMerge.set_arg(1, deviceHistograms);

    kernelReorder.set_arg(2, size);
    kernelReorder.set_arg(3, deviceHistograms);
    kernelReorder.set_arg(7, sizeof(HistogramType) * radix * items, nullptr);

    this->bound = true;
  }

  template <bool back> void _transpose() {
    kernelTranspose.set_arg(0, keysIn);
    kernelTranspose.set_arg(1, valuesIn);
    if (back) {
      kernelTranspose.set_arg(2, rowCount);
      kernelTranspose.set_arg(3, colCount);
    } else {
      kernelTranspose.set_arg(2, colCount);
      kernelTranspose.set_arg(3, rowCount);
    }
    kernelTranspose.set_arg(5, keysOut);
    kernelTranspose.set_arg(6, valuesOut);

    auto globalWorkItemsTranspose =
        back ? bc::extents<2>{colCount / tileSizeCalibrated, rowCount}
             : bc::extents<2>{rowCount / tileSizeCalibrated, colCount};
    queue.enqueue_nd_range_kernel(kernelTranspose, {0, 0},
                                  globalWorkItemsTranspose,
                                  localWorkItemsTranspose);
    queue.finish();

    boost::swap(keysIn, keysOut);
    boost::swap(valuesIn, valuesOut);
  }

  void _histogram(int pass) {
    kernelHistogram.set_arg(0, keysIn);
    kernelHistogram.set_arg(2, pass);

    queue.enqueue_1d_range_kernel(kernelHistogram, 0, groups * items, items);
    queue.finish();
  }

  void _scan() {
    kernelScan.set_arg(0, deviceHistograms);
    kernelScan.set_arg(1, deviceSum);

    size_t totalLocalScanItems = radix * groups * items / 2;
    size_t localItems = totalLocalScanItems / histosplit;
    queue.enqueue_1d_range_kernel(kernelScan, 0, totalLocalScanItems,
                                  localItems);
    queue.finish();

    kernelScan.set_arg(0, deviceSum);
    kernelScan.set_arg(1, deviceTempSum);

    totalLocalScanItems = histosplit / 2;
    localItems = totalLocalScanItems;
    queue.enqueue_1d_range_kernel(kernelScan, 0, totalLocalScanItems,
                                  localItems);
    queue.finish();

    totalLocalScanItems = radix * groups * items / 2;
    localItems = totalLocalScanItems / histosplit;
    queue.enqueue_1d_range_kernel(kernelMerge, 0, totalLocalScanItems,
                                  localItems);
    queue.finish();
  }

  void _reorder(int pass) {
    kernelReorder.set_arg(0, keysIn);
    kernelReorder.set_arg(1, valuesIn);
    kernelReorder.set_arg(4, pass);
    kernelReorder.set_arg(5, keysOut);
    kernelReorder.set_arg(6, valuesOut);

    queue.enqueue_1d_range_kernel(kernelReorder, 0, groups * items, items);
    queue.finish();

    boost::swap(keysIn, keysOut);
    boost::swap(valuesIn, valuesOut);
  }

  void _sort(bc::vector<_KeyType> &keys, bc::vector<_ValueType> &values) {
    if (transpose) {
      _transpose<false>();
    }

    for (int pass = 0; pass < passes; ++pass) {
      _histogram(pass);
      _scan();
      _reorder(pass);
    }

    if (transpose) {
      _transpose<true>();
    }

    keys = keysIn;
    values = valuesIn;
  }

  void _recompileProgram() {
    std::stringstream opts;
    opts << Opt{"_RADIX", radix} << Opt{"_BITS", bits} << Opt{"_GROUPS", groups}
         << Opt{"_ITEMS", items} << Opt{"_TILESIZE", tileSize}
         << Opt{"KEY_TYPE", cl_type_name<_KeyType>::name()}
         //TODO
         << Opt{"VALUE_TYPE", "double"}
         << Opt{"INDEX_TYPE", cl_type_name<_IndexType>::name()}
         << (transpose ? " -DTRANSPOSE" : "");
    std::cout << source << std::endl;
    program.build_with_source(source, ctx, opts.str());
    // recreate kernels
    kernelHistogram = program.create_kernel("histogram");
    /* kernelScan = program.create_kernel("scan"); */
    /* kernelMerge = program.create_kernel("merge"); */
    /* kernelReorder = program.create_kernel("reorder"); */
    /* kernelTranspose = program.create_kernel("transpose"); */
  }

public:
  typedef _KeyType KeyType;
  typedef _IndexType IndexType;

  /**
   * @brief Construct Radix sort
   * @param ctx context
   * @param queue command_queue
   */
  RadixSortBase(bc::context ctx, bc::command_queue queue)
      : ctx(ctx), queue(queue) {
    // perform initial program compilation
    _recompileProgram();
  }

  /**
   * @brief Sort given iterable using radix sort algorithm.
   * @tparam rebind recreate permutation and all information needed to run sort
   * <p/>
   * @tparam Iter random access iterator
   * @param begin begin
   * @param end end
   */
  template <bool rebind = true> void sort(bc::vector<_KeyType> &keys, bc::vector<_ValueType> &values) {
    if (!bound || rebind) {
      _init(keys, values);
    }
    _sort(keys, values);
  }

  /**
   * @brief Max possible value of vector element for this radix sort instance
   * @return max value
   */
  inline constexpr KeyType maxValue() const noexcept { return maxInt; }
};

/**
 * @brief Radix sort implementation
 * @tparam bits size of radix in bits
 * @tparam totalBits size of key in bits
 * @tparam DataType data type for key
 * @tparam IndexType data type for permutation vector
 * @tparam groups global work items count for algorithm
 * @tparam items local work items count for algorithm
 * @tparam computePermutation whether or not to compute permutation of key
 * vector.
 * @tparam histosplit size of histogram local part for "scan" part of Radix Sort
 * @tparam transpose whether or not to perform transposition of key vector. This
 * may improve cache usage on some plaforms
 * @tparam tileSize size of tile for transpose step
 * @tparam enableProfiling whether or not to enable commands profiling. Only
 * kernel executions are profiled!
 */
template <int bits = 8, int totalBits = 32,

          typename KeyType = unsigned int, 
          typename IndexType = unsigned int,
          typename ValueType = double,

          size_t groups = 128, size_t items = 8,

          size_t histosplit = 512,

          bool transpose = true, size_t tileSize = 32>

using RadixSort = RadixSortBase<bits, totalBits, KeyType, IndexType, ValueType,
                                groups, items, histosplit,
                                transpose, tileSize>;

#endif
