#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__AVX2__) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <limits>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
struct Mapping {
    const uint8_t * data = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int fd = -1;
#endif
};

uint16_t load_u16(const uint8_t * data) {
    uint16_t value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

float load_f32(const uint8_t * data) {
    float value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

float half_to_float(uint16_t bits) {
    const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
    const uint32_t exponent = (bits >> 10) & 0x1Fu;
    const uint32_t fraction = bits & 0x3FFu;
    if (exponent == 0) {
        return sign * std::ldexp(static_cast<float>(fraction), -24);
    }
    if (exponent == 31) {
        return fraction == 0 ? sign * std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    }
    return sign * std::ldexp(1.0f + static_cast<float>(fraction) / 1024.0f, static_cast<int>(exponent) - 15);
}

void unpack_q4k_scales(const uint8_t * packed, int block, int & scale, int & minimum) {
    if (block < 4) {
        scale = packed[block] & 0x3F;
        minimum = packed[block + 4] & 0x3F;
    } else {
        scale = (packed[block + 4] & 0x0F) | ((packed[block - 4] >> 6) << 4);
        minimum = (packed[block + 4] >> 4) | ((packed[block] >> 6) << 4);
    }
}

float dot_q4k(const uint8_t * data, const float * vector, uint64_t width) {
    float result = 0.0f;
    const uint64_t blocks = width / 256;
    for (uint64_t block_index = 0; block_index < blocks; ++block_index) {
        const uint8_t * block = data + block_index * 144;
        const float d = half_to_float(load_u16(block));
        const float dmin = half_to_float(load_u16(block + 2));
        const uint8_t * scales = block + 4;
        const uint8_t * qs = block + 16;
        const uint64_t base = block_index * 256;
        for (int span = 0; span < 4; ++span) {
            int sc1, m1, sc2, m2;
            unpack_q4k_scales(scales, 2 * span, sc1, m1);
            unpack_q4k_scales(scales, 2 * span + 1, sc2, m2);
            const float scale1 = d * sc1;
            const float scale2 = d * sc2;
            const float min1 = dmin * m1;
            const float min2 = dmin * m2;
            const uint8_t * q = qs + span * 32;
            for (int j = 0; j < 32; ++j) {
                result += vector[base + span * 64 + j] * (scale1 * (q[j] & 0x0F) - min1);
                result += vector[base + span * 64 + 32 + j] * (scale2 * (q[j] >> 4) - min2);
            }
        }
    }
    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
float dot_f32_avx2(const uint8_t * data, const float * vector, uint64_t width) {
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= width; i += 8) {
        const __m256 a = _mm256_loadu_ps(reinterpret_cast<const float *>(data + i * 4));
        const __m256 b = _mm256_loadu_ps(vector + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < width; ++i) result += load_f32(data + i * 4) * vector[i];
    return result;
#else
    float result = 0.0f;
    for (uint64_t i = 0; i < width; ++i) result += load_f32(data + i * 4) * vector[i];
    return result;
#endif
}

float dot_bf16(const uint8_t * data, const float * vector, uint64_t width) {
    float result = 0.0f;
    for (uint64_t i = 0; i < width; ++i) {
        result += std::bit_cast<float>(static_cast<uint32_t>(load_u16(data + i * 2)) << 16) * vector[i];
    }
    return result;
}

float dot_q4_0(const uint8_t * data, const float * vector, uint64_t width) {
    float result = 0.0f;
    for (uint64_t block_index = 0; block_index < width / 32; ++block_index) {
        const uint8_t * block = data + block_index * 18;
        const float d = half_to_float(load_u16(block));
        const uint8_t * q = block + 2;
        const uint64_t base = block_index * 32;
        alignas(32) float decoded[32];
        for (int j = 0; j < 16; ++j) {
            decoded[j] = (static_cast<int>(q[j] & 0x0F) - 8) * d;
            decoded[j + 16] = (static_cast<int>(q[j] >> 4) - 8) * d;
        }
#if defined(__AVX2__)
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_load_ps(decoded + j), _mm256_loadu_ps(vector + base + j)));
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, sum);
        result += lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
#else
        for (int j = 0; j < 32; ++j) result += decoded[j] * vector[base + j];
#endif
    }
    return result;
}

float dot_q4_1(const uint8_t * data, const float * vector, uint64_t width) {
    float result = 0.0f;
    for (uint64_t block_index = 0; block_index < width / 32; ++block_index) {
        const uint8_t * block = data + block_index * 20;
        const float d = half_to_float(load_u16(block));
        const float m = half_to_float(load_u16(block + 2));
        const uint8_t * q = block + 4;
        const uint64_t base = block_index * 32;
        alignas(32) float decoded[32];
        for (int j = 0; j < 16; ++j) {
            decoded[j] = static_cast<float>(q[j] & 0x0F) * d + m;
            decoded[j + 16] = static_cast<float>(q[j] >> 4) * d + m;
        }
#if defined(__AVX2__)
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_load_ps(decoded + j), _mm256_loadu_ps(vector + base + j)));
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, sum);
        result += lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
#else
        for (int j = 0; j < 32; ++j) result += decoded[j] * vector[base + j];
#endif
    }
    return result;
}

float dot_q5_0(const uint8_t * data, const float * vector, uint64_t width) {
    float result = 0.0f;
    for (uint64_t block_index = 0; block_index < width / 32; ++block_index) {
        const uint8_t * block = data + block_index * 22;
        const float d = half_to_float(load_u16(block));
        const uint32_t high = static_cast<uint32_t>(block[2]) | static_cast<uint32_t>(block[3]) << 8 |
                              static_cast<uint32_t>(block[4]) << 16 | static_cast<uint32_t>(block[5]) << 24;
        const uint8_t * q = block + 6;
        const uint64_t base = block_index * 32;
        for (int j = 0; j < 16; ++j) {
            result += vector[base + j] * ((static_cast<int>((q[j] & 0x0F) | (((high >> j) & 1) << 4)) - 16) * d);
            result += vector[base + 16 + j] * ((static_cast<int>((q[j] >> 4) | (((high >> (j + 16)) & 1) << 4)) - 16) * d);
        }
    }
    return result;
}

float dot_q8_0(const uint8_t * data, const float * vector, uint64_t width) {
    float result = 0.0f;
    for (uint64_t block_index = 0; block_index < width / 32; ++block_index) {
        const uint8_t * block = data + block_index * 34;
        const float d = half_to_float(load_u16(block));
        const int8_t * q = reinterpret_cast<const int8_t *>(block + 2);
        const uint64_t base = block_index * 32;
        for (int j = 0; j < 32; ++j) result += vector[base + j] * (d * q[j]);
    }
    return result;
}

float dot_q6k(const uint8_t * data, const float * vector, uint64_t width) {
    float result = 0.0f;
    for (uint64_t block_index = 0; block_index < width / 256; ++block_index) {
        const uint8_t * block = data + block_index * 210;
        const uint8_t * ql = block;
        const uint8_t * qh = block + 128;
        const int8_t * scales = reinterpret_cast<const int8_t *>(block + 192);
        const float d = half_to_float(load_u16(block + 208));
        const uint64_t base = block_index * 256;
        for (int half = 0; half < 2; ++half) {
            const uint8_t * qlh = ql + 64 * half;
            const uint8_t * qhh = qh + 32 * half;
            const int8_t * sch = scales + 8 * half;
            for (int l = 0; l < 32; ++l) {
                const int group = l / 16;
                const int q1 = (qlh[l] & 0x0F) | ((qhh[l] & 3) << 4);
                const int q2 = (qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4);
                const int q3 = (qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4);
                const int q4 = (qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4);
                const uint64_t out = base + half * 128;
                result += vector[out + l] * (d * sch[group] * (q1 - 32));
                result += vector[out + l + 32] * (d * sch[group + 2] * (q2 - 32));
                result += vector[out + l + 64] * (d * sch[group + 4] * (q3 - 32));
                result += vector[out + l + 96] * (d * sch[group + 6] * (q4 - 32));
            }
        }
    }
    return result;
}
}

extern "C" {
void * lowram_open(const char * path, char * error, size_t error_size) {
    auto * mapping = new Mapping();
#ifdef _WIN32
    mapping->file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (mapping->file == INVALID_HANDLE_VALUE) { std::snprintf(error, error_size, "cannot open model"); delete mapping; return nullptr; }
    LARGE_INTEGER size;
    GetFileSizeEx(mapping->file, &size);
    mapping->size = static_cast<size_t>(size.QuadPart);
    mapping->mapping = CreateFileMappingA(mapping->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    mapping->data = static_cast<const uint8_t *>(MapViewOfFile(mapping->mapping, FILE_MAP_READ, 0, 0, 0));
#else
    mapping->fd = open(path, O_RDONLY);
    if (mapping->fd < 0) { std::snprintf(error, error_size, "cannot open model"); delete mapping; return nullptr; }
    struct stat info{};
    if (fstat(mapping->fd, &info) != 0) { std::snprintf(error, error_size, "cannot stat model"); close(mapping->fd); delete mapping; return nullptr; }
    mapping->size = static_cast<size_t>(info.st_size);
    mapping->data = static_cast<const uint8_t *>(mmap(nullptr, mapping->size, PROT_READ, MAP_PRIVATE, mapping->fd, 0));
    if (mapping->data == MAP_FAILED) { std::snprintf(error, error_size, "cannot mmap model"); close(mapping->fd); delete mapping; return nullptr; }
#endif
    if (mapping->data == nullptr) { std::snprintf(error, error_size, "cannot map model"); delete mapping; return nullptr; }
    return mapping;
}

void lowram_close(void * handle) {
    auto * mapping = static_cast<Mapping *>(handle);
    if (!mapping) return;
#ifdef _WIN32
    UnmapViewOfFile(mapping->data); CloseHandle(mapping->mapping); CloseHandle(mapping->file);
#else
    munmap(const_cast<uint8_t *>(mapping->data), mapping->size); close(mapping->fd);
#endif
    delete mapping;
}

int lowram_matvec(void * handle, uint64_t offset, uint32_t type, uint64_t input_width, uint64_t output_width, const float * vector, float * output) {
    auto * mapping = static_cast<Mapping *>(handle);
    if (!mapping || !vector || !output || input_width == 0 || output_width == 0) return 1;
    const size_t bytes_per_column =         type == 0 ? input_width * 4 : type == 1 ? input_width * 2 : type == 30 ? input_width * 2 :
        type == 2 ? (input_width / 32) * 18 : type == 3 ? (input_width / 32) * 20 :
        type == 6 ? (input_width / 32) * 22 :
 type == 8 ? (input_width / 32) * 34 :
        type == 12 ? (input_width / 256) * 144 : type == 14 ? (input_width / 256) * 210 : 0;
    if (!bytes_per_column || offset > mapping->size || output_width > (mapping->size - offset) / bytes_per_column) return 2;
    if (type != 0 && type != 1 && type != 2 && type != 3 && type != 6 && type != 8 && type != 12 && type != 14 && type != 30) return 4;
    const uint8_t * base = mapping->data + offset;
#ifndef _WIN32
#if defined(POSIX_MADV_RANDOM)
    (void)posix_madvise(const_cast<uint8_t *>(base), static_cast<size_t>(output_width * bytes_per_column), POSIX_MADV_RANDOM);
#endif
#endif
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint64_t column = 0; column < output_width; ++column) {
        const uint8_t * column_data = base + column * bytes_per_column;
        float value = 0.0f;
        if (type == 0) {
#if defined(__AVX2__)
            value = dot_f32_avx2(column_data, vector, input_width);
#else
            for (uint64_t i = 0; i < input_width; ++i) value += load_f32(column_data + i * 4) * vector[i];
#endif
        } else if (type == 1) {
            for (uint64_t i = 0; i < input_width; ++i) value += half_to_float(load_u16(column_data + i * 2)) * vector[i];
        } else if (type == 30) {
            value = dot_bf16(column_data, vector, input_width);
        } else if (type == 2 || type == 3 || type == 6 || type == 8 || type == 12 || type == 14) {
            if (type == 2) value = dot_q4_0(column_data, vector, input_width);
            else if (type == 3) value = dot_q4_1(column_data, vector, input_width);
            else if (type == 6) value = dot_q5_0(column_data, vector, input_width);
            else if (type == 8) value = dot_q8_0(column_data, vector, input_width);
            else if (type == 12) value = dot_q4k(column_data, vector, input_width);
            else value = dot_q6k(column_data, vector, input_width);
        } else {
            // Type validity is checked before entering the parallel region.
            value = 0.0f;
        }
        output[column] = value;
    }
    return 0;
}
}
