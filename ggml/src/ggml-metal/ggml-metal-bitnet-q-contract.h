#pragma once

#include "ggml-metal-bitnet-q-config.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>

struct ggml_metal_bitnet_q_contract {
    const char * variant;
    const char * entry;
    int version;
    int rows;
    int groups;
    int grid_x;
    int threads_x;
};

static inline bool ggml_metal_bitnet_q_select(const char * variant, const char * rows,
        const char * groups, struct ggml_metal_bitnet_q_contract * result) {
    if (!result) {
        return false;
    }
    memset(result, 0, sizeof(*result));
    if (!variant) {
        const char * r = rows ? rows : "8";
        const char * g = groups ? groups : "2";
        if (strcmp(r, "4") == 0 && strcmp(g, "1") == 0) {
            variant = "stock4x1";
        } else if (strcmp(r, "8") == 0 && strcmp(g, "2") == 0) {
            variant = "c8x2";
        } else {
            return false;
        }
    }
    if (strcmp(variant, "stock4x1") == 0) {
        if ((rows && strcmp(rows, "4") != 0) || (groups && strcmp(groups, "1") != 0)) {
            return false;
        }
        result->variant = "stock4x1";
        result->entry = GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY);
        result->rows = GGML_BITNET_Q4_ROWS;
        result->groups = GGML_BITNET_Q4_NSG;
    } else if (strcmp(variant, "c8x2") == 0) {
        if ((rows && strcmp(rows, "8") != 0) || (groups && strcmp(groups, "2") != 0)) {
            return false;
        }
        result->variant = "c8x2";
        result->entry = GGML_BITNET_Q_STRINGIFY(GGML_BITNET_C8_ENTRY);
        result->rows = GGML_BITNET_C8_ROWS;
        result->groups = GGML_BITNET_C8_NSG;
    } else {
        return false;
    }
    result->version = GGML_BITNET_Q_CONFIG_VERSION;
    result->grid_x = GGML_BITNET_Q_NROWS / (result->rows * result->groups);
    result->threads_x = GGML_BITNET_Q_SIMD_WIDTH;
    return true;
}

static inline bool ggml_metal_bitnet_q_contract_valid(const struct ggml_metal_bitnet_q_contract * value) {
    struct ggml_metal_bitnet_q_contract expected;
    return value && value->variant && value->entry &&
        ggml_metal_bitnet_q_select(value->variant, NULL, NULL, &expected) &&
        strcmp(value->entry, expected.entry) == 0 && value->version == expected.version &&
        value->rows == expected.rows && value->groups == expected.groups &&
        value->grid_x == expected.grid_x && value->threads_x == expected.threads_x;
}

static inline bool ggml_metal_bitnet_q_launch_valid(const struct ggml_metal_bitnet_q_contract * value,
        int64_t nrows, int64_t ncols, int execution_width, int max_threads) {
    return ggml_metal_bitnet_q_contract_valid(value) && nrows == GGML_BITNET_Q_NROWS &&
        ncols == GGML_BITNET_Q_NCOLS && execution_width == value->threads_x &&
        max_threads >= value->threads_x * value->groups &&
        value->grid_x * value->rows * value->groups == nrows;
}

struct ggml_metal_bitnet_file_snapshot {
    char path[PATH_MAX];
    uint64_t device;
    uint64_t inode;
    uint64_t size;
    int64_t mtime_sec;
    int64_t mtime_nsec;
    int64_t ctime_sec;
    int64_t ctime_nsec;
};

static inline bool ggml_metal_bitnet_file_snapshot_get(const char * path,
        struct ggml_metal_bitnet_file_snapshot * result) {
    if (!path || !path[0] || !result) {
        return false;
    }
    memset(result, 0, sizeof(*result));
    struct stat info;
    if (!realpath(path, result->path) || stat(result->path, &info) != 0 ||
            !S_ISREG(info.st_mode) || info.st_size <= 0) {
        return false;
    }
    result->device = (uint64_t) info.st_dev;
    result->inode = (uint64_t) info.st_ino;
    result->size = (uint64_t) info.st_size;
#ifdef __APPLE__
    result->mtime_sec = info.st_mtimespec.tv_sec;
    result->mtime_nsec = info.st_mtimespec.tv_nsec;
    result->ctime_sec = info.st_ctimespec.tv_sec;
    result->ctime_nsec = info.st_ctimespec.tv_nsec;
#else
    result->mtime_sec = info.st_mtim.tv_sec;
    result->mtime_nsec = info.st_mtim.tv_nsec;
    result->ctime_sec = info.st_ctim.tv_sec;
    result->ctime_nsec = info.st_ctim.tv_nsec;
#endif
    return true;
}

static inline bool ggml_metal_bitnet_file_same(const struct ggml_metal_bitnet_file_snapshot * a,
        const struct ggml_metal_bitnet_file_snapshot * b) {
    return strcmp(a->path, b->path) == 0 && a->device == b->device && a->inode == b->inode &&
        a->size == b->size && a->mtime_sec == b->mtime_sec && a->mtime_nsec == b->mtime_nsec &&
        a->ctime_sec == b->ctime_sec && a->ctime_nsec == b->ctime_nsec;
}

static inline bool ggml_metal_bitnet_external_key(const struct ggml_metal_bitnet_file_snapshot * file,
        const char * entry, const char * contract, char * output, size_t capacity) {
    if (!file || !file->path[0] || !entry || !entry[0] || !contract || !output || capacity == 0) {
        return false;
    }
    const int n = snprintf(output, capacity,
        "bitnet-external-v2|%zu:%s|%zu:%s|%zu:%s|%llu:%llu:%llu:%lld:%lld:%lld:%lld",
        strlen(file->path), file->path, strlen(entry), entry, strlen(contract), contract,
        (unsigned long long) file->device, (unsigned long long) file->inode, (unsigned long long) file->size,
        (long long) file->mtime_sec, (long long) file->mtime_nsec, (long long) file->ctime_sec, (long long) file->ctime_nsec);
    return n >= 0 && (size_t) n < capacity;
}
