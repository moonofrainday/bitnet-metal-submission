#include "ggml-metal-bitnet-q-contract.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

static int checks = 0;
static void require(bool value, const char * message) {
    ++checks;
    if (!value) { throw std::runtime_error(message); }
}
static void write_file(const std::string & path, const std::string & content) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << content;
    require(bool(file), "write temporary file");
}
static std::string key(const ggml_metal_bitnet_file_snapshot & file,
        const char * entry = "entry", const char * contract = "contract-v1") {
    char result[2 * PATH_MAX];
    require(ggml_metal_bitnet_external_key(&file, entry, contract, result, sizeof(result)), "cache key generation");
    return result;
}
static void contracts() {
    ggml_metal_bitnet_q_contract value;
    require(ggml_metal_bitnet_q_select(nullptr, nullptr, nullptr, &value) &&
            std::string(value.variant) == "c8x2", "legacy default is C8x2");
    for (const auto & config : std::vector<std::array<const char *, 3>>{
            {"stock4x1", "4", "1"}, {"c8x2", "8", "2"}}) {
        require(ggml_metal_bitnet_q_select(config[0], config[1], config[2], &value), "explicit contract");
        require(ggml_metal_bitnet_q_contract_valid(&value), "canonical descriptor");
        require(ggml_metal_bitnet_q_launch_valid(&value, 2560, 2560, 32, 1024), "legal launch");
        ggml_metal_bitnet_q_contract inferred;
        require(ggml_metal_bitnet_q_select(nullptr, config[1], config[2], &inferred) &&
                std::string(inferred.entry) == value.entry, "exact legacy inference");
        require(ggml_metal_bitnet_q_select(config[0], nullptr, nullptr, &inferred) &&
                inferred.rows == value.rows && inferred.groups == value.groups, "explicit variant supplies dimensions");
        std::array<int, 2560> coverage{};
        for (int group = 0; group < value.grid_x; ++group) {
            for (int simd = 0; simd < value.groups; ++simd) {
                for (int row = 0; row < value.rows; ++row) {
                    const int index = (group * value.groups + simd) * value.rows + row;
                    require(index >= 0 && index < 2560, "row index bounds");
                    ++coverage[index];
                }
            }
        }
        for (int count : coverage) { require(count == 1, "every row written exactly once"); }
        auto forged = value;
        forged.entry = std::string(value.variant) == "stock4x1" ?
            GGML_BITNET_Q_STRINGIFY(GGML_BITNET_C8_ENTRY) : GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY);
        require(!ggml_metal_bitnet_q_contract_valid(&forged), "forged entry rejected");
        forged = value; ++forged.groups;
        require(!ggml_metal_bitnet_q_contract_valid(&forged), "forged group count rejected");
        forged = value; --forged.grid_x;
        require(!ggml_metal_bitnet_q_contract_valid(&forged), "forged grid rejected");
        forged = value; ++forged.version;
        require(!ggml_metal_bitnet_q_contract_valid(&forged), "forged version rejected");
        require(!ggml_metal_bitnet_q_launch_valid(&value, 2559, 2560, 32, 1024), "wrong row count rejected");
        require(!ggml_metal_bitnet_q_launch_valid(&value, 2560, 2559, 32, 1024), "wrong K rejected");
        require(!ggml_metal_bitnet_q_launch_valid(&value, 2560, 2560, 16, 1024), "wrong SIMD width rejected");
        require(!ggml_metal_bitnet_q_launch_valid(&value, 2560, 2560, 32, value.groups * 32 - 1), "thread limit rejected");
    }
    for (const auto & config : std::vector<std::array<const char *, 3>>{
            {nullptr,"4","2"}, {nullptr,"4","4"}, {nullptr,"8","1"}, {nullptr,"8","4"},
            {"stock4x1","4","2"}, {"stock4x1","8","2"}, {"c8x2","4","1"},
            {"unknown","4","1"}, {"","4","1"}, {nullptr,"04","1"}, {nullptr,"4","1x"},
            {nullptr,"4",nullptr}, {nullptr,"", "1"}, {"tq2_0_matvec_poc_c8x2","4","1"}}) {
        require(!ggml_metal_bitnet_q_select(config[0], config[1], config[2], &value), "unsupported variant/geometry rejected");
        require(value.entry == nullptr, "rejected selection has no pipeline entry");
    }
    require(!ggml_metal_bitnet_q_select(nullptr,nullptr,nullptr,nullptr), "null result rejected");
    require(!ggml_metal_bitnet_q_contract_valid(nullptr), "null descriptor rejected");
}
static void snapshots() {
    char pattern[] = "/tmp/bitnet-q-contract-XXXXXX";
    char * directory = mkdtemp(pattern);
    require(directory != nullptr, "temporary directory");
    const std::string root(directory), a = root + "/a.metallib", b = root + "/b.metallib", alias = root + "/alias.metallib";
    write_file(a, "file-a"); write_file(b, "file-b");
    require(symlink(a.c_str(), alias.c_str()) == 0, "symlink alias");
    ggml_metal_bitnet_file_snapshot original, same, other, changed;
    require(ggml_metal_bitnet_file_snapshot_get(a.c_str(), &original), "regular-file snapshot");
    require(ggml_metal_bitnet_file_snapshot_get(alias.c_str(), &same), "alias snapshot");
    require(ggml_metal_bitnet_file_same(&original, &same) && key(original) == key(same), "canonical aliases share identity");
    require(ggml_metal_bitnet_file_snapshot_get((root + "/./a.metallib").c_str(), &same) &&
            key(original) == key(same), "dot path alias canonicalized");
    require(ggml_metal_bitnet_file_snapshot_get(b.c_str(), &other) && key(original) != key(other), "different files have different keys");
    require(key(original,"other-entry") != key(original), "entry binds cache identity");
    require(key(original,"entry","other-contract") != key(original), "contract binds cache identity");
    auto artificial = original; ++artificial.mtime_nsec;
    require(key(original) != key(artificial), "nanosecond mtime binds cache identity");
    artificial = original; ++artificial.ctime_nsec;
    require(key(original) != key(artificial), "nanosecond ctime binds cache identity");
    artificial = original; ++artificial.inode;
    require(key(original) != key(artificial), "inode binds cache identity");
    const struct timespec times[] = {{0, UTIME_OMIT}, {original.mtime_sec, original.mtime_nsec}};
    write_file(a, "edited");
    require(utimensat(AT_FDCWD, a.c_str(), times, 0) == 0, "restore prior mtime after same-size edit");
    require(ggml_metal_bitnet_file_snapshot_get(a.c_str(), &changed), "edited snapshot");
    require(changed.size == original.size && changed.mtime_sec == original.mtime_sec &&
            changed.mtime_nsec == original.mtime_nsec, "same-size edit with preserved mtime");
    require(!ggml_metal_bitnet_file_same(&original, &changed) && key(original) != key(changed), "ctime detects same-size preserved-mtime edit");
    write_file(b, "edited");
    require(rename(b.c_str(), a.c_str()) == 0, "replace file atomically");
    require(ggml_metal_bitnet_file_snapshot_get(a.c_str(), &other) && other.inode != changed.inode &&
            key(other) != key(changed), "atomic replacement invalidates old key");
    require(unlink(a.c_str()) == 0, "delete previously valid path");
    require(!ggml_metal_bitnet_file_snapshot_get(a.c_str(), &same), "deleted file cannot reuse cached identity");
    require(!ggml_metal_bitnet_file_snapshot_get(alias.c_str(), &same), "dangling alias rejected");
    require(!ggml_metal_bitnet_file_snapshot_get(root.c_str(), &same), "directory rejected");
    write_file(b, "");
    require(!ggml_metal_bitnet_file_snapshot_get(b.c_str(), &same), "empty file rejected");
    char tiny[4];
    require(!ggml_metal_bitnet_external_key(&original,"entry","contract",tiny,sizeof(tiny)), "truncated key rejected");
    require(!ggml_metal_bitnet_external_key(nullptr,"entry","contract",tiny,sizeof(tiny)), "null identity rejected");
    unlink(alias.c_str()); unlink(b.c_str()); rmdir(root.c_str());
}
int main() {
    try {
        contracts(); snapshots();
        std::printf("Q_KERNEL_CONTRACT_CPU_PASS checks=%d GPU_executed=0 device_executed=0\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"Q_KERNEL_CONTRACT_CPU_FAIL checks=%d error=%s\n", checks,error.what());
        return 1;
    }
}
